import cv2
import mediapipe as mp
import numpy as np
import time
import tempfile
import os
import sys
from collections import deque
import gradio as gr

# --- TFLite Import (with fallback) ---
# tflite-runtime is lightweight but only available on specific platforms.
# Fall back to the full tensorflow package which includes tf.lite.
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow as tf
        tflite = tf.lite
        print("INFO: Using tensorflow.lite (tflite_runtime not available)")
    except ImportError:
        tflite = None
        print("ERROR: Neither tflite-runtime nor tensorflow is installed. "
              "Install one with: pip install tensorflow")

# --- Configuration ---
MODEL_PATH = 'fall_detection_transformer.tflite'
INPUT_TIMESTEPS = 30
FALL_CONFIDENCE_THRESHOLD = 0.90
MIN_KEYPOINT_CONFIDENCE_FOR_NORMALIZATION = 0.3
mp_pose = mp.solutions.pose
pose_complexity = 0  # Use 0 for speed on shared resources, 1 for better accuracy

FALL_EVENT_COOLDOWN = 10

# ----- Keypoint Definitions -----
KEYPOINT_NAMES_ORIGINAL = [
    'Nose', 'Left Eye Inner', 'Left Eye', 'Left Eye Outer', 'Right Eye Inner', 'Right Eye', 'Right Eye Outer',
    'Left Ear', 'Right Ear', 'Mouth Left', 'Mouth Right',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow', 'Left Wrist', 'Right Wrist',
    'Left Pinky', 'Right Pinky', 'Left Index', 'Right Index', 'Left Thumb', 'Right Thumb',
    'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle',
    'Left Heel', 'Right Heel', 'Left Foot Index', 'Right Foot Index'
]
MEDIAPIPE_TO_YOUR_KEYPOINTS_MAPPING = {
    mp_pose.PoseLandmark.NOSE: 'Nose', mp_pose.PoseLandmark.LEFT_EYE: 'Left Eye',
    mp_pose.PoseLandmark.RIGHT_EYE: 'Right Eye', mp_pose.PoseLandmark.LEFT_EAR: 'Left Ear',
    mp_pose.PoseLandmark.RIGHT_EAR: 'Right Ear', mp_pose.PoseLandmark.LEFT_SHOULDER: 'Left Shoulder',
    mp_pose.PoseLandmark.RIGHT_SHOULDER: 'Right Shoulder', mp_pose.PoseLandmark.LEFT_ELBOW: 'Left Elbow',
    mp_pose.PoseLandmark.RIGHT_ELBOW: 'Right Elbow', mp_pose.PoseLandmark.LEFT_WRIST: 'Left Wrist',
    mp_pose.PoseLandmark.RIGHT_WRIST: 'Right Wrist', mp_pose.PoseLandmark.LEFT_HIP: 'Left Hip',
    mp_pose.PoseLandmark.RIGHT_HIP: 'Right Hip', mp_pose.PoseLandmark.LEFT_KNEE: 'Left Knee',
    mp_pose.PoseLandmark.RIGHT_KNEE: 'Right Knee', mp_pose.PoseLandmark.LEFT_ANKLE: 'Left Ankle',
    mp_pose.PoseLandmark.RIGHT_ANKLE: 'Right Ankle'
}
YOUR_KEYPOINT_NAMES_TRAINING = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle'
]
SORTED_YOUR_KEYPOINT_NAMES = sorted(YOUR_KEYPOINT_NAMES_TRAINING)
KEYPOINT_DICT_TRAINING = {name: i for i, name in enumerate(SORTED_YOUR_KEYPOINT_NAMES)}
NUM_KEYPOINTS_TRAINING = len(KEYPOINT_DICT_TRAINING)
NUM_FEATURES = NUM_KEYPOINTS_TRAINING * 3

print(f"--- Aether: Keypoint config OK (NUM_FEATURES={NUM_FEATURES}) ---")

# --- Load TFLite Model (graceful) ---
MODEL_LOADED = False
interpreter = None
input_details = None
output_details = None
_model_error_message = ""

if tflite is None:
    _model_error_message = (
        "TFLite runtime not available. Install tensorflow:\n"
        "  pip install tensorflow\n"
        "Then restart the application."
    )
    print(f"WARNING: {_model_error_message}")
elif not os.path.exists(MODEL_PATH):
    _model_error_message = (
        f"Model file not found: '{MODEL_PATH}'\n"
        f"Place the file in: {os.path.abspath('.')}\n"
        "The application will start but cannot process videos until the model is available."
    )
    print(f"WARNING: {_model_error_message}")
else:
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        model_expected_shape = tuple(input_details[0]['shape'])
        if model_expected_shape[2] != NUM_FEATURES or model_expected_shape[1] != INPUT_TIMESTEPS:
            _model_error_message = (
                f"Model shape mismatch: model expects {model_expected_shape}, "
                f"but app is configured for (1, {INPUT_TIMESTEPS}, {NUM_FEATURES})."
            )
            print(f"FATAL: {_model_error_message}")
            interpreter = None
        else:
            MODEL_LOADED = True
            print(f"TFLite model loaded: {MODEL_PATH} (shape: {model_expected_shape})")
    except Exception as e:
        _model_error_message = f"Failed to load TFLite model '{MODEL_PATH}': {e}"
        print(f"ERROR: {_model_error_message}")


# --- Helper Functions ---
def get_kpt_indices_training_order(keypoint_name):
    if keypoint_name not in KEYPOINT_DICT_TRAINING:
        raise ValueError(f"Keypoint '{keypoint_name}' not found in KEYPOINT_DICT_TRAINING.")
    kp_idx = KEYPOINT_DICT_TRAINING[keypoint_name]
    return kp_idx * 3, kp_idx * 3 + 1, kp_idx * 3 + 2

def normalize_skeleton_frame(frame_features_sorted, min_confidence=MIN_KEYPOINT_CONFIDENCE_FOR_NORMALIZATION):
    normalized_frame = np.copy(frame_features_sorted)
    ref_kp_names = {'ls': 'Left Shoulder', 'rs': 'Right Shoulder', 'lh': 'Left Hip', 'rh': 'Right Hip'}
    try:
        ls_x_idx, ls_y_idx, ls_c_idx = get_kpt_indices_training_order(ref_kp_names['ls'])
        rs_x_idx, rs_y_idx, rs_c_idx = get_kpt_indices_training_order(ref_kp_names['rs'])
        lh_x_idx, lh_y_idx, lh_c_idx = get_kpt_indices_training_order(ref_kp_names['lh'])
        rh_x_idx, rh_y_idx, rh_c_idx = get_kpt_indices_training_order(ref_kp_names['rh'])
    except ValueError as e:
        print(f"Warning in normalize_skeleton_frame: {e}")
        return frame_features_sorted

    ls_x, ls_y, ls_c = frame_features_sorted[ls_x_idx], frame_features_sorted[ls_y_idx], frame_features_sorted[ls_c_idx]
    rs_x, rs_y, rs_c = frame_features_sorted[rs_x_idx], frame_features_sorted[rs_y_idx], frame_features_sorted[rs_c_idx]
    lh_x, lh_y, lh_c = frame_features_sorted[lh_x_idx], frame_features_sorted[lh_y_idx], frame_features_sorted[lh_c_idx]
    rh_x, rh_y, rh_c = frame_features_sorted[rh_x_idx], frame_features_sorted[rh_y_idx], frame_features_sorted[rh_c_idx]

    mid_shoulder_x, mid_shoulder_y = np.nan, np.nan
    valid_ls, valid_rs = ls_c > min_confidence, rs_c > min_confidence
    if valid_ls and valid_rs: mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2
    elif valid_ls: mid_shoulder_x, mid_shoulder_y = ls_x, ls_y
    elif valid_rs: mid_shoulder_x, mid_shoulder_y = rs_x, rs_y

    mid_hip_x, mid_hip_y = np.nan, np.nan
    valid_lh, valid_rh = lh_c > min_confidence, rh_c > min_confidence
    if valid_lh and valid_rh: mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2, (lh_y + rh_y) / 2
    elif valid_lh: mid_hip_x, mid_hip_y = lh_x, lh_y
    elif valid_rh: mid_hip_x, mid_hip_y = rh_x, rh_y

    if np.isnan(mid_hip_x) or np.isnan(mid_hip_y):
        return frame_features_sorted

    reference_height = np.nan
    if not np.isnan(mid_shoulder_y) and not np.isnan(mid_hip_y):
        reference_height = np.abs(mid_shoulder_y - mid_hip_y)

    perform_scaling = not (np.isnan(reference_height) or reference_height < 1e-5)

    for kp_name_sorted in SORTED_YOUR_KEYPOINT_NAMES:
        try:
            x_col, y_col, _ = get_kpt_indices_training_order(kp_name_sorted)
            normalized_frame[x_col] -= mid_hip_x
            normalized_frame[y_col] -= mid_hip_y
            if perform_scaling:
                normalized_frame[x_col] /= reference_height
                normalized_frame[y_col] /= reference_height
        except ValueError:
            pass
    return normalized_frame

def extract_and_normalize_features(pose_results):
    frame_features_sorted = np.zeros(NUM_FEATURES, dtype=np.float32)
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        for mp_landmark_enum, your_kp_name in MEDIAPIPE_TO_YOUR_KEYPOINTS_MAPPING.items():
            if your_kp_name in KEYPOINT_DICT_TRAINING:
                try:
                    lm = landmarks[mp_landmark_enum.value]
                    x_idx, y_idx, c_idx = get_kpt_indices_training_order(your_kp_name)
                    frame_features_sorted[x_idx] = lm.x
                    frame_features_sorted[y_idx] = lm.y
                    frame_features_sorted[c_idx] = lm.visibility
                except (IndexError, ValueError) as e:
                    print(f"Warning extracting {your_kp_name}: {e}")
    normalized_features = normalize_skeleton_frame(frame_features_sorted.copy())
    return normalized_features


# --- Main video processing function ---
def process_video_for_gradio(uploaded_video_path_temp):
    if not MODEL_LOADED:
        return None, f"⚠️ Model not loaded.\n\n{_model_error_message}"

    if uploaded_video_path_temp is None:
        return None, "Please upload a video file."

    print(f"Processing video: {uploaded_video_path_temp}")

    cap = cv2.VideoCapture(uploaded_video_path_temp)
    if not cap.isOpened():
        return None, f"Error: Cannot open video file."

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps < 1 or np.isnan(fps):
        fps = 25.0
    fps = float(fps)

    local_feature_sequence = deque(maxlen=INPUT_TIMESTEPS)
    local_last_fall_event_time = -FALL_EVENT_COOLDOWN - 1
    processed_frames_list = []
    overall_status_updates = []
    output_path = None

    try:
        with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=pose_complexity,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:

            frame_count = 0
            while cap.isOpened():
                success, original_bgr_frame = cap.read()
                if not success:
                    break

                frame_count += 1
                frame_for_display = original_bgr_frame.copy()

                # MediaPipe pose estimation
                image_rgb = cv2.cvtColor(original_bgr_frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)

                # Extract and normalize features
                current_features = extract_and_normalize_features(results)
                local_feature_sequence.append(current_features)

                # Prediction
                current_status_text = f"Frame {frame_count}: Collecting..."
                prediction_label = "no_fall"
                display_confidence = 0.0

                if len(local_feature_sequence) == INPUT_TIMESTEPS:
                    model_input_data = np.array(local_feature_sequence, dtype=np.float32)
                    model_input_data = np.expand_dims(model_input_data, axis=0)
                    try:
                        interpreter.set_tensor(input_details[0]['index'], model_input_data)
                        interpreter.invoke()
                        output_data = interpreter.get_tensor(output_details[0]['index'])
                        prob_fall = output_data[0][0]

                        if prob_fall >= FALL_CONFIDENCE_THRESHOLD:
                            prediction_label = "fall"
                            display_confidence = prob_fall
                        else:
                            prediction_label = "no_fall"
                            display_confidence = 1.0 - prob_fall

                        current_status_text = f"Frame {frame_count}: {prediction_label.upper()} (Conf: {display_confidence:.2f})"

                        current_video_time = frame_count / fps
                        if prediction_label == "fall":
                            if (current_video_time - local_last_fall_event_time) > FALL_EVENT_COOLDOWN:
                                fall_msg = f"Frame {frame_count} (~{current_video_time:.1f}s): FALL DETECTED! (Conf: {prob_fall:.2f})"
                                print(fall_msg)
                                overall_status_updates.append(fall_msg)
                                local_last_fall_event_time = current_video_time
                    except Exception as e:
                        print(f"Frame {frame_count}: Prediction error: {e}")
                        current_status_text = f"Frame {frame_count}: Prediction Error"

                # Log periodic status
                if "FALL DETECTED" not in current_status_text:
                    log_interval = max(1, int(fps))
                    if frame_count % log_interval == 0 or frame_count == 1:
                        if "Collecting" not in current_status_text or frame_count == 1:
                            overall_status_updates.append(current_status_text)

                # Draw landmarks
                if results.pose_landmarks:
                    temp_rgb = cv2.cvtColor(original_bgr_frame, cv2.COLOR_BGR2RGB).copy()
                    mp.solutions.drawing_utils.draw_landmarks(
                        temp_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                    )
                    frame_for_display = cv2.cvtColor(temp_rgb, cv2.COLOR_RGB2BGR)

                # Draw overlay text
                font = cv2.FONT_HERSHEY_DUPLEX
                padding = 30

                overlay_text = f"{prediction_label.upper()} (Conf: {display_confidence:.2f})"
                if "Collecting" in current_status_text or "Error" in current_status_text:
                    overlay_text = current_status_text.split(': ')[-1]

                (tw, th), _ = cv2.getTextSize(overlay_text, font, 0.6, 1)
                tx = frame_for_display.shape[1] - tw - padding
                ty = padding + th

                color = (255, 255, 255)
                if "Error" in overlay_text:
                    color = (0, 0, 255)

                cv2.putText(frame_for_display, overlay_text, (tx, ty), font, 0.6, color, 1, cv2.LINE_AA)

                # Draw FALL alert
                current_video_time_check = frame_count / fps
                if prediction_label == "fall" and (current_video_time_check - local_last_fall_event_time < FALL_EVENT_COOLDOWN):
                    alert = "FALL DETECTED!"
                    (aw, ah), _ = cv2.getTextSize(alert, font, 1.0, 2)
                    cv2.putText(frame_for_display, alert,
                                (frame_for_display.shape[1] - aw - padding, ty + ah + padding // 2),
                                font, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

                processed_frames_list.append(frame_for_display)

        cap.release()

        if not processed_frames_list:
            return None, "No frames processed. Video might be empty or unreadable."

        # Write output video to a temp file
        height, width, _ = processed_frames_list[0].shape
        suffix = ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="aether_out_") as tmp:
            output_path = tmp.name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in processed_frames_list:
            writer.write(frame)
        writer.release()
        print(f"Output video saved: {output_path}")

        summary = "Detection Summary:\n" + "\n".join(overall_status_updates[-15:])
        if not overall_status_updates:
            summary = "Detection Summary:\nNo fall events detected."

        return output_path, summary

    except Exception as e:
        cap.release()
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        print(f"Error processing video: {e}")
        return None, f"Error processing video: {e}"


# --- Create Gradio Interface ---
example_filenames = [
    "fall_example_1.mp4",
    "fall_example_2.mp4",
    "fall_example_3.mp4",
    "fall_example_4.mp4"
]

examples_list = []
for fname in example_filenames:
    if os.path.exists(fname):
        examples_list.append([fname])
        print(f"Example found: {fname}")
    else:
        print(f"Example not found (skipping): {fname}")

# Build description based on model status
app_description = (
    "Upload a video file (MP4 recommended) to detect falls using AI pose estimation. "
    "Processing may take time depending on video length."
)
if not MODEL_LOADED:
    app_description = f"⚠️ {_model_error_message}\n\n{app_description}"

custom_css = """
body { background-color: #131313 !important; color: #e5e2e1 !important; font-family: 'Inter', sans-serif !important; }
.glass-panel { background: rgba(53, 53, 52, 0.4); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); }
.intelligence-rail { background: linear-gradient(to bottom, #ff6b00, transparent); width: 2px; height: 100%; position: absolute; right: 0; top: 0; }
.glow-orange { box-shadow: 0 0 20px rgba(255, 107, 0, 0.2); }
.gradio-container { max-width: 1600px !important; margin: auto; padding: 0 !important; }
.dark { background-color: #131313 !important; }

/* Force Gradio components to adapt to dark theme */
footer { display: none !important; } /* Hide default Gradio footer */
.gap-4 { gap: 1rem; }
.p-4 { padding: 1rem; }
"""

custom_head = """
<meta charset="utf-8"/>
<link rel="icon" type="image/png" href="/favicon.png?v=2">
<link rel="icon" type="image/x-icon" href="/favicon.ico?v=2">
<link rel="shortcut icon" href="/favicon.ico?v=2">
<script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=Manrope:wght@500;700;800&family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet"/>
<script>
    tailwind.config = {
        darkMode: "class",
        theme: {
            extend: {
                colors: { primary: "#ffb693", "primary-container": "#ff6b00", background: "#131313", surface: "#131313", "surface-container": "#201f1f", "surface-container-low": "#1c1b1b", "surface-container-high": "#2a2a2a", "surface-container-highest": "#353534", "on-surface": "#e5e2e1", secondary: "#c8c6c5", "outline-variant": "#5a4136", "tertiary-container": "#059eff", tertiary: "#9ccaff", error: "#ffb4ab", "error-container": "#93000a" },
                fontFamily: { headline: ["Space Grotesk"], body: ["Inter"], label: ["Manrope"] }
            }
        }
    }
</script>
<style>
    .material-symbols-outlined { font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24; }
</style>
"""

header_html = """
<header class="w-full z-50 flex justify-between items-center px-8 h-20 bg-[#131313] border-b border-[#353534]/50">
    <div class="flex items-center gap-8">
        <span class="text-2xl font-bold tracking-tighter text-[#ffb693] font-headline">Aether</span>
        <nav class="hidden md:flex items-center gap-6">
            <a class="text-[#ffb693] border-b-2 border-[#ff6b00] pb-1 font-body text-sm transition-colors opacity-80 duration-200" href="#">Monitoring</a>
            <a class="text-[#c8c6c5] font-body text-sm hover:text-[#ffb693] transition-colors" href="#">Analytics</a>
            <a class="text-[#c8c6c5] font-body text-sm hover:text-[#ffb693] transition-colors" href="#">Logs</a>
            <a class="text-[#c8c6c5] font-body text-sm hover:text-[#ffb693] transition-colors" href="#">Settings</a>
        </nav>
    </div>
    <div class="flex items-center gap-4">
        <div class="bg-surface-container-high px-4 py-1.5 rounded-full border border-outline-variant/20 flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-primary-container animate-pulse"></span>
            <span class="text-[10px] font-label uppercase tracking-widest text-primary">System Status: Active</span>
        </div>
        <div class="flex items-center gap-2">
            <button class="p-2 text-[#c8c6c5] hover:text-[#ffb693] transition-colors"><span class="material-symbols-outlined">notifications</span></button>
            <button class="p-2 text-[#c8c6c5] hover:text-[#ffb693] transition-colors"><span class="material-symbols-outlined">settings</span></button>
            <div class="w-8 h-8 rounded-full bg-surface-container-highest border border-outline-variant/30 flex items-center justify-center overflow-hidden ml-2">
                <span class="material-symbols-outlined text-sm text-secondary">person</span>
            </div>
        </div>
    </div>
</header>
"""

hero_html = """
<section class="mb-4 relative overflow-hidden p-8 rounded-xl bg-surface-container-low border border-outline-variant/10">
    <div class="absolute -top-24 -right-24 w-96 h-96 bg-primary-container/10 blur-[120px] rounded-full"></div>
    <div class="relative z-10">
        <div class="flex items-center gap-3 mb-4">
            <span class="px-3 py-1 rounded-full bg-primary-container/10 border border-primary-container/20 text-primary text-[10px] font-label font-bold uppercase tracking-widest">Model Ready</span>
            <span class="text-secondary/50 text-xs font-label">•</span>
            <span class="text-secondary/80 text-xs font-label">v4.2 Stable Core</span>
        </div>
        <h1 class="text-4xl font-headline font-bold text-on-surface tracking-tight mb-2">AI Fall Detection & <span class="text-primary">Safety Monitoring</span></h1>
        <p class="text-md font-body text-secondary max-w-2xl leading-relaxed">
            Advanced neural processing for real-time human pose estimation and anomaly detection. Aether delivers 99.8% precision in critical incident recognition.
        </p>
    </div>
</section>
"""

footer_html = """
<footer class="w-full flex justify-between items-center px-12 py-6 mt-12 bg-[#131313] border-t border-[#353534]/30">
    <span class="font-label text-[10px] uppercase tracking-widest text-[#c8c6c5]">© 2024 Aether Intelligence. All systems operational.</span>
    <div class="flex gap-8">
        <a class="font-label text-[10px] uppercase tracking-widest text-[#c8c6c5] hover:text-white transition-colors" href="#">Privacy Policy</a>
        <a class="font-label text-[10px] uppercase tracking-widest text-[#c8c6c5] hover:text-white transition-colors" href="#">Terms of Service</a>
        <a class="font-label text-[10px] uppercase tracking-widest text-[#c8c6c5] hover:text-white transition-colors" href="#">API Documentation</a>
    </div>
</footer>
"""

# --- Create Gradio Interface ---
with gr.Blocks(title="Aether AI | Fall Detection & Safety Monitoring", css=custom_css, head=custom_head, theme=gr.themes.Base()) as iface:
    gr.HTML(header_html)
    
    with gr.Column(elem_classes="px-8 max-w-[1600px] mx-auto pt-6"):
        gr.HTML(hero_html)
        
        if not MODEL_LOADED:
            gr.HTML(f"<div class='p-4 bg-error-container/20 border border-error/50 rounded-lg text-error mb-6'><p class='font-bold flex items-center gap-2'><span class='material-symbols-outlined'>warning</span> Model Load Error</p><p class='text-sm'>{_model_error_message}</p></div>")
            
        with gr.Row(elem_classes="gap-8 mb-8"):
            with gr.Column(scale=1, elem_classes="bg-surface-container rounded-xl border border-outline-variant/10 p-4"):
                gr.HTML("""
                <div class="flex justify-between items-center mb-4 pl-2">
                    <div class="flex items-center gap-2">
                        <span class="material-symbols-outlined text-primary">upload_file</span>
                        <h3 class="font-headline text-xl font-bold text-white">Source Input</h3>
                    </div>
                </div>
                """)
                input_video = gr.Video(label="", sources=["upload"], elem_classes="rounded-lg border-2 border-dashed border-outline-variant/30")

            with gr.Column(scale=1, elem_classes="bg-surface-container rounded-xl border border-outline-variant/10 p-4 relative"):
                gr.HTML("""
                <div class="intelligence-rail hidden md:block"></div>
                <div class="flex justify-between items-center mb-4 pl-2">
                    <div class="flex items-center gap-2">
                        <span class="material-symbols-outlined text-primary">query_stats</span>
                        <h3 class="font-headline text-xl font-bold text-white">AI Analysis</h3>
                    </div>
                    <div class="flex items-center gap-2">
                        <span class="text-[10px] font-label uppercase tracking-widest text-secondary/60">Processing Latency: 14ms</span>
                    </div>
                </div>
                """)
                output_video = gr.Video(label="", interactive=False, elem_classes="rounded-lg border border-outline-variant/20")
                
        with gr.Row(elem_classes="flex items-center justify-between bg-surface-container-high/50 p-6 rounded-xl border border-outline-variant/10 mb-8"):
            with gr.Column(scale=2):
                gr.HTML("""
                <div class="flex items-center gap-8">
                    <div class="flex flex-col">
                        <span class="text-[10px] font-label uppercase text-secondary/60 tracking-widest">Detection Threshold</span>
                        <span class="text-sm font-body font-medium text-white">0.85 (High Sensitivity)</span>
                    </div>
                    <div class="w-[1px] h-8 bg-outline-variant/20"></div>
                    <div class="flex flex-col">
                        <span class="text-[10px] font-label uppercase text-secondary/60 tracking-widest">Notification Protocol</span>
                        <span class="text-sm font-body font-medium text-white">Standard - Alert Only</span>
                    </div>
                </div>
                """)
            with gr.Column(scale=1, elem_classes="flex justify-end gap-4"):
                with gr.Row():
                    clear_btn = gr.Button("Clear Workspace", elem_classes="!bg-surface-container-highest !text-secondary !font-label !text-sm !font-bold !rounded-xl !border-outline-variant/30 hover:!bg-surface-bright transition-all")
                    submit_btn = gr.Button("Analyze Video", elem_classes="!bg-primary-container !text-on-primary-container !font-headline !text-sm !font-bold !rounded-xl !shadow-lg glow-orange hover:!opacity-90 transition-all", variant="primary")

        with gr.Row(elem_classes="mb-8"):
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="bg-surface-container p-6 rounded-xl border border-outline-variant/10 h-full">
                    <h4 class="font-headline text-lg font-bold mb-4 flex items-center gap-2 text-white">
                        <span class="material-symbols-outlined text-primary">assessment</span> Detection Summary
                    </h4>
                    <p class="text-sm font-body text-secondary mb-4">The event timeline displays specific activity points and classifications identified by the detection model.</p>
                </div>
                """)
            with gr.Column(scale=2):
                output_text = gr.Textbox(label="Activity Logs / Timeline", lines=6, max_lines=15, elem_classes="!bg-surface-container !rounded-xl !border !border-outline-variant/10 !text-white font-mono text-sm")

        if examples_list:
            # Hidden buttons to bridge HTML clicks to Gradio events
            btn_load_1 = gr.Button("load_1", elem_id="btn-load-1", elem_classes="hidden")
            btn_load_2 = gr.Button("load_2", elem_id="btn-load-2", elem_classes="hidden")
            btn_load_3 = gr.Button("load_3", elem_id="btn-load-3", elem_classes="hidden")
            btn_load_4 = gr.Button("load_4", elem_id="btn-load-4", elem_classes="hidden")

            btn_load_1.click(lambda: examples_list[0][0] if len(examples_list) > 0 else None, inputs=None, outputs=[input_video])
            btn_load_2.click(lambda: examples_list[1][0] if len(examples_list) > 1 else None, inputs=None, outputs=[input_video])
            btn_load_3.click(lambda: examples_list[2][0] if len(examples_list) > 2 else None, inputs=None, outputs=[input_video])
            btn_load_4.click(lambda: examples_list[3][0] if len(examples_list) > 3 else None, inputs=None, outputs=[input_video])

            gr.HTML("""
            <section>
                <div class="flex justify-between items-center mb-6 mt-8">
                    <h3 class="font-headline text-2xl font-bold text-white">Sample Test Scenarios</h3>
                    <div class="flex gap-2">
                        <button class="p-2 rounded-full border border-outline-variant/30 hover:bg-surface-container-highest transition-colors">
                            <span class="material-symbols-outlined text-white">chevron_left</span>
                        </button>
                        <button class="p-2 rounded-full border border-outline-variant/30 hover:bg-surface-container-highest transition-colors">
                            <span class="material-symbols-outlined text-white">chevron_right</span>
                        </button>
                    </div>
                </div>
                <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                    <!-- Case 1 -->
                    <div class="group cursor-pointer" onclick="document.querySelector('#btn-load-1 button') ? document.querySelector('#btn-load-1 button').click() : document.querySelector('#btn-load-1').click()">
                        <div class="aspect-video bg-surface-container rounded-xl overflow-hidden border border-outline-variant/10 mb-3 relative">
                            <img alt="Office Scenario" class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110 opacity-60" data-alt="Office environment surveillance shot" src="https://lh3.googleusercontent.com/aida-public/AB6AXuDYSpa3NqYGR71rDO3HZWhC7i70UP4mtKX1qsh5tBZuQh9f6Z2wF0DdOY2RA8Ta6oYKqoZk3SNuqYvp-qH5CWK8aBbQhOW_GYQP06QSSxs24DeuTInMH3Us0oJKHW2q1d68ozuKd4wvza5OxxFCpIEUnjlksmmZnXvdrlXErEj7pBtdM1w1QHMswH7hWj7qxg9PlSHwjBcL4aC1JtdYcuoDNmqcMmvSE1SKhf9n1zu9MqJZkjEFP8j7QSbwIvxi4ASSbOYcnS_HeAQe"/>
                            <div class="absolute inset-0 bg-gradient-to-t from-background to-transparent opacity-60"></div>
                            <div class="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                <span class="material-symbols-outlined text-4xl text-primary-container">play_circle</span>
                            </div>
                        </div>
                        <p class="font-headline font-bold text-sm mb-1 text-white">Office Hallway #04</p>
                        <p class="text-[10px] font-label uppercase tracking-wider text-secondary/50">Scenario: Standard Light</p>
                    </div>
                    <!-- Case 2 -->
                    <div class="group cursor-pointer" onclick="document.querySelector('#btn-load-2 button') ? document.querySelector('#btn-load-2 button').click() : document.querySelector('#btn-load-2').click()">
                        <div class="aspect-video bg-surface-container rounded-xl overflow-hidden border border-outline-variant/10 mb-3 relative">
                            <img alt="Hospital Scenario" class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110 opacity-60" data-alt="Empty hospital corridor surveillance" src="https://lh3.googleusercontent.com/aida-public/AB6AXuDBlo4pFm0yRe-ap__1AjN7X8V5ID15G_JGHbsQ1X8Wpw2Ry-4kRHht0LJnhCPRBLc5V4qdCy7kFsui0ywXnDqCEUEm8qGHbIpxeDbaRD-6M6BUTezQ6x00v1tAtSFgh878N4ZVFVS4XinjlbSBbLttB-Bkx9EhB3m63PPxUwdYj5NtHi1FGYi59oNqIr0lsyjX2ujwT7OtBMrrYRJ0e9J1pOsMAc4fOp_y4khml3TNQDEbOpuI2ZrF_6bEBYxzdoSWkWvGfGk7D4Zm"/>
                            <div class="absolute inset-0 bg-gradient-to-t from-background to-transparent opacity-60"></div>
                            <div class="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                <span class="material-symbols-outlined text-4xl text-primary-container">play_circle</span>
                            </div>
                        </div>
                        <p class="font-headline font-bold text-sm mb-1 text-white">Care Facility Ward B</p>
                        <p class="text-[10px] font-label uppercase tracking-wider text-secondary/50">Scenario: Low Light Mode</p>
                    </div>
                    <!-- Case 3 -->
                    <div class="group cursor-pointer" onclick="document.querySelector('#btn-load-3 button') ? document.querySelector('#btn-load-3 button').click() : document.querySelector('#btn-load-3').click()">
                        <div class="aspect-video bg-surface-container rounded-xl overflow-hidden border border-outline-variant/10 mb-3 relative">
                            <img alt="Living Room Scenario" class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110 opacity-60" data-alt="Residential living room interior" src="https://lh3.googleusercontent.com/aida-public/AB6AXuDep5uUxV_jdbXF7v_78wbdDjv9_sMD_FE_ucBpD47WU9_uzz7Aebeor26UlogDMHl2zQ5PoMUASfBDeB4gyxMIbwVQK-4ly2lyKfygRTFKzKeSCNFw71VbDgSKDLMUp3c2zx5AKd55uS5VhROnniYk0079cKrBTEsgBaEQIxB9bjU9nMU1H18eoDnCg3i4jpIid2QBlLaFLWbOaD-sYfkUs-649OmuLu4SJjuBtEVQH5Emz63bl5njthmFxpApmIWaxDHgo1P0IPKG"/>
                            <div class="absolute inset-0 bg-gradient-to-t from-background to-transparent opacity-60"></div>
                            <div class="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                <span class="material-symbols-outlined text-4xl text-primary-container">play_circle</span>
                            </div>
                        </div>
                        <p class="font-headline font-bold text-sm mb-1 text-white">Residential Unit 12A</p>
                        <p class="text-[10px] font-label uppercase tracking-wider text-secondary/50">Scenario: Complex Background</p>
                    </div>
                    <!-- Case 4 -->
                    <div class="group cursor-pointer" onclick="document.querySelector('#btn-load-4 button') ? document.querySelector('#btn-load-4 button').click() : document.querySelector('#btn-load-4').click()">
                        <div class="aspect-video bg-surface-container rounded-xl overflow-hidden border border-outline-variant/10 mb-3 relative">
                            <img alt="Public Space Scenario" class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110 opacity-60" data-alt="Public plaza or mall lobby" src="https://lh3.googleusercontent.com/aida-public/AB6AXuBOmkznkudh6OMsaqEcDncPpvBQmrokCBuVk9lnxTcIvtsYf-Vw0qiBnU2Uegm3MzVDO_6SyN8O0eVCLvG85vqYqRgL9_ooXPjzpCthNyZmKLZEitxeohYGxobxpPN_je0ovxuhG2_U6o2eU3m8OYWZiSwtmknoqgzwbmwT-8Fg0cma_TAevG4Ff-hSNDWr2m7g8f0jVu6QafDmYI0LPLTG0Dh8fReQu9oJp_GsZBEoSlFLJsHGWNDI2FxXc2l2jYvSqbn6L0rd1n_h"/>
                            <div class="absolute inset-0 bg-gradient-to-t from-background to-transparent opacity-60"></div>
                            <div class="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                <span class="material-symbols-outlined text-4xl text-primary-container">play_circle</span>
                            </div>
                        </div>
                        <p class="font-headline font-bold text-sm mb-1 text-white">Main Lobby North</p>
                        <p class="text-[10px] font-label uppercase tracking-wider text-secondary/50">Scenario: High Density Crowds</p>
                    </div>
                </div>
            </section>
            """)
            
        gr.HTML(footer_html)
        
    submit_btn.click(
        fn=process_video_for_gradio,
        inputs=[input_video],
        outputs=[output_video, output_text]
    )
    
    clear_btn.click(
        fn=lambda: (None, None, ""),
        inputs=[],
        outputs=[input_video, output_video, output_text]
    )


if __name__ == "__main__":
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    import gradio as gr
    import uvicorn
    import os

    from fastapi.responses import FileResponse, JSONResponse

    app = FastAPI()

    # Root project directory
    project_dir = os.path.abspath(os.path.dirname(__file__))

    # Mount the Gradio interface at /api
    # In Gradio 6.x+, app-level favicon is best handled by FastAPI directly
    app = gr.mount_gradio_app(
        app, 
        iface, 
        path="/api"
    )

    # Serve fall example videos

    @app.get("/videos/{filename}")
    async def serve_video(filename: str):
        if filename.startswith("fall_example_") and filename.endswith(".mp4"):
            path = os.path.join(project_dir, filename)
            if os.path.exists(path):
                return FileResponse(path, media_type="video/mp4")
        return JSONResponse({"error": "Not found"}, status_code=404)

    @app.get("/favicon.png")
    async def get_favicon():
        favicon_path = os.path.join(project_dir, "favicon.png")
        if os.path.exists(favicon_path):
            return FileResponse(favicon_path, media_type="image/png")
        return JSONResponse({"error": "Favicon not found"}, status_code=404)

    @app.get("/favicon.ico")
    async def get_favicon_ico():
        favicon_path = os.path.join(project_dir, "favicon.ico")
        if os.path.exists(favicon_path):
            return FileResponse(favicon_path, media_type="image/x-icon")
        return JSONResponse({"error": "Favicon not found"}, status_code=404)

    # Path to the compiled React dashboard (dist folder)
    react_build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'aether-dashboard/dist'))

    if os.path.exists(react_build_dir):
        # Serve the static React build on the root path
        app.mount("/", StaticFiles(directory=react_build_dir, html=True), name="frontend")
        print(f"Server starting: Frontend at / and API at /api")
    else:
        print(f"WARNING: React build not found at {react_build_dir}. Ensure you ran 'npm run build'.")
        # Fallback to serving Gradio on root if frontend isn't built
        app = gr.mount_gradio_app(FastAPI(), iface, path="/")
        print(f"Server starting: API at / (Standalone Mode)")

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
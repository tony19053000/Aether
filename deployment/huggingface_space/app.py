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
    local_last_fall_event_time = 0.0
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

iface = gr.Interface(
    fn=process_video_for_gradio,
    inputs=gr.Video(label="Upload Video File (.mp4)", sources=["upload"]),
    outputs=[
        gr.Video(label="Processed Video with Detections"),
        gr.Textbox(label="Detection Summary")
    ],
    title="Aether: AI-Based Fall Detection",
    description=app_description,
    examples=examples_list if examples_list else None,
    flagging_mode="never",
    cache_examples=False
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port)
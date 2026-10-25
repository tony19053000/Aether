const $ = (id) => document.getElementById(id);
const aboutDialog = $('about-dialog');
$('about-open').onclick = () => {
  aboutDialog.showModal();
  aboutDialog.scrollTop = 0;
  document.body.style.overflow = 'hidden';
};
$('about-close').onclick = () => aboutDialog.close();
aboutDialog.addEventListener('close', () => {
  document.body.style.overflow = '';
  $('about-open').focus();
});
const dropZone = $('drop-zone');
const preview = document.createElement('video');
preview.controls = true;
preview.hidden = true;
dropZone.after(preview);
const status = document.createElement('p');
status.className = 'text-sm text-primary mt-4';
status.setAttribute('role', 'status');
dropZone.parentElement.append(status);
let selected = null;
let previewUrl = null;
let busy = false;
let summary = '';
const outputVideo = $('output-video');
function setAnalyzing(value) {
  $('analysis-loading').hidden = !value;
  $('output-placeholder').setAttribute('aria-busy', String(value));
  $('analysis-placeholder-text').textContent = value ? 'Analyzing your video…' : 'Ready when you are';
}

function select(file) {
  if (busy || !file) return;
  if (file.type && !file.type.startsWith('video/')) {
    status.textContent = 'Please select a video file.';
    return;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  selected = file;
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  dropZone.hidden = true;
  status.textContent = file.name;
  resetResults();
}

function resetResults() {
  setAnalyzing(false);
  $('output-video').pause();
  $('output-video').removeAttribute('src');
  $('output-video').load();
  $('output-video').hidden = true;
  $('output-placeholder').hidden = false;
  $('confidence-score').textContent = '—';
  $('confidence-bar').style.width = '0%';
  $('result-label').textContent = 'Awaiting analysis';
  $('result-state').textContent = 'Select a video and click Analyze Video.';
  $('processing-time').textContent = 'Ready to analyze';
  $('activity-log').replaceChildren();
  $('export-log').disabled = true;
  summary = '';
}

$('select-file').onclick = () => $('video-file').click();
$('video-file').onchange = (event) => select(event.target.files[0]);
for (const event of ['dragover', 'drop']) dropZone.addEventListener(event, (e) => e.preventDefault());
dropZone.addEventListener('drop', (e) => select(e.dataTransfer.files[0]));
$('clear-workspace').onclick = () => {
  if (busy) return;
  selected = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  preview.pause();
  preview.removeAttribute('src');
  preview.load();
  preview.hidden = true;
  dropZone.hidden = false;
  $('video-file').value = '';
  status.textContent = '';
  resetResults();
};

function setBusy(value) {
  busy = value;
  for (const id of ['analyze-video', 'clear-workspace', 'select-file', 'video-file']) $(id).disabled = value;
  for (const card of document.querySelectorAll('[data-sample]')) card.setAttribute('aria-disabled', String(value));
}

for (const card of document.querySelectorAll('[data-sample]')) {
  const load = async () => {
    if (busy) return;
    setBusy(true);
    status.textContent = 'Loading sample video…';
    try {
      const name = `fall_example_${card.dataset.sample}.mp4`;
      const response = await fetch(`/videos/${name}`);
      if (!response.ok) throw new Error('Unable to load sample video.');
      const file = new File([await response.blob()], name, {type: 'video/mp4'});
      setBusy(false);
      select(file);
      preview.scrollIntoView({behavior: 'smooth', block: 'center'});
    } catch (error) { status.textContent = error.message; }
    finally { setBusy(false); }
  };
  card.onclick = load;
  card.onkeydown = (e) => { if (['Enter', ' '].includes(e.key)) { e.preventDefault(); load(); } };
}

$('analyze-video').onclick = async () => {
  if (busy) return;
  if (!selected) { status.textContent = 'Select a video or sample scenario first.'; return; }
  resetResults();
  setBusy(true);
  status.textContent = 'Analyzing video…';
  setAnalyzing(true);
  $('result-label').textContent = 'Analysis in progress';
  $('result-state').textContent = 'Reviewing motion in your video…';
  $('processing-time').textContent = 'Processing…';
  try {
    const body = new FormData();
    body.append('video', selected);
    const response = await fetch('/analyze', {method: 'POST', body});
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Analysis failed. Please try another video.');
    summary = result.summary;
    $('output-video').src = result.video_url;
    $('output-video').hidden = false;
    $('output-placeholder').hidden = true;
    $('processing-time').textContent = `Processed in ${result.elapsed}s`;
    const falls = result.events.filter(event => event.fall);
    const scored = (falls.length ? falls : result.events).filter(event => event.confidence !== null);
    const confidence = scored.length ? Math.max(...scored.map(event => event.confidence)) * 100 : null;
    $('confidence-score').textContent = confidence === null ? '—' : `${confidence.toFixed(1)}%`;
    $('confidence-bar').style.width = `${confidence || 0}%`;
    $('result-label').textContent = falls.length ? 'Fall detected' : 'No fall detected';
    $('result-state').textContent = 'Analysis complete';
    for (const event of result.events) {
      const row = document.createElement('tr');
      row.className = 'border-t border-outline-variant/10';
      const frame = event.description.match(/^Frame (\d+)/);
      const seconds = event.seconds;
      const timestamp = Number.isFinite(seconds)
        ? `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${(seconds % 60).toFixed(1).padStart(4, '0')}`
        : frame ? `Frame ${frame[1]}` : '—';
      const observation = /error/i.test(event.description) ? 'Prediction unavailable'
        : event.fall ? 'Fall detected' : event.confidence === null ? 'Gathering motion data…' : 'No fall detected';
      const values = [timestamp, observation,
        event.confidence === null ? '—' : `${(event.confidence * 100).toFixed(1)}%`];
      for (const value of values) { const cell = document.createElement('td'); cell.className = 'py-4 pr-4'; cell.textContent = value; row.append(cell); }
      $('activity-log').append(row);
    }
    $('export-log').disabled = false;
    status.textContent = 'Analysis complete. Your annotated video is ready.';
    // Muted inline playback permits autoplay after the asynchronous analysis.
    outputVideo.muted = true;
    try { await outputVideo.play(); }
    catch { status.textContent = 'Analysis complete. Use the video’s play control to start playback.'; }
  } catch (error) {
    status.textContent = error.message;
    $('processing-time').textContent = 'Analysis failed';
    $('result-label').textContent = 'Analysis unavailable';
    $('result-state').textContent = 'Try again or choose another video.';
  } finally { setBusy(false); setAnalyzing(false); }
};

$('export-log').onclick = () => {
  const url = URL.createObjectURL(new Blob([summary], {type: 'text/plain'}));
  const link = document.createElement('a');
  link.href = url; link.download = 'aether-analysis.txt'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

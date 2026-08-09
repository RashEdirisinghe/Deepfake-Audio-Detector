// Detector page: file selection, submission to the backend, and rendering results.
// The backend endpoint is expected at POST /predict, matching the existing Flask app.
// Response is read defensively — optional fields (compression provenance, calibration,
// stability heatmaps) render only when the backend actually returns them, so this page
// works against a minimal backend today and a fuller one later without changes.

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('uploadForm');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('audioFile');
  const fileChip = document.getElementById('fileChip');
  const fileChipName = document.getElementById('fileChipName');
  const fileChipClear = document.getElementById('fileChipClear');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const formError = document.getElementById('formError');

  const placeholderPanel = document.getElementById('placeholderPanel');
  const spinnerPanel = document.getElementById('spinnerPanel');
  const spinnerText = document.getElementById('spinnerText');
  const results = document.getElementById('results');

  const verdictBadge = document.getElementById('verdictBadge');
  const verdictLabel = document.getElementById('verdictLabel');
  const confValue = document.getElementById('confValue');
  const confFill = document.getElementById('confFill');
  const verdictFile = document.getElementById('verdictFile');
  const heatmapImage = document.getElementById('heatmapImage');

  const provenanceCard = document.getElementById('provenanceCard');
  const provenanceValue = document.getElementById('provenanceValue');
  const calibrationCard = document.getElementById('calibrationCard');
  const calibrationValue = document.getElementById('calibrationValue');
  const stabilityCard = document.getElementById('stabilityCard');
  const stabilityClean = document.getElementById('stabilityClean');
  const stabilityCompressed = document.getElementById('stabilityCompressed');

  let selectedFile = null;

  // ---- File selection ----

  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });

  ['dragenter', 'dragover'].forEach(evt =>
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add('drag-over'); })
  );
  ['dragleave', 'drop'].forEach(evt =>
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove('drag-over'); })
  );
  dropzone.addEventListener('drop', (e) => {
    const dropped = e.dataTransfer.files && e.dataTransfer.files[0];
    if (dropped) setSelectedFile(dropped);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) setSelectedFile(fileInput.files[0]);
  });

  fileChipClear.addEventListener('click', (e) => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = '';
    fileChip.classList.remove('visible');
    analyzeBtn.disabled = true;
    hideError();
  });

  function setSelectedFile(file) {
    if (!file.name.toLowerCase().endsWith('.wav')) {
      showError('Please choose a .wav file — other formats aren\u2019t supported yet.');
      return;
    }
    hideError();
    selectedFile = file;
    fileChipName.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
    fileChip.classList.add('visible');
    analyzeBtn.disabled = false;
  }

  function showError(msg) {
    formError.textContent = msg;
    formError.classList.add('visible');
  }
  function hideError() {
    formError.classList.remove('visible');
  }

  // ---- Submit ----

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!selectedFile) { showError('Choose a .wav file first.'); return; }

    hideError();
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analysing…';
    placeholderPanel.style.display = 'none';
    results.classList.remove('visible');
    spinnerPanel.classList.add('visible');
    spinnerText.textContent = 'Running the model…';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('/predict', { method: 'POST', body: formData });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}`);
      }

      const result = await response.json();

      if (result.error) {
        throw new Error(result.error);
      }

      renderResults(result);
    } catch (err) {
      spinnerPanel.classList.remove('visible');
      placeholderPanel.style.display = '';
      showError(
        err && err.message
          ? `Analysis failed: ${err.message}`
          : 'Analysis failed. Check that the backend is running and try again.'
      );
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = 'Analyse audio';
    }
  });

  // ---- Render ----

  function renderResults(result) {
    spinnerPanel.classList.remove('visible');

    // Verdict
    const isReal = String(result.prediction).toLowerCase() === 'real';
    verdictBadge.classList.remove('real', 'fake');
    verdictBadge.classList.add(isReal ? 'real' : 'fake');
    verdictLabel.textContent = isReal ? 'REAL' : 'SYNTHETIC';

    // Confidence
    const confidence = Number(result.confidence);
    const confDisplay = Number.isFinite(confidence) ? confidence : null;
    confValue.textContent = confDisplay !== null ? `${confDisplay}%` : '—';
    confFill.style.width = confDisplay !== null ? `${Math.min(100, Math.max(0, confDisplay))}%` : '0%';

    verdictFile.textContent = selectedFile ? selectedFile.name : '';

    // Heatmap (required-ish — hide the card gracefully if absent)
    const heatmapCard = heatmapImage.closest('.card');
    if (result.heatmap_url) {
      heatmapImage.src = `${result.heatmap_url}?t=${Date.now()}`;
      heatmapCard.style.display = '';
    } else {
      heatmapCard.style.display = 'none';
    }

    // Optional: compression provenance
    const provenance = result.compression_provenance || result.compression || result.provenance;
    if (provenance) {
      provenanceValue.textContent = provenance;
      provenanceCard.style.display = '';
    } else {
      provenanceCard.style.display = 'none';
    }

    // Optional: calibration
    const calibration = result.calibration ?? result.calibration_score ?? result.reliability;
    if (calibration !== undefined && calibration !== null) {
      calibrationValue.textContent =
        typeof calibration === 'number' ? `${calibration}%` : String(calibration);
      calibrationCard.style.display = '';
    } else {
      calibrationCard.style.display = 'none';
    }

    // Optional: explanation stability (clean vs compressed heatmaps)
    const cleanUrl = result.stability_clean_url || result.heatmap_clean_url;
    const compressedUrl = result.stability_compressed_url || result.heatmap_compressed_url;
    if (cleanUrl && compressedUrl) {
      stabilityClean.src = `${cleanUrl}?t=${Date.now()}`;
      stabilityCompressed.src = `${compressedUrl}?t=${Date.now()}`;
      stabilityCard.style.display = '';
    } else {
      stabilityCard.style.display = 'none';
    }

    results.classList.add('visible');
  }
});
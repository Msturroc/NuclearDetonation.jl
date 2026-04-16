// All API calls to the Julia backend

export async function fetchStatus() {
  const resp = await fetch('/api/status');
  return resp.json();
}

export async function startSimulation(params) {
  const resp = await fetch('/api/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.error || 'Failed to start simulation');
  }
  return resp.json();
}

export async function loadDataset(dataset) {
  const resp = await fetch('/api/load-dataset', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset }),
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || 'Failed to load dataset');
  return data;
}

export async function fetchERA5Bounds() {
  const resp = await fetch('/api/era5-bounds');
  if (!resp.ok) return null;
  return resp.json();
}

export async function loadARLFromPath(path) {
  const resp = await fetch('/api/load-arl', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || 'Failed to load ARL data');
  return data;
}

export async function uploadARLFiles(files) {
  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file, file.name);
  }
  const resp = await fetch('/api/upload-arl', { method: 'POST', body: formData });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || 'Upload failed');
  return data;
}

export async function fetchPrediction({ site, date, hour, release_duration, release_height }) {
  const resp = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ site, date, hour, release_duration, release_height }),
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || 'Prediction failed');
  return data;
}

export async function fetchObservations() {
  const resp = await fetch('/api/observations');
  if (!resp.ok) throw new Error('No observations');
  return resp.json();
}

export async function fetchAnimationLevels() {
  const resp = await fetch('/api/animation-levels');
  if (!resp.ok) throw new Error('Failed to load levels');
  return resp.json();
}

export async function fetchAnimationFrames(level) {
  const resp = await fetch('/api/animation-frames', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ level }),
  });
  return resp.json();
}

export async function stitchFrames(frames, fps, format) {
  const resp = await fetch('/api/stitch-frames', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ frames, fps, format }),
  });
  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.error || 'Export failed');
  }
  return resp.blob();
}

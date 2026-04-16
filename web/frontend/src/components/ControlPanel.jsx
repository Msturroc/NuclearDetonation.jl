import { useState, useCallback } from 'react';
import { formatValue, getUnitInfo, BOMB_BASE_LEVELS, BOMB_COLORS, NPP_BASE_LEVELS, NPP_COLORS, DOSE_UNITS } from '../constants';
import { loadARLFromPath, uploadARLFiles } from '../api';
import AnimationPanel from './AnimationPanel';
import HistoryPanel from './HistoryPanel';

export default function ControlPanel({
  dataset, onDatasetChange, datasetLoading,
  releaseMode, onReleaseModeChange,
  weatherSource, onWeatherSourceChange,
  lat, onLatChange, lon, onLonChange,
  yieldKt, onYieldChange,
  activityTbq, onActivityChange,
  releaseDuration, onReleaseDurationChange,
  stackHeight, onStackHeightChange,
  isotope, onIsotopeChange,
  startDate, onStartDateChange,
  dateMin, dateMax,
  startHour, onStartHourChange,
  duration, onDurationChange,
  particles, onParticlesChange,
  simRunning, onRunSimulation, onResetSimulation,
  progressPct, progressMsg, error,
  results, geojson,
  showContours, onToggleContours,
  showObs, onToggleObs,
  baseUnits, displayUnit, onDisplayUnitChange,
  arlMetadata, onArlMetadataChange,
  era5Bounds,
  onDateMinChange, onDateMaxChange, onMapZoomChange,
  animData, onAnimDataChange,
  onLoadHistoryRun,
  selectedNpp,
  prediction,
  predictionLoading,
}) {
  const [arlPath, setArlPath] = useState('');
  const [arlStatus, setArlStatus] = useState(null);
  const [arlError, setArlError] = useState(false);
  const [arlLoading, setArlLoading] = useState(false);

  const handleLoadARL = useCallback(async () => {
    if (!arlPath.trim()) {
      setArlStatus('Please enter a file or directory path');
      setArlError(true);
      return;
    }
    setArlLoading(true);
    setArlStatus('Scanning ARL files...');
    setArlError(false);
    try {
      const data = await loadARLFromPath(arlPath.trim());
      data.dir_path = arlPath.trim();
      onArlMetadataChange(data);
      onDateMinChange(data.date_min);
      onDateMaxChange(data.date_max);
      onStartDateChange(data.date_min);
      onStartHourChange(0);
      const centerLat = (data.lat_min + data.lat_max) / 2;
      const centerLon = (data.lon_min + data.lon_max) / 2;
      onLatChange(parseFloat(centerLat.toFixed(4)));
      onLonChange(parseFloat(centerLon.toFixed(4)));
      setArlStatus(
        `Loaded ${data.n_files} ARL files (${data.resolution}\u00B0 resolution, ` +
        `${data.pressure_levels.length} levels). ` +
        `Dates: ${data.date_min} to ${data.date_max}. ` +
        `Click within the blue rectangle on the map.`
      );
    } catch (e) {
      setArlStatus(e.message);
      setArlError(true);
    } finally {
      setArlLoading(false);
    }
  }, [arlPath, onArlMetadataChange, onDateMinChange, onDateMaxChange,
      onStartDateChange, onStartHourChange, onLatChange, onLonChange]);

  const handleUploadARL = useCallback(async (files) => {
    if (!files || files.length === 0) return;
    setArlLoading(true);
    setArlStatus(`Uploading ${files.length} file(s)...`);
    setArlError(false);
    try {
      const data = await uploadARLFiles(files);
      setArlStatus(`Uploaded ${data.n_files} file(s). Loading metadata...`);
      setArlPath(data.upload_dir);
      // Now load via path
      const meta = await loadARLFromPath(data.upload_dir);
      meta.dir_path = data.upload_dir;
      onArlMetadataChange(meta);
      onDateMinChange(meta.date_min);
      onDateMaxChange(meta.date_max);
      onStartDateChange(meta.date_min);
      onStartHourChange(0);
      const centerLat = (meta.lat_min + meta.lat_max) / 2;
      const centerLon = (meta.lon_min + meta.lon_max) / 2;
      onLatChange(parseFloat(centerLat.toFixed(4)));
      onLonChange(parseFloat(centerLon.toFixed(4)));
      setArlStatus(
        `Loaded ${meta.n_files} ARL files (${meta.resolution}\u00B0 resolution, ` +
        `${meta.pressure_levels.length} levels). ` +
        `Dates: ${meta.date_min} to ${meta.date_max}.`
      );
    } catch (e) {
      setArlStatus('Upload error: ' + e.message);
      setArlError(true);
    } finally {
      setArlLoading(false);
    }
  }, [onArlMetadataChange, onDateMinChange, onDateMaxChange,
      onStartDateChange, onStartHourChange, onLatChange, onLonChange]);

  const era5NoteText = dataset === 'etex'
    ? `ERA5 data covers Europe (lat ${era5Bounds?.lat_min?.toFixed(0) || '?'}\u2013${era5Bounds?.lat_max?.toFixed(0) || '?'}, lon ${era5Bounds?.lon_min?.toFixed(0) || '?'}\u2013${era5Bounds?.lon_max?.toFixed(0) || '?'}), 23\u201327 October 1994. Click the map to place the release.`
    : 'ERA5 data covers Nevada Test Site region (lat 35\u201342, lon -120 to -110), 24\u201327 March 1953. Click the map to place the release.';

  const unitInfo = getUnitInfo(baseUnits, displayUnit);

  return (
    <div className="panel">
      <h1>Atmospheric Dispersion<small>NuclearDetonation.jl</small></h1>

      {/* Dataset selector */}
      <div className="form-group">
        <label>Dataset</label>
        <select
          value={dataset}
          onChange={e => onDatasetChange(e.target.value)}
          disabled={datasetLoading}
        >
          <option value="nancy">Nancy (NTS) — Nevada, Mar 1953</option>
          <option value="etex">ETEX (Europe) — Monterfil, Oct 1994</option>
        </select>
      </div>

      {/* Release mode toggle */}
      <div className="mode-toggle">
        <input type="radio" name="release_mode" id="mode-bomb" value="bomb"
          checked={releaseMode === 'bomb'} onChange={() => onReleaseModeChange('bomb')} />
        <label htmlFor="mode-bomb">Bomb Release</label>
        <input type="radio" name="release_mode" id="mode-npp" value="npp"
          checked={releaseMode === 'npp'} onChange={() => onReleaseModeChange('npp')} />
        <label htmlFor="mode-npp">Point Release (NPP)</label>
      </div>

      {/* New simulation button */}
      {results && (
        <button className="btn-new-sim" onClick={onResetSimulation}>
          New Simulation
        </button>
      )}

      {/* Weather source toggle */}
      <div className="mode-toggle" style={{ marginTop: 4 }}>
        <input type="radio" name="weather_source" id="ws-era5" value="era5"
          checked={weatherSource === 'era5'} onChange={() => onWeatherSourceChange('era5')} />
        <label htmlFor="ws-era5">Built-in ERA5</label>
        <input type="radio" name="weather_source" id="ws-arl" value="arl"
          checked={weatherSource === 'arl'} onChange={() => onWeatherSourceChange('arl')} />
        <label htmlFor="ws-arl">Local ARL Files</label>
      </div>

      {/* ERA5 note */}
      {weatherSource === 'era5' && (
        <div className="note">{era5NoteText}</div>
      )}

      {/* ARL section */}
      {weatherSource === 'arl' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div className="form-group">
            <label>ARL File or Directory</label>
            <div className="arl-path-row">
              <input type="text" value={arlPath}
                onChange={e => setArlPath(e.target.value)}
                placeholder="/path/to/Weather_data/ or /path/to/file.ARL" />
              <button className="btn-load-arl" onClick={handleLoadARL} disabled={arlLoading}>
                {arlLoading ? 'Loading...' : 'Load'}
              </button>
            </div>
          </div>
          <div className="form-group" style={{ marginTop: 4 }}>
            <label>Or Upload ARL Files</label>
            <input type="file" multiple accept=".ARL,.arl"
              style={{ fontSize: 12 }}
              onChange={e => handleUploadARL(e.target.files)} />
          </div>
          {arlStatus && (
            <div className={`arl-status${arlError ? ' error' : ''}`}>{arlStatus}</div>
          )}
        </div>
      )}

      {/* Coordinates */}
      <div className="form-row">
        <div className="form-group">
          <label>Latitude</label>
          <input type="number" value={lat} step="0.01"
            onChange={e => onLatChange(parseFloat(e.target.value))} />
        </div>
        <div className="form-group">
          <label>Longitude</label>
          <input type="number" value={lon} step="0.01"
            onChange={e => onLonChange(parseFloat(e.target.value))} />
        </div>
      </div>

      {/* Bomb fields */}
      {releaseMode === 'bomb' && (
        <div className="form-group">
          <label>Yield (kT)</label>
          <input type="number" value={yieldKt} step="1" min="0.1" max="1000"
            onChange={e => onYieldChange(parseFloat(e.target.value))} />
        </div>
      )}

      {/* NPP fields */}
      {releaseMode === 'npp' && (
        <>
          <div className="form-row">
            <div className="form-group">
              <label>Activity (TBq)</label>
              <input type="number" value={activityTbq} step="0.1" min="0.001" max="100000"
                onChange={e => onActivityChange(parseFloat(e.target.value))} />
            </div>
            <div className="form-group">
              <label>Release Dur. (h)</label>
              <input type="number" value={releaseDuration} step="0.5" min="0.1" max="48"
                onChange={e => onReleaseDurationChange(parseFloat(e.target.value))} />
            </div>
          </div>
          <div className="form-row" style={{ marginTop: 4 }}>
            <div className="form-group">
              <label>Stack Height (m)</label>
              <input type="number" value={stackHeight} step="10" min="10" max="500"
                onChange={e => onStackHeightChange(parseFloat(e.target.value))} />
            </div>
            <div className="form-group">
              <label>Isotope</label>
              <select value={isotope} onChange={e => onIsotopeChange(e.target.value)}>
                <option value="Cs-137">Cs-137</option>
                <option value="I-131">I-131</option>
                <option value="Sr-90">Sr-90</option>
                <option value="Generic">Generic (no decay)</option>
              </select>
            </div>
          </div>
        </>
      )}

      {/* Date/time */}
      <div className="form-row">
        <div className="form-group">
          <label>Start date</label>
          <input type="date" value={startDate} min={dateMin} max={dateMax}
            onChange={e => onStartDateChange(e.target.value)} />
        </div>
        <div className="form-group">
          <label>Hour (UTC)</label>
          <input type="number" value={startHour} step="1" min="0" max="23"
            onChange={e => onStartHourChange(parseInt(e.target.value))} />
        </div>
      </div>

      <div className="form-row">
        <div className="form-group">
          <label>Duration (hours)</label>
          <input type="number" value={duration} step="1" min="1" max="168"
            onChange={e => onDurationChange(parseInt(e.target.value))} />
        </div>
        <div className="form-group">
          <label>Particles</label>
          <input type="number" value={particles} step="500" min="100" max="50000"
            onChange={e => onParticlesChange(parseInt(e.target.value))} />
        </div>
      </div>

      {/* Impact prediction banner */}
      {predictionLoading && (
        <div className="prediction-banner loading">
          Running impact prediction...
        </div>
      )}
      {prediction && !predictionLoading && (
        <div className={`prediction-banner ${prediction.impact ? 'impact' : 'no-impact'}`}>
          <div className="prediction-title">
            XGBoost Prediction: {selectedNpp?.name || prediction.site}
          </div>
          <div className="prediction-result">
            {prediction.impact ? 'Ireland WILL be impacted' : 'Ireland will NOT be impacted'}
          </div>
          <div className="prediction-prob">
            Probability: {(prediction.probability * 100).toFixed(1)}%
          </div>
        </div>
      )}

      {/* Run button */}
      <button className="btn-run"
        disabled={simRunning || datasetLoading}
        onClick={onRunSimulation}>
        {simRunning ? 'Running...' : datasetLoading ? 'Loading dataset...' : 'Run Simulation'}
      </button>

      {/* Progress */}
      {simRunning && (
        <div>
          <div className="progress-bar-track">
            <div className="progress-bar" style={{ width: `${progressPct}%` }} />
          </div>
          <div className="progress-msg">{progressMsg}</div>
        </div>
      )}

      {/* Error */}
      {error && <div className="error-box">{error}</div>}

      {/* Results */}
      {results && (
        <div>
          <div className="stat">
            Peak dose: <strong>{formatValue(results.maxDose, displayUnit, baseUnits)} {unitInfo.label}</strong>
          </div>
          <div className="stat">
            Deposition events: {results.nEvents.toLocaleString()}
          </div>
          <div style={{ display: 'flex', gap: 6, marginTop: 8, flexWrap: 'wrap' }}>
            {geojson && (
              <button className={`btn-secondary${showContours ? ' active' : ''}`}
                onClick={onToggleContours}>
                {showContours ? 'Hide Contours' : 'Show Dose Contours'}
              </button>
            )}
            <button className={`btn-obs${showObs ? ' active' : ''}`}
              onClick={onToggleObs}>
              {showObs ? 'Hide Observations' : 'Show Observations'}
            </button>
            <button className="btn-secondary"
              onClick={() => window.open('/api/results.csv', '_blank')}>
              Export CSV
            </button>
          </div>
        </div>
      )}

      {/* Animation panel */}
      {results && (
        <AnimationPanel
          animData={animData}
          onAnimDataChange={onAnimDataChange}
          releaseMode={releaseMode}
          startDate={startDate}
          duration={duration}
        />
      )}

      {/* Legend */}
      {results && (
        <div className="legend">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
            <h3 style={{ margin: 0, flex: 1 }}>{unitInfo.title}</h3>
            {baseUnits !== 'kBq/m\u00B2' && (
              <select value={displayUnit} onChange={e => onDisplayUnitChange(e.target.value)}
                style={{ fontSize: 11, padding: '2px 4px', border: '1px solid #ccc', borderRadius: 3 }}>
                <option value="mSv/h">mSv/h</option>
                <option value="\u03BCSv/h">{'\u03BC'}Sv/h</option>
                <option value="mR/h">mR/h</option>
              </select>
            )}
          </div>
          <div>
            {baseUnits === 'kBq/m\u00B2'
              ? NPP_BASE_LEVELS.map((lv, i) => (
                  <div key={i} className="legend-item">
                    <span className="legend-swatch" style={{ background: NPP_COLORS[i] }} />
                    {lv >= 100 ? lv.toFixed(0) : lv >= 1 ? lv.toFixed(1) : lv.toFixed(3)}
                  </div>
                ))
              : BOMB_BASE_LEVELS.map((lv, i) => {
                  const info = DOSE_UNITS[displayUnit] || DOSE_UNITS['mSv/h'];
                  const converted = lv * info.factor;
                  const label = converted >= 100 ? converted.toFixed(0) : converted >= 1 ? converted.toFixed(1) : converted.toFixed(3);
                  return (
                    <div key={i} className="legend-item">
                      <span className="legend-swatch" style={{ background: BOMB_COLORS[i] }} />
                      {label}
                    </div>
                  );
                })
            }
          </div>
        </div>
      )}

      {/* Simulation history */}
      <HistoryPanel onLoadRun={onLoadHistoryRun} />
    </div>
  );
}

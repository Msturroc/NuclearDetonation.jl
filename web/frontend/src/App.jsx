import { useState, useCallback, useRef, useEffect } from 'react';
import ControlPanel from './components/ControlPanel';
import MapView from './components/MapView';
import { DATASET_DEFAULTS } from './constants';
import { fetchStatus, startSimulation, loadDataset, fetchERA5Bounds } from './api';
import './App.css';

export default function App() {
  // --- Form state ---
  const [dataset, setDataset] = useState('nancy');
  const [releaseMode, setReleaseMode] = useState('bomb');
  const [weatherSource, setWeatherSource] = useState('era5');
  const [lat, setLat] = useState(37.0956);
  const [lon, setLon] = useState(-116.1028);
  const [yieldKt, setYieldKt] = useState(24.0);
  const [activityTbq, setActivityTbq] = useState(1.0);
  const [releaseDuration, setReleaseDuration] = useState(1.0);
  const [stackHeight, setStackHeight] = useState(100);
  const [isotope, setIsotope] = useState('Cs-137');
  const [startDate, setStartDate] = useState('1953-03-24');
  const [startHour, setStartHour] = useState(13);
  const [duration, setDuration] = useState(12);
  const [particles, setParticles] = useState(2500);
  const [dateMin, setDateMin] = useState('1953-03-24');
  const [dateMax, setDateMax] = useState('1953-03-27');

  // --- ARL state ---
  const [arlMetadata, setArlMetadata] = useState(null);

  // --- Simulation state ---
  const [simRunning, setSimRunning] = useState(false);
  const [progressPct, setProgressPct] = useState(0);
  const [progressMsg, setProgressMsg] = useState('');
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [geojson, setGeojson] = useState(null);
  const [showContours, setShowContours] = useState(false);
  const [showObs, setShowObs] = useState(false);

  // --- Display state ---
  const [baseUnits, setBaseUnits] = useState('mSv/h');
  const [displayUnit, setDisplayUnit] = useState('mSv/h');
  const [era5Bounds, setEra5Bounds] = useState(null);
  const [mapZoom, setMapZoom] = useState(7);

  // --- Animation state ---
  const [animData, setAnimData] = useState(null);

  // --- Dataset loading ---
  const [datasetLoading, setDatasetLoading] = useState(false);

  const pollTimerRef = useRef(null);

  // Fetch ERA5 bounds on mount
  useEffect(() => {
    fetchERA5Bounds().then(b => {
      if (b) setEra5Bounds(b);
    }).catch(() => {});
  }, []);

  // Apply dataset defaults
  const applyDefaults = useCallback((ds) => {
    const d = DATASET_DEFAULTS[ds];
    if (!d) return;
    setLat(d.lat);
    setLon(d.lon);
    setStartDate(d.date);
    setDateMin(d.date_min);
    setDateMax(d.date_max);
    setStartHour(d.hour);
    setDuration(d.duration);
    setMapZoom(d.zoom);
    setReleaseMode(d.mode);
    if (d.mode === 'bomb') {
      setYieldKt(d.yield_kt || 24.0);
    } else {
      if (d.activity_tbq) setActivityTbq(d.activity_tbq);
      if (d.stack_height_m) setStackHeight(d.stack_height_m);
      if (d.isotope) setIsotope(d.isotope);
      if (d.release_duration) setReleaseDuration(d.release_duration);
    }
  }, []);

  // Reset simulation state
  const resetSimulation = useCallback(() => {
    setResults(null);
    setGeojson(null);
    setShowContours(false);
    setShowObs(false);
    setError(null);
    setProgressPct(0);
    setProgressMsg('');
    setSimRunning(false);
    setAnimData(null);
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  // Switch dataset
  const handleDatasetChange = useCallback(async (ds) => {
    if (datasetLoading) return;
    setDatasetLoading(true);
    resetSimulation();
    try {
      const data = await loadDataset(ds);
      setDataset(ds);
      applyDefaults(ds);
      setWeatherSource('era5');
      setEra5Bounds(data);
    } catch (e) {
      setError('Dataset switch failed: ' + e.message);
    } finally {
      setDatasetLoading(false);
    }
  }, [datasetLoading, applyDefaults, resetSimulation]);

  // Run simulation
  const handleRunSimulation = useCallback(async () => {
    setSimRunning(true);
    setError(null);
    setResults(null);
    setGeojson(null);
    setShowContours(false);
    setShowObs(false);
    setAnimData(null);
    setProgressPct(0);
    setProgressMsg('Starting...');

    const params = {
      lat, lon,
      start_date: startDate,
      start_hour: startHour,
      duration_hours: duration,
      n_particles: particles,
      release_mode: releaseMode,
      weather_source: weatherSource,
    };

    if (weatherSource === 'arl') {
      if (!arlMetadata) {
        setError('Please load ARL data first');
        setSimRunning(false);
        return;
      }
      params.arl_dir = arlMetadata.dir_path;
    }

    if (releaseMode === 'bomb') {
      params.yield_kt = yieldKt;
    } else {
      params.activity_tbq = activityTbq;
      params.stack_height_m = stackHeight;
      params.isotope = isotope;
      params.release_duration_hours = releaseDuration;
    }

    try {
      await startSimulation(params);
      pollTimerRef.current = setInterval(async () => {
        try {
          const s = await fetchStatus();
          setProgressPct(s.progress_pct);
          setProgressMsg(s.progress_msg);

          if (s.error_msg) {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
            setError(s.error_msg);
            setSimRunning(false);
            return;
          }

          if (!s.running && s.complete) {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
            setSimRunning(false);

            const units = s.units || 'mSv/h';
            setBaseUnits(units);
            setResults({
              maxDose: s.max_dose,
              nEvents: s.n_events,
            });
            if (s.geojson) {
              setGeojson(JSON.parse(s.geojson));
            }
          }
        } catch {
          // network error, keep polling
        }
      }, 1000);
    } catch (e) {
      setError(e.message);
      setSimRunning(false);
    }
  }, [lat, lon, startDate, startHour, duration, particles, releaseMode,
      weatherSource, yieldKt, activityTbq, stackHeight, isotope,
      releaseDuration, arlMetadata]);

  // Map click handler
  const handleMapClick = useCallback((clickLat, clickLon) => {
    if (weatherSource === 'arl' && arlMetadata) {
      const m = arlMetadata;
      if (clickLat < m.lat_min || clickLat > m.lat_max ||
          clickLon < m.lon_min || clickLon > m.lon_max) {
        return;
      }
    }
    setLat(clickLat);
    setLon(clickLon);
  }, [weatherSource, arlMetadata]);

  // NPP marker click
  const handleNPPClick = useCallback((plant) => {
    setLat(plant.lat);
    setLon(plant.lon);
    setReleaseMode('npp');
  }, []);

  // Weather bounds to display
  const weatherBounds = weatherSource === 'arl' && arlMetadata
    ? { lat_min: arlMetadata.lat_min, lat_max: arlMetadata.lat_max,
        lon_min: arlMetadata.lon_min, lon_max: arlMetadata.lon_max }
    : era5Bounds;

  return (
    <div className="app">
      <ControlPanel
        dataset={dataset}
        onDatasetChange={handleDatasetChange}
        datasetLoading={datasetLoading}
        releaseMode={releaseMode}
        onReleaseModeChange={setReleaseMode}
        weatherSource={weatherSource}
        onWeatherSourceChange={setWeatherSource}
        lat={lat} onLatChange={setLat}
        lon={lon} onLonChange={setLon}
        yieldKt={yieldKt} onYieldChange={setYieldKt}
        activityTbq={activityTbq} onActivityChange={setActivityTbq}
        releaseDuration={releaseDuration} onReleaseDurationChange={setReleaseDuration}
        stackHeight={stackHeight} onStackHeightChange={setStackHeight}
        isotope={isotope} onIsotopeChange={setIsotope}
        startDate={startDate} onStartDateChange={setStartDate}
        dateMin={dateMin} dateMax={dateMax}
        startHour={startHour} onStartHourChange={setStartHour}
        duration={duration} onDurationChange={setDuration}
        particles={particles} onParticlesChange={setParticles}
        simRunning={simRunning}
        onRunSimulation={handleRunSimulation}
        onResetSimulation={resetSimulation}
        progressPct={progressPct}
        progressMsg={progressMsg}
        error={error}
        results={results}
        geojson={geojson}
        showContours={showContours}
        onToggleContours={() => setShowContours(c => !c)}
        showObs={showObs}
        onToggleObs={() => setShowObs(o => !o)}
        baseUnits={baseUnits}
        displayUnit={displayUnit}
        onDisplayUnitChange={setDisplayUnit}
        arlMetadata={arlMetadata}
        onArlMetadataChange={setArlMetadata}
        era5Bounds={era5Bounds}
        onDateMinChange={setDateMin}
        onDateMaxChange={setDateMax}
        onMapZoomChange={setMapZoom}
        animData={animData}
        onAnimDataChange={setAnimData}
      />
      <MapView
        lat={lat}
        lon={lon}
        zoom={mapZoom}
        releaseMode={releaseMode}
        geojson={geojson}
        showContours={showContours}
        showObs={showObs}
        weatherBounds={weatherBounds}
        baseUnits={baseUnits}
        displayUnit={displayUnit}
        onMapClick={handleMapClick}
        onNPPClick={handleNPPClick}
        animData={animData}
        dataset={dataset}
      />
    </div>
  );
}

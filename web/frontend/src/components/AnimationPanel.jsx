import { useState, useEffect, useRef, useCallback } from 'react';
import { fetchAnimationLevels, fetchAnimationFrames, stitchFrames } from '../api';
import { concColormap, sciLabel } from '../constants';

export default function AnimationPanel({ animData, onAnimDataChange, releaseMode, startDate, duration }) {
  const [levels, setLevels] = useState([]);
  const [selectedLevel, setSelectedLevel] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [statusMsg, setStatusMsg] = useState('');
  const [exportFormat, setExportFormat] = useState('gif');
  const [exportFps, setExportFps] = useState(2);
  const [exporting, setExporting] = useState(false);
  const playTimerRef = useRef(null);

  // Load animation levels when results appear
  useEffect(() => {
    fetchAnimationLevels().then(data => {
      const levs = data.levels.slice();
      const colTotal = levs.filter(lv => lv.index === 0);
      const individual = levs.filter(lv => lv.index > 0).reverse();
      setLevels([...colTotal, ...individual]);
      if (colTotal.length > 0) {
        setSelectedLevel(colTotal[0].index);
      }
    }).catch(e => {
      console.error('Failed to load animation levels:', e);
    });
  }, []);

  // Load frames when level changes
  useEffect(() => {
    if (levels.length === 0) return;
    stopPlayback();
    setStatusMsg('Loading frames...');

    fetchAnimationFrames(selectedLevel).then(data => {
      if (data.error) {
        setStatusMsg(data.error);
        return;
      }
      if (!data.n_frames || data.n_frames === 0) {
        setStatusMsg(data.message || 'No data at this level');
        onAnimDataChange(null);
        return;
      }
      onAnimDataChange({
        frames: data.frames,
        bounds: data.bounds,
        width: data.width,
        height: data.height,
        times: data.times_hours,
        maxValue: data.max_value,
        units: data.units || '',
        levelLabel: data.level_label || 'Column Total',
        currentIndex: 0,
        visible: false,
      });
      setStatusMsg(
        `${data.n_frames} frames, ${data.level_label || 'Column Total'}, max=${data.max_value.toExponential(1)} Bq`
      );
    }).catch(e => {
      setStatusMsg('Error: ' + e.message);
    });
  }, [selectedLevel, levels.length, onAnimDataChange]);

  const currentIndex = animData?.currentIndex || 0;
  const nFrames = animData?.frames?.length || 0;
  const currentTime = animData?.times?.[currentIndex];

  const setFrameIndex = useCallback((idx) => {
    if (!animData) return;
    onAnimDataChange({ ...animData, currentIndex: idx, visible: true });
  }, [animData, onAnimDataChange]);

  const stopPlayback = useCallback(() => {
    setPlaying(false);
    if (playTimerRef.current) {
      clearInterval(playTimerRef.current);
      playTimerRef.current = null;
    }
  }, []);

  const togglePlay = useCallback(() => {
    if (playing) {
      stopPlayback();
    } else if (nFrames > 0) {
      setPlaying(true);
    }
  }, [playing, nFrames, stopPlayback]);

  // Playback timer
  useEffect(() => {
    if (playing && nFrames > 0) {
      playTimerRef.current = setInterval(() => {
        onAnimDataChange(prev => {
          if (!prev) return prev;
          return { ...prev, currentIndex: (prev.currentIndex + 1) % prev.frames.length, visible: true };
        });
      }, 500);
      return () => {
        if (playTimerRef.current) clearInterval(playTimerRef.current);
      };
    }
  }, [playing, nFrames, onAnimDataChange]);

  const stepForward = useCallback(() => {
    stopPlayback();
    if (nFrames > 0) setFrameIndex((currentIndex + 1) % nFrames);
  }, [stopPlayback, nFrames, currentIndex, setFrameIndex]);

  const stepBack = useCallback(() => {
    stopPlayback();
    if (nFrames > 0) setFrameIndex((currentIndex - 1 + nFrames) % nFrames);
  }, [stopPlayback, nFrames, currentIndex, setFrameIndex]);

  // Export animation
  const handleExport = useCallback(async () => {
    if (nFrames === 0) {
      setStatusMsg('No animation frames loaded');
      return;
    }
    setExporting(true);
    const wasPlaying = playing;
    stopPlayback();

    try {
      // For now, send raw frames to server for stitching
      // (client-side capture with map tiles would require more complex setup)
      setStatusMsg(`Encoding ${exportFormat.toUpperCase()}...`);
      const blob = await stitchFrames(animData.frames, exportFps, exportFormat);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const date = startDate.replace(/-/g, '');
      a.download = `${releaseMode}_${date}_${duration}h.${exportFormat}`;
      a.click();
      URL.revokeObjectURL(url);
      setStatusMsg(`${exportFormat.toUpperCase()} downloaded`);
    } catch (e) {
      setStatusMsg('Export error: ' + e.message);
    } finally {
      setExporting(false);
      if (wasPlaying) togglePlay();
    }
  }, [nFrames, playing, stopPlayback, animData, exportFormat, exportFps,
      releaseMode, startDate, duration, togglePlay]);

  if (levels.length === 0) return null;

  return (
    <div className="animation-panel" style={{ borderTop: '1px solid #eee', paddingTop: 10 }}>
      <h3>Animation</h3>

      <div className="form-group" style={{ marginBottom: 6 }}>
        <label>Height Level</label>
        <select value={selectedLevel} onChange={e => setSelectedLevel(parseInt(e.target.value))}>
          {levels.map(lv => (
            <option key={lv.index} value={lv.index}>{lv.label}</option>
          ))}
        </select>
      </div>

      <div className="anim-controls">
        <button onClick={stepBack} title="Step back">{'\u25C0\u25C0'}</button>
        <button onClick={togglePlay} title="Play/Pause"
          className={playing ? 'active' : ''}>
          {playing ? '\u23F8' : '\u25B6'}
        </button>
        <button onClick={stepForward} title="Step forward">{'\u25B6\u25B6'}</button>
        <span className="anim-time">
          {currentTime !== undefined ? `H+${currentTime.toFixed(0)}` : '--'}
        </span>
      </div>

      <input
        type="range"
        className="anim-slider"
        min="0"
        max={Math.max(0, nFrames - 1)}
        value={currentIndex}
        onChange={e => {
          stopPlayback();
          setFrameIndex(parseInt(e.target.value));
        }}
      />

      <div className="anim-row-export">
        <select value={exportFormat} onChange={e => setExportFormat(e.target.value)}>
          <option value="gif">GIF</option>
          <option value="mp4">MP4</option>
        </select>
        <select value={exportFps} onChange={e => setExportFps(parseInt(e.target.value))}>
          <option value="1">1 fps</option>
          <option value="2">2 fps</option>
          <option value="4">4 fps</option>
          <option value="8">8 fps</option>
        </select>
        <button onClick={handleExport} disabled={exporting || nFrames === 0}>
          {exporting ? 'Exporting...' : 'Download'}
        </button>
      </div>

      {statusMsg && <div className="anim-status">{statusMsg}</div>}
    </div>
  );
}

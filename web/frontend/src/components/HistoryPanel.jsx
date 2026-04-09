import { useState, useEffect, useCallback } from 'react';

export default function HistoryPanel({ onLoadRun }) {
  const [runs, setRuns] = useState([]);
  const [total, setTotal] = useState(0);
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingId, setLoadingId] = useState(null);

  const fetchRuns = useCallback(async () => {
    setLoading(true);
    try {
      const resp = await fetch('/api/runs?limit=20');
      if (!resp.ok) return;
      const data = await resp.json();
      setRuns(data.runs || []);
      setTotal(data.total || 0);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (expanded) fetchRuns();
  }, [expanded, fetchRuns]);

  const handleLoadRun = useCallback(async (run) => {
    setLoadingId(run.id);
    try {
      const resp = await fetch(`/api/runs/${run.id}/load`, { method: 'POST' });
      if (!resp.ok) {
        const err = await resp.json();
        console.error('Failed to load run:', err.error);
        return;
      }
      const data = await resp.json();
      if (onLoadRun) onLoadRun(data);
    } catch (e) {
      console.error('Failed to load run:', e);
    } finally {
      setLoadingId(null);
    }
  }, [onLoadRun]);

  const formatDate = (iso) => {
    if (!iso) return '';
    const d = new Date(iso);
    return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
      + ' ' + d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
  };

  const statusBadge = (status) => {
    const colours = { completed: '#2a7', running: '#4A90D9', failed: '#c00' };
    return (
      <span style={{
        fontSize: 10, fontWeight: 600, color: '#fff',
        background: colours[status] || '#888',
        padding: '1px 6px', borderRadius: 3,
      }}>
        {status}
      </span>
    );
  };

  return (
    <div style={{ borderTop: '1px solid #eee', paddingTop: 10 }}>
      <button
        onClick={() => setExpanded(e => !e)}
        style={{
          width: '100%', padding: '8px 12px', fontSize: 13, fontWeight: 600,
          border: '1px solid #888', background: expanded ? '#f0f0f0' : '#fff',
          color: '#555', borderRadius: 4, cursor: 'pointer',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}
      >
        <span>Simulation History</span>
        <span style={{ fontSize: 11, color: '#888' }}>
          {total > 0 ? `${total} run${total !== 1 ? 's' : ''}` : ''}
          {' '}{expanded ? '\u25B2' : '\u25BC'}
        </span>
      </button>

      {expanded && (
        <div style={{ marginTop: 8, maxHeight: 300, overflowY: 'auto' }}>
          {loading && <div style={{ fontSize: 12, color: '#888', padding: 8 }}>Loading...</div>}
          {!loading && runs.length === 0 && (
            <div style={{ fontSize: 12, color: '#888', padding: 8 }}>No simulations recorded yet.</div>
          )}
          {runs.map(run => {
            const isCompleted = run.status === 'completed';
            const isLoading = loadingId === run.id;
            return (
              <div
                key={run.id}
                onClick={() => isCompleted && !isLoading && handleLoadRun(run)}
                style={{
                  padding: '8px 10px', borderBottom: '1px solid #eee',
                  fontSize: 12, lineHeight: 1.5,
                  cursor: isCompleted ? 'pointer' : 'default',
                  background: isLoading ? '#f0f6ff' : 'transparent',
                  transition: 'background 0.15s',
                }}
                onMouseEnter={e => { if (isCompleted) e.currentTarget.style.background = '#f5f9ff'; }}
                onMouseLeave={e => { if (isCompleted && !isLoading) e.currentTarget.style.background = 'transparent'; }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontWeight: 600, color: '#333' }}>
                    #{run.id} {run.release_mode === 'bomb' ? '\u2622' : '\u26A0'}{' '}
                    {run.release_mode === 'bomb'
                      ? `${run.yield_kt} kT`
                      : `${run.activity_tbq} TBq ${run.isotope || ''}`
                    }
                  </span>
                  <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                    {isCompleted && !isLoading && (
                      <span style={{ fontSize: 10, color: '#4A90D9', fontWeight: 600 }}>Load</span>
                    )}
                    {isLoading && (
                      <span style={{ fontSize: 10, color: '#4A90D9', fontWeight: 600 }}>Loading...</span>
                    )}
                    {statusBadge(run.status)}
                  </div>
                </div>
                <div style={{ color: '#666' }}>
                  {formatDate(run.created_at)} &middot; {run.duration_hours}h &middot; {run.n_particles.toLocaleString()} particles
                </div>
                <div style={{ color: '#666' }}>
                  ({run.latitude.toFixed(2)}, {run.longitude.toFixed(2)}) &middot; {run.weather_source}
                </div>
                {isCompleted && (
                  <div style={{ color: '#333', marginTop: 2 }}>
                    Peak: <strong style={{ color: '#c00' }}>
                      {run.peak_dose != null ? run.peak_dose.toFixed(2) : '?'} {run.dose_units || 'mSv/h'}
                    </strong>
                    {' '}&middot; {run.n_events?.toLocaleString() || '?'} events
                    {' '}&middot; {run.elapsed_seconds != null ? `${run.elapsed_seconds.toFixed(1)}s` : ''}
                  </div>
                )}
                {run.status === 'failed' && run.error_message && (
                  <div style={{ color: '#c00', marginTop: 2, fontSize: 11 }}>
                    {run.error_message.substring(0, 80)}
                  </div>
                )}
              </div>
            );
          })}
          {!loading && (
            <button onClick={fetchRuns} style={{
              width: '100%', padding: 6, fontSize: 11, border: '1px solid #ccc',
              borderRadius: 4, background: '#fff', color: '#555', cursor: 'pointer',
              marginTop: 4,
            }}>
              Refresh
            </button>
          )}
        </div>
      )}
    </div>
  );
}

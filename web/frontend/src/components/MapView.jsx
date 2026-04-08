import { useEffect, useRef, useState, useCallback } from 'react';
import { MapContainer, TileLayer, Marker, Rectangle, GeoJSON, ImageOverlay, useMapEvents, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { NPP_PLANTS, DOSE_UNITS, DEP_UNITS } from '../constants';
import { fetchObservations } from '../api';

// Star icon for detonation marker
const starIcon = L.divIcon({
  html: '<svg width="24" height="24" viewBox="0 0 24 24"><polygon points="12,2 15,9 22,9 16,14 18,22 12,17 6,22 8,14 2,9 9,9" fill="#c00" stroke="#800" stroke-width="1"/></svg>',
  iconSize: [24, 24],
  iconAnchor: [12, 12],
  className: '',
});

// NPP icon
const nppIcon = L.divIcon({
  html: '\u2622',
  className: 'npp-marker',
  iconSize: [22, 22],
  iconAnchor: [11, 11],
});

// Click handler component
function MapClickHandler({ onClick }) {
  useMapEvents({
    click(e) {
      const lat = Math.round(e.latlng.lat * 10000) / 10000;
      const lon = Math.round(e.latlng.lng * 10000) / 10000;
      onClick(lat, lon);
    },
  });
  return null;
}

// Component to sync map view when lat/lon/zoom changes
function MapSync({ lat, lon, zoom }) {
  const map = useMap();
  const prevRef = useRef({ lat, lon, zoom });

  useEffect(() => {
    const prev = prevRef.current;
    if (prev.zoom !== zoom) {
      map.setView([lat, lon], zoom);
    } else if (prev.lat !== lat || prev.lon !== lon) {
      map.panTo([lat, lon]);
    }
    prevRef.current = { lat, lon, zoom };
  }, [lat, lon, zoom, map]);

  return null;
}

// Contour layer with tooltips
function ContourLayer({ geojson, baseUnits, displayUnit }) {
  const getStyle = useCallback((feature) => ({
    color: feature.properties.color,
    weight: 2.5,
    opacity: 0.9,
  }), []);

  const onEachFeature = useCallback((feature, layer) => {
    const baseVal = feature.properties.level;
    const info = baseUnits === 'kBq/m\u00B2'
      ? (DEP_UNITS[displayUnit] || DEP_UNITS['kBq/m\u00B2'])
      : (DOSE_UNITS[displayUnit] || DOSE_UNITS['mSv/h']);
    const converted = baseVal * info.factor;
    const label = (converted >= 1 ? converted.toFixed(converted >= 100 ? 0 : 1) : converted.toFixed(3)) + ' ' + info.label;
    layer.bindTooltip(label, { sticky: true });
  }, [baseUnits, displayUnit]);

  // Force re-render when display unit changes by using a key
  return <GeoJSON key={`contours-${displayUnit}`} data={geojson} style={getStyle} onEachFeature={onEachFeature} />;
}

// Observation overlay
function ObservationLayer({ dataset }) {
  const [obsData, setObsData] = useState(null);

  useEffect(() => {
    fetchObservations()
      .then(data => setObsData(data))
      .catch(() => setObsData(null));
  }, [dataset]);

  if (!obsData || !obsData.geojson) return null;

  const type = obsData.type;
  const prefix = type === 'etex' ? 'ETEX observed: ' : 'Observed: ';

  return (
    <GeoJSON
      key={`obs-${dataset}`}
      data={obsData.geojson}
      style={(feature) => ({
        color: feature.properties.color,
        weight: 2.5,
        opacity: 0.85,
        fillOpacity: type === 'nancy' ? 0.12 : 0,
        dashArray: '6,4',
      })}
      onEachFeature={(feature, layer) => {
        layer.bindTooltip(prefix + feature.properties.label, { sticky: true });
      }}
    />
  );
}

// Animation overlay
function AnimationOverlay({ animData }) {
  const [dataUrl, setDataUrl] = useState(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!animData || !animData.frames || animData.frames.length === 0) {
      setDataUrl(null);
      return;
    }

    const idx = animData.currentIndex || 0;
    if (idx >= animData.frames.length) return;

    const raw = atob(animData.frames[idx]);
    const rgba = new Uint8ClampedArray(raw.length);
    for (let i = 0; i < raw.length; i++) rgba[i] = raw.charCodeAt(i);

    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = animData.width;
    tmpCanvas.height = animData.height;
    const tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.putImageData(new ImageData(rgba, animData.width, animData.height), 0, 0);

    const scale = Math.max(4, Math.ceil(800 / Math.max(animData.width, animData.height)));
    if (!canvasRef.current) canvasRef.current = document.createElement('canvas');
    const canvas = canvasRef.current;
    canvas.width = animData.width * scale;
    canvas.height = animData.height * scale;
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(tmpCanvas, 0, 0, canvas.width, canvas.height);

    setDataUrl(canvas.toDataURL());
  }, [animData]);

  if (!dataUrl || !animData?.bounds) return null;

  const bounds = [
    [animData.bounds.lat_min, animData.bounds.lon_min],
    [animData.bounds.lat_max, animData.bounds.lon_max],
  ];

  return <ImageOverlay url={dataUrl} bounds={bounds} opacity={0.8} />;
}

export default function MapView({
  lat, lon, zoom, releaseMode,
  geojson, showContours, showObs,
  weatherBounds, baseUnits, displayUnit,
  onMapClick, onNPPClick,
  animData, dataset,
}) {
  const [hintHidden, setHintHidden] = useState(false);

  const handleClick = useCallback((clickLat, clickLon) => {
    setHintHidden(true);
    onMapClick(clickLat, clickLon);
  }, [onMapClick]);

  return (
    <div className="map-container">
      <div className={`map-hint${hintHidden ? ' hidden' : ''}`}>
        Click the map to place release location
      </div>
      <MapContainer
        center={[lat, lon]}
        zoom={zoom}
        style={{ width: '100%', height: '100%' }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; OpenStreetMap contributors'
          maxZoom={18}
        />
        <MapClickHandler onClick={handleClick} />
        <MapSync lat={lat} lon={lon} zoom={zoom} />

        {/* Detonation marker */}
        <Marker position={[lat, lon]} icon={starIcon} />

        {/* NPP plant markers (only in NPP mode) */}
        {releaseMode === 'npp' && NPP_PLANTS.map(plant => (
          <Marker
            key={plant.name}
            position={[plant.lat, plant.lon]}
            icon={nppIcon}
            zIndexOffset={500}
            eventHandlers={{
              click: () => {
                setHintHidden(true);
                onNPPClick(plant);
              },
            }}
          />
        ))}

        {/* Weather bounds rectangle */}
        {weatherBounds && (
          <Rectangle
            bounds={[
              [weatherBounds.lat_min, weatherBounds.lon_min],
              [weatherBounds.lat_max, weatherBounds.lon_max],
            ]}
            pathOptions={{
              color: '#4A90D9', weight: 2, fill: true,
              fillOpacity: 0.05, dashArray: '5,5',
            }}
          />
        )}

        {/* Contour overlay */}
        {showContours && geojson && (
          <ContourLayer geojson={geojson} baseUnits={baseUnits} displayUnit={displayUnit} />
        )}

        {/* Observation overlay */}
        {showObs && <ObservationLayer dataset={dataset} />}

        {/* Animation overlay */}
        {animData?.visible && <AnimationOverlay animData={animData} />}
      </MapContainer>
    </div>
  );
}

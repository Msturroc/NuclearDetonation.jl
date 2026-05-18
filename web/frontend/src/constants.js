// Dataset defaults (matching server-side presets)
export const DATASET_DEFAULTS = {
  nancy: {
    lat: 37.0956, lon: -116.1028, date: '1953-03-24', hour: 13,
    date_min: '1953-03-24', date_max: '1953-03-27',
    duration: 12, zoom: 7, mode: 'bomb', yield_kt: 24.0,
  },
  etex: {
    lat: 48.058, lon: -2.008, date: '1994-10-23', hour: 16,
    date_min: '1994-10-23', date_max: '1994-10-27',
    duration: 48, zoom: 5, mode: 'npp',
    activity_tbq: 1.0, stack_height_m: 10, isotope: 'Generic',
    release_duration: 12,
  },
};

// Common reactor / fallout isotopes shown as datalist suggestions on the
// source-term editor. Any built-in name (Cs-137, I-131, Sr-90, Generic) is
// resolved against ISOTOPE_HALFLIVES in web/src/simulation.jl when no
// halflife override is supplied; everything else needs an explicit halflife.
export const ISOTOPE_SUGGESTIONS = [
  { name: 'Cs-137',  halflife_hours: 30.17 * 365.25 * 24 },
  { name: 'I-131',   halflife_hours: 8.02 * 24 },
  { name: 'Sr-90',   halflife_hours: 28.9 * 365.25 * 24 },
  { name: 'Cs-134',  halflife_hours: 2.062 * 365.25 * 24 },
  { name: 'Co-60',   halflife_hours: 5.27 * 365.25 * 24 },
  { name: 'I-133',   halflife_hours: 20.8 },
  { name: 'Ru-103',  halflife_hours: 39.26 * 24 },
  { name: 'Ru-106',  halflife_hours: 373.6 * 24 },
  { name: 'Te-132',  halflife_hours: 3.20 * 24 },
  { name: 'Xe-133',  halflife_hours: 5.25 * 24 },
  { name: 'Pu-239',  halflife_hours: 24110 * 365.25 * 24 },
  { name: 'Generic', halflife_hours: 0 },
];

// NPP plant markers (site key matches prediction model filenames)
export const NPP_PLANTS = [
  { name: 'Hinkley Point C', site: 'hinkley',     lat: 51.2086, lon: -3.1304 },
  { name: 'Wylfa',           site: 'wylfa',        lat: 53.4167, lon: -4.4822 },
  { name: 'Paluel',          site: 'paluel',        lat: 49.8584, lon: 0.6354 },
  { name: 'Flamanville',     site: 'flamanville',   lat: 49.5381, lon: -1.8802 },
  { name: 'Sizewell B',      site: 'sizewell',      lat: 52.2145, lon: 1.6206 },
  { name: 'Heysham',         site: 'heysham',        lat: 54.0285, lon: -2.9161 },
];

// Contour levels and colours
export const BOMB_BASE_LEVELS = [1.0, 0.4, 0.1, 0.04, 0.01, 0.004];
export const BOMB_COLORS = ['#CC0000', '#FF8800', '#CCCC00', '#33AA33', '#00CCCC', '#3366FF'];
// Reversed for top-down legend display. Must mirror NPP_LEVELS/NPP_COLORS in
// web/src/contours.jl. Lower three levels are visualisation-only; upper four
// match IAEA/Chernobyl Cs-137 zoning thresholds.
export const NPP_BASE_LEVELS = [1480, 555, 185, 37, 10, 1, 0.1, 0.01, 0.001];
export const NPP_COLORS = ['#CC0000', '#FF8800', '#CCCC00', '#33AA33', '#00CCCC',
                            '#3366FF', '#5A8DD0', '#8FB7E8', '#C8DCFF'];

// Dose rate unit conversions (from mSv/h)
export const DOSE_UNITS = {
  'mSv/h':  { factor: 1,     label: 'mSv/h',  title: 'Dose Rate (mSv/h)' },
  'μSv/h':  { factor: 1000,  label: 'μSv/h',  title: 'Dose Rate (μSv/h)' },
  'Sv/h':   { factor: 0.001, label: 'Sv/h',   title: 'Dose Rate (Sv/h)'  },
  'mR/h':   { factor: 100,   label: 'mR/h',   title: 'Dose Rate (mR/h)'  },
};

// Deposition unit conversions (from kBq/m2)
export const DEP_UNITS = {
  'kBq/m²':  { factor: 1,       label: 'kBq/m²',  title: 'Deposition (kBq/m²)' },
  'Ci/km²':  { factor: 0.02703, label: 'Ci/km²',  title: 'Deposition (Ci/km²)' },
};

export function formatValue(val, unit, baseUnits) {
  const info = baseUnits === 'kBq/m²' ? DEP_UNITS[unit] : DOSE_UNITS[unit];
  if (!info) return val.toFixed(2);
  const converted = val * info.factor;
  if (converted >= 100) return converted.toFixed(0);
  if (converted >= 1) return converted.toFixed(1);
  if (converted >= 0.01) return converted.toFixed(3);
  return converted.toExponential(2);
}

export function getUnitInfo(baseUnits, displayUnit) {
  if (baseUnits === 'kBq/m²') return DEP_UNITS[displayUnit] || DEP_UNITS['kBq/m²'];
  return DOSE_UNITS[displayUnit] || DOSE_UNITS['mSv/h'];
}

// Colormap: blue -> cyan -> green -> yellow -> red (t in [0,1])
export function concColormap(t) {
  let r, g, b;
  if (t < 0.25)      { r = 0;              g = t / 0.25;              b = 1; }
  else if (t < 0.5)  { r = 0;              g = 1;                     b = 1 - (t - 0.25) / 0.25; }
  else if (t < 0.75) { r = (t - 0.5)/0.25; g = 1;                     b = 0; }
  else               { r = 1;              g = 1 - (t - 0.75) / 0.25; b = 0; }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

export function sciLabel(val) {
  if (val >= 1) return val.toPrecision(2);
  const e = Math.floor(Math.log10(val));
  const m = val / Math.pow(10, e);
  return m.toFixed(1) + 'e' + e;
}

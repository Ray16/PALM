/* PALM Data Splitting — client. Plain JS, no build step. */
'use strict';

// ── Metadata ────────────────────────────────────────────────────────────────

const FEATURES = {
  molecule: [
    {id: 'rdkit_descriptors', label: 'RDKit descriptors', desc: 'MW, H-bond donors/acceptors, TPSA, LogP'},
    {id: 'composition', label: 'Composition', desc: 'Element counts and weighted atomic properties'},
    {id: 'physicochemical', label: 'Physicochemical', desc: 'Atom/ring counts, aromaticity, sp3 fraction'},
    {id: 'morgan_fingerprint', label: 'Morgan FP (ECFP4)', desc: '2048-bit circular fingerprint; Tanimoto similarity'},
    {id: 'maccs_keys', label: 'MACCS keys', desc: '167 structural keys'},
    {id: 'rdkit_descriptors_full', label: 'RDKit (full)', desc: 'All ~200 RDKit 2D descriptors'},
    {id: 'chemberta_embedding', label: 'ChemBERTa', desc: 'Chemical language model embedding (GPU recommended)'},
  ],
  biomolecule: [
    {id: 'esm_embedding', label: 'ESM2-650M', desc: 'Protein language model embedding, 1280-d (GPU recommended)'},
    {id: 'sequence_properties', label: 'Sequence properties', desc: 'Length, MW, hydrophobicity, charge, AA fractions'},
    {id: 'precomputed_embedding', label: 'Precomputed', desc: 'Embeddings from a supplied file'},
  ],
  gene: [
    {id: 'nucleotide_composition', label: 'Nucleotide composition', desc: 'GC content, base frequencies, CpG O/E'},
    {id: 'kmer_frequencies', label: 'k-mer frequencies', desc: 'Di- and tri-nucleotide frequencies'},
    {id: 'nt_embedding', label: 'Nucleotide Transformer', desc: 'DNA language model embedding (GPU recommended)'},
    {id: 'precomputed_embedding', label: 'Precomputed', desc: 'Embeddings from a supplied file'},
  ],
  material: [
    {id: 'magpie_properties', label: 'Magpie', desc: 'Statistical elemental descriptors'},
    {id: 'stoichiometry', label: 'Stoichiometry', desc: 'Element count, entropy, Lp norms'},
    {id: 'electronic', label: 'Electronic', desc: 'Valence/d-electron and oxidation-state statistics'},
    {id: 'bonding', label: 'Bonding', desc: 'Electronegativity differences, radii, polarizability'},
    {id: 'thermodynamic', label: 'Thermodynamic', desc: 'Melting points, formation enthalpies'},
    {id: 'classification', label: 'Element classes', desc: 'Transition/noble/rare-earth fractions'},
    {id: 'mat2vec_embedding', label: 'mat2vec', desc: 'Composition-weighted mat2vec embedding'},
    {id: 'matminer_elementproperty', label: 'matminer Magpie', desc: 'Requires matminer'},
    {id: 'crystalnn_fingerprint', label: 'CrystalNN', desc: 'Site fingerprints; needs structure files'},
    {id: 'soap_descriptor', label: 'SOAP', desc: '3-D descriptor; needs structure files + dscribe'},
  ],
};

// Default features = what the splitter uses by default (proteins: ESM2-650M).
const FEAT_DEFAULTS = {
  molecule: ['rdkit_descriptors', 'composition', 'physicochemical', 'morgan_fingerprint'],
  biomolecule: ['esm_embedding'],
  gene: ['nucleotide_composition', 'kmer_frequencies'],
  material: ['magpie_properties', 'stoichiometry', 'electronic', 'bonding', 'thermodynamic', 'classification'],
};

const TECH = {
  R:          {name: 'Random', family: 'Baseline', desc: 'Uniform random split — no leakage control. Reference point for the other methods.'},
  I1e:        {name: 'Identity · e1', family: 'DataSAIL', desc: 'No e1 entity appears in more than one split.'},
  I1f:        {name: 'Identity · e2', family: 'DataSAIL', desc: 'No e2 entity appears in more than one split.'},
  I2:         {name: 'Identity · both', family: 'DataSAIL', desc: 'Cold on both axes; cross-split pairs are dropped.'},
  C1e:        {name: 'Cluster · e1', family: 'DataSAIL', desc: 'Clusters of similar e1 entities are kept together (MILP solver).'},
  C1f:        {name: 'Cluster · e2', family: 'DataSAIL', desc: 'Clusters of similar e2 entities are kept together.'},
  C2:         {name: 'Cluster · both', family: 'DataSAIL', desc: 'Similarity clusters on both axes — strictest, slowest; drops cross-split pairs.'},
  scaffold:   {name: 'Scaffold', family: 'Chemistry', desc: 'Bemis–Murcko scaffold groups, greedily packed (molecules only).'},
  hypergraph: {name: 'Hypergraph', family: 'PALM', desc: 'k-NN similarity hyperedges partitioned with Mt-KaHyPar (KM1 cut). 2D: per-axis k-NN hyperedges.', isNew: true},
  lowrank:    {name: 'Low-rank', family: 'Low-rank', desc: 'Nyström low-rank similarity; balanced Lloyd + FM refinement. Linear memory — scales to millions.', isNew: true},
};
const TECH_1D = [['Baseline', ['R']], ['Leakage-minimizing', ['lowrank', 'hypergraph']],
                 ['DataSAIL', ['I1e', 'C1e']], ['Chemistry', ['scaffold']]];
const TECH_2D = [['Baseline', ['R']], ['Leakage-minimizing', ['lowrank', 'hypergraph']],
                 ['DataSAIL', ['I1e', 'I1f', 'I2', 'C1e', 'C1f', 'C2']]];
const TECH_DEFAULTS = {'1d': ['R', 'C1e', 'hypergraph', 'lowrank'], '2d': ['R', 'I1e', 'hypergraph', 'lowrank']};

const TYPE_LABEL = {molecule: 'Molecule', biomolecule: 'Protein', gene: 'DNA/RNA', material: 'Material'};
const SPLIT_SLOT = {train: 1, test: 2, val: 3};

// ── State ───────────────────────────────────────────────────────────────────

const S = {
  jobId: null, columns: [], demo: null, mode: '1d', info: {},
  maps: {},             // map key -> map payload
  exploreAxis: 'e1', resultAxis: 'e1', resultTech: null,
  run: null,            // payload of the last run
  comparison: null, splitLabels: {},  // axis -> {tech: labels}
  evt: null, pollTimer: null, startedAt: 0, clock: null,
};

// ── Helpers ─────────────────────────────────────────────────────────────────

const $ = id => document.getElementById(id);
const REDUCED_MOTION = matchMedia('(prefers-reduced-motion: reduce)').matches;
// Anonymous per-browser id: the server shows each browser only its own runs.
const CLIENT_ID = localStorage.getItem('palm-client') ||
  (localStorage.setItem('palm-client', crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2)),
   localStorage.getItem('palm-client'));
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const fmtInt = n => Number(n).toLocaleString();
const show = (el, on = true) => (typeof el === 'string' ? $(el) : el).classList.toggle('hidden', !on);
const ICON_ALERT = '<svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/></svg>';

function alertBox(id, msg) {
  const el = $(id);
  el.innerHTML = msg ? ICON_ALERT + '<div>' + esc(msg) + '</div>' : '';
  show(el, !!msg);
}

async function api(path, opts = {}) {
  const res = await fetch(path, {...opts, headers: {'X-Client-Id': CLIENT_ID, ...(opts.headers || {})}});
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-JSON */ }
  if (!res.ok) throw new Error((data && (data.detail || data.error)) || `${res.status} ${res.statusText}`);
  return data;
}

function shortId(s, n = 42) { s = String(s ?? ''); return s.length > n ? s.slice(0, n - 1) + '…' : s; }

// ── Navigation ──────────────────────────────────────────────────────────────

let maxStep = 1;
function goStep(n) {
  if (n <= 4) maxStep = Math.max(maxStep, n);
  document.querySelectorAll('.panel').forEach((p, i) => {
    const on = i + 1 === n;
    if (on && !p.classList.contains('active'))      // stagger the cards in
      p.querySelectorAll(':scope .card, :scope .cols > div > .card, :scope .rec, :scope .tiles')
        .forEach((c, k) => c.style.setProperty('--i', k));
    p.classList.toggle('active', on);
  });
  $('runs-btn').classList.toggle('on', n === 5);
  document.querySelectorAll('.step').forEach(b => {
    const k = +b.dataset.step;
    const reachable = k <= 2 ? k <= maxStep : k === 3 ? !!S.runId : !!(S.runId && S.comparison);
    b.classList.toggle('active', k === n);
    b.classList.toggle('done', k !== n && reachable);
    b.querySelector('.dot').textContent = k !== n && reachable ? '✓' : k;
  });
  window.scrollTo({top: 0, behavior: 'smooth'});
  if (n === 2) requestAnimationFrame(() => resizePlot('explore-plot'));
  if (n === 4) requestAnimationFrame(() => resizePlot('result-plot'));
}

// ── Theme ───────────────────────────────────────────────────────────────────

function initTheme() {
  const saved = localStorage.getItem('palm-theme');
  if (saved) document.documentElement.dataset.theme = saved;
  $('theme-btn').addEventListener('click', () => {
    const dark = document.documentElement.dataset.theme === 'dark' ||
      (!document.documentElement.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
    document.documentElement.dataset.theme = dark ? 'light' : 'dark';
    localStorage.setItem('palm-theme', document.documentElement.dataset.theme);
    redrawPlots();
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redrawPlots);
}

// ── Step 1: data ────────────────────────────────────────────────────────────

function initUpload() {
  const dz = $('dropzone'), fi = $('file-input'), di = $('folder-input');
  dz.addEventListener('click', e => { if (e.target.id !== 'folder-link') fi.click(); });
  dz.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fi.click(); } });
  $('folder-link').addEventListener('click', e => { e.stopPropagation(); di.click(); });
  ['dragenter', 'dragover'].forEach(t => dz.addEventListener(t, e => { e.preventDefault(); dz.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(t => dz.addEventListener(t, e => { e.preventDefault(); dz.classList.remove('over'); }));
  dz.addEventListener('drop', e => pickFiles(e.dataTransfer.files));
  fi.addEventListener('change', () => pickFiles(fi.files));
  di.addEventListener('change', () => pickFiles(di.files));
  $('to-config').addEventListener('click', () => { goStep(2); maybeAutoMap(); });
}

async function pickFiles(list) {
  if (!list || !list.length) return;
  alertBox('upload-error', '');
  try {
    if (list.length > 1 || list[0].webkitRelativePath) {
      const blob = await buildZip(list);
      await uploadFile(new File([blob], 'upload.zip', {type: 'application/zip'}), `${list.length} files`);
    } else {
      await uploadFile(list[0]);
    }
  } catch (err) { alertBox('upload-error', err.message); }
}

async function uploadFile(file, label) {
  $('file-chip-text').textContent = `Uploading ${label || file.name}…`;
  const fd = new FormData();
  fd.append('file', file);
  const data = await api('/api/upload', {method: 'POST', body: fd});
  S.demo = null;
  onDataset(data, `${label || file.name} · ${(file.size / 1024).toFixed(1)} KB`);
}

async function loadDemos() {
  try {
    const {demos} = await api('/api/demos');
    $('demo-grid').innerHTML = demos.map(d => `
      <button class="demo" data-demo="${esc(d.id)}">
        <span class="demo-domain">${esc(d.domain)}</span>
        <span class="demo-title">${esc(d.title)}</span>
        <span class="demo-desc">${esc(d.description)}</span>
        <span class="demo-src">${esc(d.source)}</span>
      </button>`).join('') || '<div class="hint">No demo datasets installed.</div>';
    $('demo-grid').querySelectorAll('.demo').forEach(b => b.addEventListener('click', () => loadDemo(b)));
  } catch (err) { $('demo-grid').innerHTML = `<div class="hint">Could not load demos: ${esc(err.message)}</div>`; }
}

async function loadDemo(btn) {
  alertBox('upload-error', '');
  btn.classList.add('loading');
  try {
    const data = await api(`/api/demo/${encodeURIComponent(btn.dataset.demo)}`, {method: 'POST'});
    S.demo = data.demo;
    onDataset(data, `Demo · ${data.demo.title}`);
    applyDemo(data.demo);
    goStep(2);
    maybeAutoMap();
  } catch (err) { alertBox('upload-error', err.message); }
  finally { btn.classList.remove('loading'); }
}

function onDataset(data, label) {
  S.jobId = data.job_id;
  S.columns = data.columns || [];
  S.maps = {}; S.run = null; S.comparison = null; S.splitLabels = {};
  maxStep = 1;
  resetExplore();
  if (data.load_error) alertBox('upload-error', data.load_error);

  $('file-chip-text').textContent = label;
  const st = data.stats || {};
  const tiles = [];
  if (st.total_rows != null) tiles.push([fmtInt(st.total_rows), 'rows']);
  if (st.total_columns != null) tiles.push([st.total_columns, 'columns']);
  if (data.format && data.format !== 'unknown') tiles.push([data.format.toUpperCase(), 'format']);
  for (const [col, ci] of Object.entries(st.columns || {}))
    tiles.push([fmtInt(ci.unique), `unique ${TYPE_LABEL[ci.type] || ci.type} · ${col}`]);
  $('stats').innerHTML = tiles.map(([v, l]) =>
    `<div class="stat"><div class="stat-val">${esc(v)}</div><div class="stat-label">${esc(l)}</div></div>`).join('');

  const pv = data.preview;
  $('preview-table').innerHTML = pv && pv.rows && pv.rows.length
    ? '<thead><tr>' + pv.columns.map(c => `<th>${esc(c)}</th>`).join('') + '</tr></thead><tbody>' +
      pv.rows.map(r => '<tr>' + r.map(v => `<td title="${esc(v)}">${esc(v)}</td>`).join('') + '</tr>').join('') + '</tbody>'
    : '<tbody><tr><td>No preview available</td></tr></tbody>';

  fillColumns();
  applyHints(data.hints);
  $('detect-text').textContent = describeDetection(data.hints);
  $('dataset-name').value = (S.demo && S.demo.dataset_name) ||
    (data.filename || 'dataset').replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_').slice(0, 40) || 'dataset';
  show('dataset-card');
}

function describeDetection(h) {
  if (!h || !h.suggested_e) return 'Pick entity columns on the next step.';
  const parts = [h.suggested_e, h.suggested_f].filter(Boolean)
    .map(s => `${s.column} (${TYPE_LABEL[s.type] || s.type})`);
  return 'Detected: ' + parts.join(' + ');
}

function fillColumns() {
  for (const ax of ['e1', 'e2']) {
    $(`${ax}-col`).innerHTML = S.columns.length
      ? S.columns.map(c => `<option value="${esc(c)}">${esc(c)}</option>`).join('')
      : '<option value="">— no columns detected —</option>';
  }
  if (S.columns.length > 1) $('e2-col').value = S.columns[1];
}

function applyHints(h) {
  if (!h || !h.suggested_e) { setMode(S.mode); return; }
  applyEntity('e1', {name: h.suggested_e.name, type: h.suggested_e.type, column: h.suggested_e.column});
  if (h.suggested_f) applyEntity('e2', {name: h.suggested_f.name, type: h.suggested_f.type, column: h.suggested_f.column});
  setMode(h.suggested_f ? '2d' : '1d');
}

function applyEntity(ax, spec) {
  if (spec.type) $(`${ax}-type`).value = spec.type;
  if (spec.column && S.columns.includes(spec.column)) $(`${ax}-col`).value = spec.column;
  if (spec.name) $(`${ax}-name`).value = spec.name;
  renderFeatures(ax, spec.feature_sets);
}

function applyDemo(d) {
  applyEntity('e1', d.e1);
  if (d.e2) applyEntity('e2', d.e2);
  setMode(d.e2 ? '2d' : '1d', d.techniques);
  if (d.dataset_name) $('dataset-name').value = d.dataset_name;
}

// ZIP (store-only) of a multi-file / folder selection
async function buildZip(fileList) {
  const enc = new TextEncoder(), parts = [], central = [];
  let offset = 0;
  const crcTable = new Uint32Array(256).map((_, n) => {
    let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; return c >>> 0;
  });
  const crc32 = buf => { let c = 0xFFFFFFFF; for (const b of buf) c = crcTable[(c ^ b) & 0xFF] ^ (c >>> 8); return (c ^ 0xFFFFFFFF) >>> 0; };
  for (const f of fileList) {
    const name = enc.encode(f.webkitRelativePath || f.name);
    const data = new Uint8Array(await f.arrayBuffer());
    const crc = crc32(data);
    const loc = new Uint8Array(30 + name.length), lv = new DataView(loc.buffer);
    lv.setUint32(0, 0x04034b50, true); lv.setUint16(4, 20, true); lv.setUint32(14, crc, true);
    lv.setUint32(18, data.length, true); lv.setUint32(22, data.length, true); lv.setUint16(26, name.length, true);
    loc.set(name, 30);
    const cd = new Uint8Array(46 + name.length), cv = new DataView(cd.buffer);
    cv.setUint32(0, 0x02014b50, true); cv.setUint16(4, 20, true); cv.setUint16(6, 20, true); cv.setUint32(16, crc, true);
    cv.setUint32(20, data.length, true); cv.setUint32(24, data.length, true); cv.setUint16(28, name.length, true);
    cv.setUint32(42, offset, true); cd.set(name, 46);
    parts.push(loc, data); central.push(cd); offset += loc.length + data.length;
  }
  const size = central.reduce((a, c) => a + c.length, 0);
  const end = new Uint8Array(22), ev = new DataView(end.buffer);
  ev.setUint32(0, 0x06054b50, true); ev.setUint16(8, central.length, true); ev.setUint16(10, central.length, true);
  ev.setUint32(12, size, true); ev.setUint32(16, offset, true);
  return new Blob([...parts, ...central, end], {type: 'application/zip'});
}

// ── Step 2: configure ───────────────────────────────────────────────────────

function initConfig() {
  $('mode-seg').querySelectorAll('button').forEach(b => b.addEventListener('click', () => setMode(b.dataset.mode)));
  for (const ax of ['e1', 'e2']) {
    $(`${ax}-type`).addEventListener('change', () => { renderFeatures(ax); buildTechniques(); onEntityChanged(ax); });
    $(`${ax}-col`).addEventListener('change', () => onEntityChanged(ax));
    $(`${ax}-name`).addEventListener('input', renderExploreAxis);
  }
  $('run-btn').addEventListener('click', startRun);
  renderFeatures('e1'); renderFeatures('e2');
}

function setMode(mode, techniques) {
  S.mode = mode;
  $('mode-seg').querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.mode === mode));
  show('entity-e2', mode === '2d');
  show('incomplete-opt', mode === '2d');
  $('e1-title').textContent = mode === '2d' ? 'First entity' : 'Entity';
  $('mode-hint').textContent = mode === '2d'
    ? 'Interaction data (e.g. drug–target pairs): each row links two entities, and leakage is controlled on both axes.'
    : 'Property data (e.g. molecule → solubility): each row is one entity, split by similarity.';
  buildTechniques(techniques);
  if (S.exploreAxis === 'e2' && mode === '1d') S.exploreAxis = 'e1';
  renderExploreAxis();
}

function renderFeatures(ax, selected) {
  const type = $(`${ax}-type`).value;
  const chosen = selected && selected.length ? selected : FEAT_DEFAULTS[type] || [];
  $(`${ax}-feats`).innerHTML = (FEATURES[type] || []).map(f => `
    <label class="pill" title="${esc(f.desc)}">
      <input type="checkbox" name="${ax}-feat" value="${esc(f.id)}" ${chosen.includes(f.id) ? 'checked' : ''}>${esc(f.label)}
    </label>`).join('');
  $(`${ax}-feats`).querySelectorAll('input').forEach(i => i.addEventListener('change', () => onEntityChanged(ax)));
}

function buildTechniques(selected) {
  const groups = S.mode === '2d' ? TECH_2D : TECH_1D;
  const prev = new Set(selected || [...document.querySelectorAll('input[name="tech"]:checked')].map(i => i.value));
  const use = prev.size ? prev : new Set(TECH_DEFAULTS[S.mode]);
  const isMol = $('e1-type').value === 'molecule';
  $('tech-groups').innerHTML = groups.map(([label, ids]) => {
    const items = ids.filter(t => t !== 'scaffold' || isMol);
    if (!items.length) return '';
    return `<div class="tech-group"><div class="hint" style="margin-bottom:.35rem">${esc(label)}</div><div class="tech-list">` +
      items.map(t => `
        <label class="tech">
          <input type="checkbox" name="tech" value="${t}" ${use.has(t) ? 'checked' : ''}>
          <div>
            <div class="tech-name">${esc(TECH[t].name)} <span class="tag">${t}</span>${TECH[t].isNew ? '<span class="tag new">new</span>' : ''}</div>
            <div class="tech-desc">${esc(TECH[t].desc)}</div>
          </div>
        </label>`).join('') + '</div></div>';
  }).join('');
}

function entitySpec(ax) {
  return {
    name: $(`${ax}-name`).value.trim(),
    type: $(`${ax}-type`).value,
    column: $(`${ax}-col`).value,
    feature_sets: [...document.querySelectorAll(`input[name="${ax}-feat"]:checked`)].map(i => i.value),
  };
}

function buildPayload() {
  const e1 = entitySpec('e1');
  const techniques = [...document.querySelectorAll('input[name="tech"]:checked')].map(i => i.value);
  if (!S.jobId) throw new Error('Load a dataset first.');
  if (!e1.column) throw new Error('Choose a column for the entity.');
  if (!e1.name) throw new Error('Give the entity a name.');
  if (!e1.feature_sets.length) throw new Error('Select at least one feature set.');
  if (!techniques.length) throw new Error('Select at least one method.');
  const payload = {dataset_name: $('dataset-name').value.trim().replace(/[^\w-]+/g, '_') || 'dataset',
                   e1, splitting: {techniques, ratio: $('ratio').value,
                                   allow_incomplete: S.mode === '2d' && $('allow-incomplete').checked}};
  if (S.mode === '2d') {
    const e2 = entitySpec('e2');
    if (e2.column === e1.column) throw new Error('The two entities must use different columns.');
    if (!e2.name || e2.name === e1.name) throw new Error('The two entities need different, non-empty names.');
    if (!e2.feature_sets.length) throw new Error('Select at least one feature set for the second entity.');
    payload.e2 = e2;
  }
  return payload;
}

// ── Data map (explore) ──────────────────────────────────────────────────────

function resetExplore() {
  S.exploreAxis = 'e1';
  if (window.Plotly) Plotly.purge('explore-plot');
  show('explore-empty');
  $('explore-msg').textContent = 'Compute a 2-D map of the selected entity column using the features chosen on the left.';
  $('explore-legend').innerHTML = ''; $('explore-meta').textContent = '';
  $('color-by').innerHTML = ''; show('highlight', false);
}

function renderExploreAxis() {
  const seg = $('explore-axis');
  if (S.mode !== '2d') { show(seg, false); return; }
  show(seg);
  seg.innerHTML = ['e1', 'e2'].map(ax =>
    `<button data-ax="${ax}" class="${S.exploreAxis === ax ? 'on' : ''}">${esc($(`${ax}-name`).value || ax)}</button>`).join('');
  seg.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    S.exploreAxis = b.dataset.ax; renderExploreAxis(); onEntityChanged(S.exploreAxis, true);
  }));
}

function onEntityChanged(ax, force) {
  if (ax !== S.exploreAxis && !force) return;
  const spec = entitySpec(S.exploreAxis);
  const cached = Object.values(S.maps).find(m => sameSpec(m, spec));
  if (cached) { drawExplore(cached); return; }
  if (window.Plotly) Plotly.purge('explore-plot');
  show('explore-empty');
  $('explore-legend').innerHTML = ''; $('explore-meta').textContent = '';
  $('explore-msg').textContent = 'Settings changed — compute the map to see this entity.';
  if (S.demo) computeMap();
}

function sameSpec(m, spec) {
  return m.column === spec.column && m.type === spec.type &&
    [...m.feature_sets].sort().join() === [...spec.feature_sets].sort().join();
}

function maybeAutoMap() {
  // Demos (features cached) map immediately; user data waits for the button
  // because featurization (e.g. ESM2 on CPU) can take a while.
  if (S.demo) computeMap();
}

async function computeMap() {
  if (!S.jobId) return;
  const spec = entitySpec(S.exploreAxis);
  if (!spec.column || !spec.feature_sets.length) {
    $('explore-msg').textContent = 'Choose a column and at least one feature set first.'; return;
  }
  const token = (computeMap.token = (computeMap.token || 0) + 1);
  show('explore-empty');
  const slow = spec.feature_sets.some(f => /embedding/.test(f)) && !S.info.gpu;
  $('explore-msg').innerHTML = `<span class="spinner"></span>&nbsp; Computing ${esc(TYPE_LABEL[spec.type] || spec.type)} embedding…` +
    (slow ? '<div class="hint" style="margin-top:.5rem">Language-model embeddings run on CPU on this server and may take several minutes for new data (cached afterwards).</div>' : '');
  $('map-btn').disabled = true;
  try {
    const m = await fetchMap(S.jobId, {column: spec.column, type: spec.type, feature_sets: spec.feature_sets},
      msg => { if (token === computeMap.token) $('explore-msg').innerHTML = `<span class="spinner"></span>&nbsp; ${esc(msg)}`; });
    if (token === computeMap.token) drawExplore(m);
  } catch (err) {
    if (token === computeMap.token) $('explore-msg').textContent = 'Could not compute map: ' + err.message;
  } finally { $('map-btn').disabled = false; }
}

async function fetchMap(datasetId, req, onMsg) {
  let st = await api(`/api/map/${datasetId}`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(req)});
  const key = st.key;
  while (true) {
    if (st.status === 'failed') throw new Error(st.error || 'failed');
    if (st.status === 'completed') {
      const full = st.map ? st : await api(`/api/map/${datasetId}/${key}`);
      if (full.map) { S.maps[key] = full.map; return full.map; }
    }
    if (onMsg && st.message) onMsg(st.message);
    await new Promise(r => setTimeout(r, 1000));
    st = await api(`/api/map/${datasetId}/${key}`);
  }
}

function drawExplore(m) {
  S.exploreMap = m;
  const sel = $('color-by');
  const attrs = Object.keys(m.attributes || {});
  const preferred = S.demo && S.demo.color_by && S.demo.color_by[S.exploreAxis];
  const prev = sel.value;
  sel.innerHTML = '<option value="">None</option>' + attrs.map(a => `<option value="${esc(a)}">${esc(a)}</option>`).join('');
  sel.value = attrs.includes(prev) ? prev : attrs.includes(preferred) ? preferred : (attrs[0] || '');
  sel.onchange = () => { S.highlight = null; renderExploreColors(); };
  $('highlight').onchange = () => { S.highlight = $('highlight').value; renderExploreColors(); };
  S.highlight = null;
  renderExploreColors();
  $('explore-meta').textContent =
    `${m.embedding} · ${fmtInt(m.dims)}-d → t-SNE · ${fmtInt(m.n_shown)}` +
    (m.n_shown < m.n_entities ? ` of ${fmtInt(m.n_entities)} entities (random sample)` : ' entities');
}

function renderExploreColors() {
  const m = S.exploreMap; if (!m) return;
  const col = $('color-by').value;
  const attr = col ? m.attributes[col] : null;
  const hl = $('highlight');
  show(hl, false);
  const hover = hoverText(m);
  if (!attr) {
    plotScatter('explore-plot', [trace(m, idx(m), css('--series-1'), 'entities', hover)]);
    $('explore-legend').innerHTML = '';
    return;
  }
  if (attr.kind === 'numeric') {
    const vals = attr.values;
    const finite = vals.filter(v => v != null);
    const lo = Math.min(...finite), hi = Math.max(...finite);
    const ramp = sequentialRamp();
    const ids = idx(m).filter(i => vals[i] != null), miss = idx(m).filter(i => vals[i] == null);
    const traces = [];
    if (miss.length) traces.push(trace(m, miss, css('--other'), 'no value', hover));
    traces.push({
      ...trace(m, ids, null, col, hover.map((h, i) => `${h}<br>${esc(col)}: ${fmtNum(vals[i])}`)),
      marker: {size: markerSize(m), color: ids.map(i => vals[i]), colorscale: ramp, cmin: lo, cmax: hi,
               line: {width: 1, color: css('--chart-surface')}, opacity: .9},
    });
    plotScatter('explore-plot', traces);
    $('explore-legend').innerHTML = `<div class="colorbar"><span>${esc(col)}</span><span>${fmtNum(lo)}</span>
      <span class="ramp" style="background:linear-gradient(90deg,${ramp.map(r => r[1]).join(',')})"></span><span>${fmtNum(hi)}</span>
      ${miss.length ? `<span class="legend-item"><span class="swatch" style="background:${css('--other')}"></span>no value (${fmtInt(miss.length)})</span>` : ''}</div>`;
    return;
  }
  // categorical
  const counts = {};
  attr.values.forEach(v => { const k = v ?? '—'; counts[k] = (counts[k] || 0) + 1; });
  const cats = Object.keys(counts).sort((a, b) => counts[b] - counts[a] || a.localeCompare(b));
  const withVal = hover.map((h, i) => `${h}<br>${esc(col)}: ${esc(attr.values[i] ?? '—')}`);
  if (cats.length <= 3) {
    const traces = cats.map((c, k) => trace(m, idx(m).filter(i => (attr.values[i] ?? '—') === c),
      css(`--series-${k + 1}`), c, withVal));
    plotScatter('explore-plot', traces);
    $('explore-legend').innerHTML = legendHtml(cats.map((c, k) => [css(`--series-${k + 1}`), c, counts[c]]));
    return;
  }
  // More than three classes: highlight one class against the rest (a scatter
  // cannot keep >3 hues distinguishable for colour-vision-deficient readers).
  if (!S.highlight || !cats.includes(S.highlight)) S.highlight = cats[0];
  hl.innerHTML = cats.map(c => `<option value="${esc(c)}">${esc(c)} (${counts[c]})</option>`).join('');
  hl.value = S.highlight; show(hl);
  const on = idx(m).filter(i => (attr.values[i] ?? '—') === S.highlight);
  const off = idx(m).filter(i => (attr.values[i] ?? '—') !== S.highlight);
  plotScatter('explore-plot', [trace(m, off, css('--other'), 'other classes', withVal),
                               trace(m, on, css('--series-1'), S.highlight, withVal)]);
  $('explore-legend').innerHTML = legendHtml([[css('--series-1'), S.highlight, counts[S.highlight]],
    [css('--other'), `other ${cats.length - 1} classes`, m.n_shown - counts[S.highlight]]]) +
    '<div class="hint" style="margin-top:.3rem">Pick a class in the second menu to highlight it.</div>';
}

// ── Plot primitives ─────────────────────────────────────────────────────────

const idx = m => m.x.map((_, i) => i);
const markerSize = m => (m.n_shown > 2500 ? 5 : m.n_shown > 800 ? 7 : 9);
function fmtNum(v) {
  if (v == null || Number.isNaN(v)) return '—';
  const a = Math.abs(v);
  return a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(2) : (+v.toPrecision(4)).toLocaleString();
}
function hoverText(m) {
  return m.ids.map((id, i) => {
    const lab = m.labels && m.labels[i];
    return lab ? `<b>${esc(lab)}</b><br><span>${esc(shortId(id, 60))}</span>` : `<b>${esc(shortId(id, 60))}</b>`;
  });
}
function trace(m, ids, color, name, hover) {
  return {
    type: 'scatter', mode: 'markers', name: String(name),
    x: ids.map(i => m.x[i]), y: ids.map(i => m.y[i]),
    text: ids.map(i => hover[i]), hovertemplate: '%{text}<extra></extra>',
    marker: {size: markerSize(m), color, opacity: .85, line: {width: 0.8, color: css('--chart-surface')}},
  };
}
function sequentialRamp() {
  const dark = css('--chart-surface').toLowerCase() === '#1a1a19';
  const steps = dark ? ['#184f95', '#1c5cab', '#256abf', '#2a78d6', '#3987e5', '#5598e7', '#6da7ec', '#86b6ef', '#9ec5f4']
                     : ['#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#1c5cab', '#0d366b'];
  return steps.map((c, i) => [i / (steps.length - 1), c]);
}
function legendHtml(items) {
  return '<div class="legend">' + items.map(([c, l, n]) =>
    `<span class="legend-item"><span class="swatch" style="background:${c}"></span>${esc(l)}` +
    (n != null ? ` <span class="legend-count">${fmtInt(n)}</span>` : '') + '</span>').join('') + '</div>';
}
// Maps up to this many points use SVG scatter (animatable); larger ones WebGL.
const GL_THRESHOLD = 2500;

function plotScatter(el, traces, key) {
  if (!window.Plotly) { $(el.replace('-plot', '-msg')).textContent = 'Plotting library failed to load.'; return; }
  const node = $(el);
  show(el.replace('-plot', '-empty'), false);
  key = key || (el === 'explore-plot' && S.exploreMap ? S.exploreMap.key : el);
  const first = node._key !== key;
  node._key = key;

  const nPts = traces.reduce((a, t) => a + t.x.length, 0);
  const gl = nPts > GL_THRESHOLD;
  traces.forEach(t => { t.type = gl ? 'scattergl' : 'scatter'; });

  // Fixed, padded ranges (equal aspect) so the layout never jumps between
  // recolourings; uirevision keeps the user's zoom/pan across updates.
  const xs = traces.flatMap(t => t.x), ys = traces.flatMap(t => t.y);
  const pad = (lo, hi) => { const d = (hi - lo) * 0.06 || 1; return [lo - d, hi + d]; };
  const xr = pad(Math.min(...xs), Math.max(...xs)), yr = pad(Math.min(...ys), Math.max(...ys));
  const ax = {showgrid: false, zeroline: false, showticklabels: false, ticks: '', showline: false,
              title: {text: ''}, automargin: false};
  const layout = {
    margin: {l: 8, r: 8, t: 8, b: 8}, showlegend: false, hovermode: 'closest', dragmode: 'pan',
    uirevision: key,
    paper_bgcolor: css('--chart-surface'), plot_bgcolor: css('--chart-surface'),
    xaxis: {...ax, range: xr}, yaxis: {...ax, range: yr, scaleanchor: 'x', scaleratio: 1},
    hoverlabel: {bgcolor: css('--surface'), bordercolor: css('--axis'), font: {color: css('--ink'), family: css('--sans'), size: 12}},
    font: {family: css('--sans'), color: css('--ink-2')},
  };
  const config = {
    responsive: true, displaylogo: false, scrollZoom: true, doubleClick: 'reset',
    modeBarButtonsToRemove: ['select2d', 'lasso2d', 'zoom2d', 'autoScale2d', 'toggleSpikelines'],
    toImageButtonOptions: {format: 'png', filename: 'palm_tsne', scale: 2},
  };
  attachPinch(node);

  if (first && !gl && !REDUCED_MOTION) {
    // points start at the centre and spread out to their t-SNE positions
    const cx = (xr[0] + xr[1]) / 2, cy = (yr[0] + yr[1]) / 2;
    const start = traces.map(t => ({...t, x: t.x.map(() => cx), y: t.y.map(() => cy),
                                    marker: {...t.marker, opacity: 0}}));
    Plotly.react(node, start, layout, config).then(() =>
      Plotly.animate(node, {data: traces.map(t => ({x: t.x, y: t.y, marker: t.marker}))},
                     {transition: {duration: 800, easing: 'cubic-out'}, frame: {duration: 800, redraw: false}}));
    return;
  }
  if (!first && !REDUCED_MOTION) {           // recolour: quick cross-fade
    node.classList.remove('flash'); void node.offsetWidth; node.classList.add('flash');
  }
  Plotly.react(node, traces, layout, config);
}

// Safari reports trackpad pinches as gesture events instead of ctrl+wheel.
function attachPinch(node) {
  if (node._pinch) return;
  node._pinch = true;
  let base = null;
  const zoom = (r, f) => { const c = (r[0] + r[1]) / 2, h = (r[1] - r[0]) / 2 * f; return [c - h, c + h]; };
  node.addEventListener('gesturestart', e => {
    e.preventDefault();
    const l = node._fullLayout;
    if (l) base = {x: l.xaxis.range.slice(), y: l.yaxis.range.slice()};
  });
  node.addEventListener('gesturechange', e => {
    e.preventDefault();
    if (base) Plotly.relayout(node, {'xaxis.range': zoom(base.x, 1 / e.scale), 'yaxis.range': zoom(base.y, 1 / e.scale)});
  });
  node.addEventListener('gestureend', e => { e.preventDefault(); base = null; });
}

function resizePlot(el) { if (window.Plotly && $(el).data) Plotly.Plots.resize(el); }
function redrawPlots() {
  requestAnimationFrame(() => {
    if (S.exploreMap && $('explore-plot').data) renderExploreColors();
    if ($('result-plot').data) renderResultMap();
  });
}

// ── Runs: submit, follow, list ──────────────────────────────────────────────

const RUN_STATUS = {
  queued: 'Queued', running: 'Running', completed: 'Completed', failed: 'Failed',
  cancelled: 'Cancelled', interrupted: 'Interrupted',
};
const ACTIVE = new Set(['queued', 'running']);

async function startRun() {
  alertBox('config-error', '');
  let payload;
  try { payload = buildPayload(); } catch (err) { alertBox('config-error', err.message); return; }
  const btn = $('run-btn');
  btn.disabled = true;
  try {
    const r = await api(`/api/run/${S.jobId}`, {method: 'POST', headers: {'Content-Type': 'application/json'},
                                              body: JSON.stringify(payload)});
    toast(`Run submitted — ${methodsLabel(payload.splitting.techniques)}`, 'info');
    refreshRuns();
    await openRun(r.run_id);
  } catch (err) { alertBox('config-error', 'Could not submit: ' + err.message); }
  finally { btn.disabled = false; }
}

function methodsLabel(techs) {
  const names = techs.map(t => (TECH[t] ? TECH[t].name : t));
  return names.length > 3 ? `${names.slice(0, 3).join(', ')} +${names.length - 3}` : names.join(', ');
}

// Open one run in the Run / Results views.
async function openRun(runId) {
  const run = await api(`/api/runs/${runId}`);
  S.runId = run.id; S.run = run.request; S.runDataset = run.dataset_id; S.runMeta = run;
  S.comparison = null; S.splitLabels = {};
  $('run-title').textContent = run.label || 'Run';
  $('run-sub').textContent = `${methodsLabel(run.request.splitting.techniques)} · ${run.request.splitting.ratio.replace(/\//g, ' / ')}` +
    (run.request.e2 ? ` · ${run.request.e1.name} × ${run.request.e2.name}` : ` · ${run.request.e1.name}`);
  maxStep = Math.max(maxStep, 3);
  stopFollowing();
  $('console').innerHTML = ''; logSeen = 0;
  show('to-results', run.status === 'completed');
  if (run.status === 'completed') {
    renderRunState(run, true);
    await loadResults();
    goStep(4);
  } else {
    goStep(3);
    renderRunState(run, true);
    if (ACTIVE.has(run.status)) follow(run.id);
    else (run.log || []).forEach(l => log(l));
  }
}

function stopFollowing() {
  if (S.evt) { S.evt.close(); S.evt = null; }
  clearTimeout(S.pollTimer); clearInterval(S.clock);
}

function follow(runId) {
  S.evt = new EventSource(`/api/runs/${runId}/stream?client_id=${encodeURIComponent(CLIENT_ID)}`);
  S.evt.onmessage = e => { if (S.runId === runId) onProgress(JSON.parse(e.data)); };
  S.evt.onerror = () => { if (S.evt) S.evt.close(); S.evt = null; if (S.runId === runId) poll(runId); };
  clearInterval(S.clock);
  S.clock = setInterval(() => renderElapsed(), 1000);
}

let logSeen = 0;
async function poll(runId) {
  clearTimeout(S.pollTimer);
  try {
    const run = await api(`/api/runs/${runId}`);
    if (S.runId !== runId) return;
    onProgress({...run, log: (run.log || []).slice(logSeen)});
    if (ACTIVE.has(run.status)) S.pollTimer = setTimeout(() => poll(runId), 2000);
  } catch (_) { S.pollTimer = setTimeout(() => poll(runId), 4000); }
}

function onProgress(d) {
  if (d.error && !d.status) { log('✗ ' + d.error); return; }
  (d.log || []).forEach(l => log(l));
  S.runMeta = {...S.runMeta, ...d, log: undefined};
  renderRunState(S.runMeta);
  if (!ACTIVE.has(d.status)) {
    stopFollowing();
    renderElapsed();
    if (d.status === 'completed') {
      show('to-results');
      // jump to results only if the user is still watching this run's progress
      if ($('p3').classList.contains('active'))
        loadResults().then(() => goStep(4)).catch(err => log('✗ Could not load results: ' + err.message));
    }
  }
}

function renderRunState(run, reset) {
  const st = run.status;
  setBadge(st, st === 'queued' && run.queue_position ? `Queued · #${run.queue_position} in line` : RUN_STATUS[st] || st);
  const pct = st === 'completed' ? 100 : run.progress || 0;
  setProgress(pct, st === 'queued' ? 'Waiting for a free worker…' : run.phase || '', reset);
  $('progress-fill').classList.toggle('live', st === 'running');
  show('cancel-run', ACTIVE.has(st));
  renderElapsed();
}

function renderElapsed() {
  const r = S.runMeta;
  if (!r) return;
  const now = Date.now() / 1000;
  const end = r.finished || now;
  if (r.status === 'queued' || !r.started) { $('run-elapsed').textContent = `submitted ${ago(r.created)}`; return; }
  $('run-elapsed').textContent = `${dur(end - r.started)} ${r.finished ? 'total' : 'elapsed'}`;
}

function dur(sec) {
  sec = Math.max(0, Math.round(sec));
  const h = Math.floor(sec / 3600), m = Math.floor(sec / 60) % 60, s2 = sec % 60;
  return h ? `${h}h ${m}m` : `${m}:${String(s2).padStart(2, '0')}`;
}
function ago(ts) {
  const d = Date.now() / 1000 - ts;
  if (d < 60) return 'just now';
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

function log(line) {
  logSeen++;
  const box = $('console'), div = document.createElement('div');
  const t = line.trim();
  div.className = t.startsWith('✗') || /FAILED|Warning|error/i.test(t) ? 'err'
    : t.startsWith('✓') || /^Done!/.test(t) ? 'ok' : /^\[\d\/5\]|^===|Recommended/.test(t) ? 'head' : '';
  div.textContent = line;
  const stick = box.scrollTop + box.clientHeight >= box.scrollHeight - 30;
  box.appendChild(div);
  if (stick) box.scrollTop = box.scrollHeight;
}

function setProgress(pct, label, instant) {
  const fill = $('progress-fill');
  if (instant) { fill.style.transition = 'none'; requestAnimationFrame(() => { fill.style.transition = ''; }); }
  fill.style.width = pct + '%';
  countTo($('progress-pct'), pct, '%');
  if (label) $('progress-label').textContent = label;
  document.querySelectorAll('.phase').forEach((p, i, all) => {
    const at = +p.dataset.at, next = all[i + 1] ? +all[i + 1].dataset.at : 101;
    p.classList.toggle('done', pct >= next);
    p.classList.toggle('on', pct >= at && pct < next);
  });
}

function setBadge(kind, text) { $('run-badge').className = 'badge ' + kind; $('run-badge-text').textContent = text; }

async function cancelRun(runId) {
  try {
    await api(`/api/runs/${runId}/cancel`, {method: 'POST'});
    toast('Cancellation requested', 'info');
    refreshRuns();
  } catch (err) { toast('Could not cancel: ' + err.message, 'error'); }
}

async function deleteRun(runId) {
  try {
    await api(`/api/runs/${runId}`, {method: 'DELETE'});
    if (S.runId === runId) { S.runId = null; S.run = null; maxStep = Math.min(maxStep, 2); }
    refreshRuns();
  } catch (err) { toast('Could not delete: ' + err.message, 'error'); }
}

// Re-load a run's dataset and settings into Configure, ready to tweak and resubmit.
async function editRun(runId) {
  try {
    const run = await api(`/api/runs/${runId}`);
    const data = await api(`/api/datasets/${run.dataset_id}`);
    S.demo = data.demo || null;
    onDataset(data, run.label || data.filename);
    applyRequest(run.request);
    goStep(2);
    computeMap();
  } catch (err) { toast('Could not load that run: ' + err.message, 'error'); }
}

function applyRequest(req) {
  applyEntity('e1', req.e1);
  if (req.e2) applyEntity('e2', req.e2);
  setMode(req.e2 ? '2d' : '1d', req.splitting.techniques);
  $('ratio').value = req.splitting.ratio;
  $('dataset-name').value = req.dataset_name;
  $('allow-incomplete').checked = !!req.splitting.allow_incomplete;
}

// ── Runs panel + background poller ──────────────────────────────────────────

S.runsList = []; S.knownStatus = {};
async function refreshRuns() {
  clearTimeout(S.runsTimer);
  try {
    const {runs, max_concurrent} = await api('/api/runs');
    for (const r of runs) {                   // completion notifications
      const prev = S.knownStatus[r.id];
      if (prev && ACTIVE.has(prev) && !ACTIVE.has(r.status)) notifyFinished(r);
      S.knownStatus[r.id] = r.status;
    }
    S.runsList = runs; S.maxConcurrent = max_concurrent;
    renderRunsBadge();
    if ($('p5').classList.contains('active')) renderRunsPanel();
  } catch (_) { /* server restarting; try again */ }
  const busy = S.runsList.some(r => ACTIVE.has(r.status));
  S.runsTimer = setTimeout(refreshRuns, document.hidden ? 15000 : busy || $('p5').classList.contains('active') ? 2500 : 10000);
}

function notifyFinished(r) {
  const what = r.label || 'Run';
  if (r.status === 'completed') toast(`${what} finished`, 'good', {label: 'View results', run: r.id});
  else if (r.status === 'failed') toast(`${what} failed`, 'error', {label: 'Details', run: r.id});
}

function renderRunsBadge() {
  const active = S.runsList.filter(r => ACTIVE.has(r.status)).length;
  const b = $('runs-count');
  b.textContent = active || S.runsList.length || '';
  b.classList.toggle('live', active > 0);
  show(b, S.runsList.length > 0);
}

function renderRunsPanel() {
  const runs = S.runsList;
  const active = runs.filter(r => ACTIVE.has(r.status)).length;
  $('runs-sub').textContent = runs.length
    ? `${runs.length} run${runs.length === 1 ? '' : 's'} · ${active} active · up to ${S.maxConcurrent || 2} execute at once, the rest wait in line`
    : '';
  show('runs-empty', !runs.length);
  const list = $('runs-list');
  const seen = new Set();
  runs.forEach((r, i) => {
    seen.add(r.id);
    let row = list.querySelector(`[data-run="${r.id}"]`);
    if (!row) {
      row = document.createElement('div');
      row.className = 'run-row';
      row.dataset.run = r.id;
      row.style.setProperty('--i', Math.min(i, 8));
      list.appendChild(row);
    }
    row.classList.toggle('viewing', r.id === S.runId);
    const html = runRowHtml(r);
    if (row._html !== html) { row.innerHTML = html; row._html = html; }
    if (list.children[i] !== row) list.insertBefore(row, list.children[i]);
  });
  [...list.children].forEach(el => { if (!seen.has(el.dataset.run)) el.remove(); });
}

function runRowHtml(r) {
  const techs = r.request.splitting.techniques;
  const ent = r.request.e2 ? `${r.request.e1.name} × ${r.request.e2.name}` : r.request.e1.name;
  const live = ACTIVE.has(r.status);
  const pct = r.status === 'completed' ? 100 : r.progress || 0;
  const timing = r.status === 'queued' ? `#${r.queue_position || '?'} in line`
    : r.started ? dur((r.finished || Date.now() / 1000) - r.started) : '';
  return `
    <div class="run-main">
      <span class="badge ${r.status}"><span class="bdot"></span>${esc(RUN_STATUS[r.status] || r.status)}</span>
      <div class="run-info">
        <div class="run-name">${esc(r.label || 'Run')} <span class="hint">· ${esc(ent)} · ${esc(r.request.splitting.ratio)}</span></div>
        <div class="run-tags">${techs.map(t => `<span class="tag">${esc(t)}</span>`).join('')}</div>
      </div>
      <div class="run-when"><div>${esc(ago(r.created))}</div><div class="hint">${esc(timing)}</div></div>
      <div class="run-actions">
        <button class="btn btn-secondary btn-sm" data-act="open">${r.status === 'completed' ? 'Results' : 'Open'}</button>
        <button class="btn btn-ghost btn-sm" data-act="edit" title="Load this dataset and settings into Configure">Edit &amp; rerun</button>
        ${live ? '<button class="btn btn-ghost btn-sm" data-act="cancel">Cancel</button>'
               : '<button class="btn btn-ghost btn-sm" data-act="delete" title="Delete this run and its outputs">Delete</button>'}
      </div>
    </div>
    ${live ? `<div class="run-progress"><div class="progress"><div class="progress-fill ${r.status === 'running' ? 'live' : ''}" style="width:${pct}%"></div></div>
      <span class="hint">${esc(r.status === 'queued' ? 'Waiting for a free worker' : r.phase || '')} · ${pct}%</span></div>` : ''}
    ${r.status === 'failed' && r.error ? `<div class="run-error">${esc(r.error)}</div>` : ''}`;
}

function initRunsPanel() {
  $('runs-btn').addEventListener('click', () => { goStep(5); renderRunsPanel(); refreshRuns(); });
  $('runs-list').addEventListener('click', e => {
    const btn = e.target.closest('button[data-act]');
    if (!btn) return;
    const id = btn.closest('.run-row').dataset.run;
    ({open: () => openRun(id), edit: () => editRun(id), cancel: () => cancelRun(id),
      delete: () => deleteRun(id)})[btn.dataset.act]();
  });
  $('runs-new').addEventListener('click', () => goStep(S.jobId ? 2 : 1));
  $('cancel-run').addEventListener('click', () => S.runId && cancelRun(S.runId));
  $('another-run').addEventListener('click', () => {
    if (S.runDataset && S.runDataset !== S.jobId) editRun(S.runId); else goStep(2);
  });
  document.addEventListener('visibilitychange', () => { if (!document.hidden) refreshRuns(); });
}

// ── Toasts ──────────────────────────────────────────────────────────────────

function toast(msg, kind = 'info', action) {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.innerHTML = `<span class="tdot"></span><span class="tmsg">${esc(msg)}</span>` +
    (action ? `<button class="linkish">${esc(action.label)}</button>` : '') +
    '<button class="tclose" aria-label="Dismiss">×</button>';
  const close = () => { el.classList.add('out'); setTimeout(() => el.remove(), 260); };
  el.querySelector('.tclose').onclick = close;
  if (action) el.querySelector('.linkish').onclick = () => { close(); openRun(action.run); };
  $('toasts').appendChild(el);
  setTimeout(close, action ? 9000 : 4000);
}

// Animate a number's text from its current value to ``to``.
function countTo(el, to, suffix = '') {
  const from = parseFloat(el.textContent) || 0;
  if (REDUCED_MOTION || from === to) { el.textContent = to + suffix; return; }
  const t0 = performance.now(), d = 450;
  cancelAnimationFrame(el._raf);
  const step = now => {
    const k = Math.min(1, (now - t0) / d), e = 1 - Math.pow(1 - k, 3);
    el.textContent = Math.round(from + (to - from) * e) + suffix;
    if (k < 1) el._raf = requestAnimationFrame(step);
  };
  el._raf = requestAnimationFrame(step);
}

// ── Step 4: results ─────────────────────────────────────────────────────────

async function loadResults() {
  const ds = S.run.dataset_name, id = S.runId;
  const [files, metrics] = await Promise.all([api(`/api/runs/${id}/files`), api(`/api/runs/${id}/metrics`)]);
  const cmp = (metrics.metrics || {})[`comparison_${ds}`] || {};
  S.comparison = cmp;
  const status = cmp.status || {};
  const valid = Object.keys(status).filter(t => status[t] === 'valid');
  const failed = cmp.failed || {};
  const axes = S.run.e2 ? ['e1', 'e2'] : ['e1'];

  $('results-sub').textContent = `${S.runMeta.label || 'Run'} · ${methodsLabel(S.run.splitting.techniques)} · finished ${ago(S.runMeta.finished || S.runMeta.created)}`;
  renderRecommendation(cmp, valid, axes);
  const per = metrics.metrics || {};
  const nRows = (() => { for (const t of valid) { const m = per[`${t}_${ds}`]; if (m && m.n_rows) return m.n_rows; } return null; })();
  const tiles = [
    nRows != null ? [nRows, 'rows split', true] : null,
    [valid.length, `valid split${valid.length === 1 ? '' : 's'}`, true],
    [Object.keys(failed).length, 'failed / rejected', true],
    [S.run.splitting.ratio.replace(/\//g, ' / '), 'target proportions', false],
  ].filter(Boolean);
  $('tiles').innerHTML = tiles.map(([v, l, n], i) =>
    `<div class="tile" style="--i:${i}"><div class="tile-val" ${n ? `data-count="${v}"` : ''}>${n ? 0 : esc(v)}</div><div class="tile-label">${esc(l)}</div></div>`).join('');
  $('tiles').querySelectorAll('[data-count]').forEach(el => countUp(el, +el.dataset.count));

  S.timings = metrics.timings || {};
  renderComparison(cmp, axes, S.timings);
  renderDownloads(files, ds, valid);
  $('figs').innerHTML = (files.plots || []).map(p =>
    `<div class="${p.startsWith('comparison_') ? 'wide' : ''}"><img loading="lazy" src="/api/runs/${id}/plot/${encodeURIComponent(p)}?client_id=${encodeURIComponent(CLIENT_ID)}" alt="${esc(p)}"></div>`).join('')
    || '<div class="hint">No figures were produced.</div>';

  S.resultTech = cmp.recommendation && valid.includes(cmp.recommendation) ? cmp.recommendation : valid[0];
  // 2D: open on the axis the shown method actually separates (lowest overlap)
  const shown = (cmp.techniques || {})[S.resultTech] || {};
  S.resultAxis = axes.reduce((best, ax) =>
    (shown[`overlap_frac_${ax}`] ?? 1) < (shown[`overlap_frac_${best}`] ?? 1) ? ax : best, 'e1');
  renderResultAxis(axes);
  renderTechSeg(valid);
  loadSplitMap();
}

function countUp(el, to) {
  if (REDUCED_MOTION) { el.textContent = fmtInt(to); return; }
  const t0 = performance.now(), d = 700;
  const step = now => {
    const k = Math.min(1, (now - t0) / d), e = 1 - Math.pow(1 - k, 3);
    el.textContent = fmtInt(Math.round(to * e));
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function techName(t) { return TECH[t] ? `${TECH[t].name} (${t})` : t; }

// 3 significant figures; tiny values in exponent form so they never print as 0.000.
function sig3(v) {
  if (v == null || Number.isNaN(+v)) return '—';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a < 1e-2) return (+v).toExponential(1);
  return (+(+v).toPrecision(3)).toLocaleString();
}

// Per entity axis: is NN separation relative to a random split (×) or a raw distance?
function sepIsRelative(cmp, ax) {
  const rows = cmp.techniques || {};
  return Object.keys(rows).some(t => rows[t] && rows[t][`nn_rel_${ax}`] != null);
}

// Best valid non-random method for unseen entities on one axis:
// lowest overlap, then largest NN separation.
function bestForAxis(cmp, valid, ax) {
  const rows = cmp.techniques || {};
  const sep = r => r[`nn_rel_${ax}`] ?? r[`nn_sep_${ax}`] ?? -Infinity;
  return valid.filter(t => t !== 'R' && rows[t]).sort((a, b) =>
    (rows[a][`overlap_frac_${ax}`] ?? 1) - (rows[b][`overlap_frac_${ax}`] ?? 1) ||
    sep(rows[b]) - sep(rows[a]) || a.localeCompare(b))[0];
}

const REC_ICON = {
  ok: '<svg viewBox="0 0 24 24" fill="none" stroke-width="2.6" stroke-linecap="round"><path d="M20 6 9 17l-5-5"/></svg>',
  warn: '<svg viewBox="0 0 24 24" fill="none" stroke-width="2.6" stroke-linecap="round"><path d="M12 7v6M12 17h.01"/></svg>',
};

function renderRecommendation(cmp, valid, axes) {
  const rec = cmp.recommendation;
  show('rec', !!rec);
  if (!rec) return;
  $('rec-title').textContent = `Recommended: ${techName(rec)}`;
  const box = $('rec');
  box.classList.remove('warn');
  box.querySelector('.rec-icon').innerHTML = REC_ICON.ok;
  if (axes.length < 2) {
    $('rec-sub').textContent = 'Pareto-optimal and the best similarity separation between train and test among the valid splits.';
    return;
  }
  // 2D: a split can be cold on one axis and fully shared on the other — say so.
  const r = (cmp.techniques || {})[rec] || {};
  const name = ax => S.run[ax].name;
  const shared = axes.filter(ax => (r[`overlap_frac_${ax}`] ?? 0) >= 0.5);
  const best = axes.map(ax => [ax, bestForAxis(cmp, valid, ax)]).filter(([, t]) => t);
  const goals = best.map(([ax, t]) => `unseen ${name(ax)}s in test → ${techName(t)}`).join(' · ');
  if (shared.length) {
    box.classList.add('warn');
    box.querySelector('.rec-icon').innerHTML = REC_ICON.warn;
    const parts = shared.map(ax => `${Math.round(r[`overlap_frac_${ax}`] * 100)}% of ${name(ax)}s appear in more than one split`);
    $('rec-sub').textContent = `Pareto-optimal with the best overall NN separation, but ${parts.join(' and ')} — ` +
      `it is not a cold split on ${shared.map(name).join(' or ')}. ` + (goals ? `By goal: ${goals}.` : '');
  } else {
    $('rec-sub').textContent = 'Pareto-optimal and the best similarity separation between train and test among the valid splits.' +
      (goals ? ` By goal: ${goals}.` : '');
  }
}

function renderComparison(cmp, axes, timings) {
  const techs = Object.keys(cmp.status || {});
  const rows = cmp.techniques || {};
  const name = ax => esc(S.run[ax].name);
  const rel = Object.fromEntries(axes.map(ax => [ax, sepIsRelative(cmp, ax)]));
  // In 1D every entity is one row, so overlap is 0 by construction: only show it in 2D.
  const showOverlap = axes.length > 1;
  const hasTime = techs.some(t => timings[t] != null);
  let head = '<thead><tr><th>Method</th><th>Size deviation</th>';
  axes.forEach(ax => {
    head += (showOverlap ? `<th>Overlap · ${name(ax)}</th>` : '') +
      `<th title="${rel[ax] ? 'Mean NN distance relative to a random split of the same rows' : 'Mean NN distance (raw; no random baseline on this axis)'}">` +
      `NN separation · ${name(ax)}<span class="th-unit">${rel[ax] ? '× random' : 'distance'}</span></th>`;
  });
  axes.forEach(ax => { head += `<th>Shift · ${name(ax)}</th>`; });
  // Status only matters when something failed (failed rows are listed below too).
  const anyFailed = techs.some(t => cmp.status[t] !== 'valid');
  head += (hasTime ? '<th>Time</th>' : '') + '<th>Pareto</th>' + (anyFailed ? '<th>Status</th>' : '') + '</tr></thead>';
  // Valid first; then the recommendation, other Pareto-optimal methods, the rest.
  const rank = t => (cmp.status[t] === 'valid' ? 0 : 3) +
    (t === cmp.recommendation ? 0 : rows[t] && rows[t].pareto_optimal ? 1 : 2);
  const order = techs.sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  const pct = v => (v == null ? '—' : (v * 100).toFixed(1) + '%');
  const secs = v => (v == null ? '—' : v < 60 ? `${v.toFixed(v < 10 ? 1 : 0)} s` : dur(v));
  const body = order.map((t, i) => {
    const r = rows[t] || {}, ok = cmp.status[t] === 'valid';
    let tr = `<tr style="--i:${i}" class="${t === cmp.recommendation ? 'recommended' : ''} ${ok ? '' : 'failed'}">` +
      `<td><span class="tname">${esc(TECH[t] ? TECH[t].name : t)} <span class="tag">${esc(t)}</span></span>` +
      `<span class="tfam">${esc(TECH[t] ? TECH[t].family : '')}</span></td><td>${ok ? pct(r.size_dev) : '—'}</td>`;
    axes.forEach(ax => {
      const sep = rel[ax] ? r[`nn_rel_${ax}`] : r[`nn_sep_${ax}`];
      const txt = sep == null ? '—' : sig3(sep) + (rel[ax] ? '×' : '');
      tr += (showOverlap ? `<td>${ok ? pct(r[`overlap_frac_${ax}`]) : '—'}</td>` : '') + `<td>${ok ? txt : '—'}</td>`;
    });
    axes.forEach(ax => { tr += `<td>${ok ? sig3(r[`dist_shift_${ax}`]) : '—'}</td>`; });
    tr += (hasTime ? `<td>${secs(timings[t])}</td>` : '') +
      `<td>${r.pareto_optimal ? '<span class="check" title="Pareto-optimal">✓</span>' : ''}</td>` +
      (anyFailed ? `<td>${ok ? '<span class="status-ok">valid</span>' : '<span class="status-bad">failed</span>'}</td>` : '') + '</tr>';
    return tr;
  }).join('');
  $('cmp-table').innerHTML = head + '<tbody>' + body + '</tbody>';
  const anyRel = axes.some(ax => rel[ax]), anyRaw = axes.some(ax => !rel[ax]);
  $('metric-note').innerHTML =
    '<b>Size deviation</b>: largest gap between a split’s share of rows and its target (lower is better). ' +
    (showOverlap ? '<b>Overlap</b>: share of entities that appear in more than one split (lower is better). ' : '') +
    '<b>NN separation</b>: mean distance from each held-out entity to its nearest training entity (higher means less leakage)' +
    (anyRel ? '; <b>× random</b> = relative to a random split of the same rows (1× = random)' : '') +
    (anyRaw ? `; <b>distance</b> = raw value, comparable only within a column${showOverlap ? ' (a random row split shares every entity, so there is no random baseline)' : ''}` : '') +
    '. <b>Shift</b>: normalized feature-distribution shift between splits (reported, not ranked).' +
    (hasTime ? ' <b>Time</b>: splitting wall-clock time on this server.' : '');
  const failed = cmp.failed || {};
  $('failed-list').innerHTML = Object.entries(failed).map(([t, reasons]) =>
    `<div class="failed-item"><b>${esc(t)}</b> — ${esc([].concat(reasons).join('; '))}</div>`).join('');
}

function renderDownloads(files, ds, valid) {
  const byName = Object.fromEntries((files.techniques || []).map(t => [t.name, t]));
  $('downloads').innerHTML = valid.map(t => {
    const folder = byName[`${t}_${ds}`];
    const list = folder ? folder.files.map(f => f.split('/').pop()) : [];
    return `<div class="dl-row"><div><div class="name">${esc(techName(t))}</div>
      <div class="files">${esc(list.slice(0, 4).join(' · '))}${list.length > 4 ? ` · +${list.length - 4} more` : ''}</div></div>
      ${folder ? `<a class="btn btn-secondary btn-sm" href="/api/runs/${S.runId}/download/${encodeURIComponent(folder.name)}?client_id=${encodeURIComponent(CLIENT_ID)}">Download</a>` : ''}</div>`;
  }).join('') || '<div class="hint">No valid splits to download.</div>';
}

function renderResultAxis(axes) {
  const seg = $('result-axis');
  show(seg, axes.length > 1);
  seg.innerHTML = axes.map(ax => `<button data-ax="${ax}" class="${ax === S.resultAxis ? 'on' : ''}">${esc(S.run[ax].name)}</button>`).join('');
  seg.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    S.resultAxis = b.dataset.ax; renderResultAxis(axes); loadSplitMap();
  }));
}

function renderTechSeg(valid) {
  const seg = $('tech-seg');
  seg.innerHTML = valid.map(t => `<button data-t="${esc(t)}" class="${t === S.resultTech ? 'on' : ''}">${esc(TECH[t] ? TECH[t].name : t)}</button>`).join('');
  seg.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    S.resultTech = b.dataset.t;
    seg.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    renderResultMap();
  }));
}

async function loadSplitMap() {
  const ax = S.resultAxis, id = S.runId;
  show('result-empty');
  $('result-msg').innerHTML = '<span class="spinner"></span>&nbsp; Preparing map…';
  try {
    let r = await api(`/api/runs/${id}/splitmap/${ax}`);
    if (r.status === 'missing') {
      await fetchMap(r.dataset_id, r.map_request, msg => { $('result-msg').innerHTML = `<span class="spinner"></span>&nbsp; ${esc(msg)}`; });
      r = await api(`/api/runs/${id}/splitmap/${ax}`);
    }
    if (S.runId !== id) return;
    S.splitLabels[ax] = {key: r.key, labels: r.labels};
    if (!S.maps[r.key]) S.maps[r.key] = (await api(`/api/map/${r.dataset_id}/${r.key}`)).map;
    if (ax === S.resultAxis) renderResultMap();
  } catch (err) { $('result-msg').textContent = 'Could not build the split map: ' + err.message; }
}

function renderResultMap() {
  const sl = S.splitLabels[S.resultAxis];
  if (!sl || !S.resultTech) return;
  const m = S.maps[sl.key], labels = sl.labels[S.resultTech];
  if (!m || !labels) { show('result-empty'); $('result-msg').textContent = 'No assignment available for this method.'; return; }
  const names = S.run.splitting.ratio.split('/').length === 3 ? ['train', 'val', 'test'] : ['train', 'test'];
  const counts = {};
  labels.forEach(l => { const k = l ?? 'unassigned'; counts[k] = (counts[k] || 0) + 1; });
  const hover = hoverText(m).map((h, i) => `${h}<br>split: ${esc(labels[i] ?? 'unassigned')}`);
  const color = n => (SPLIT_SLOT[n] ? css(`--series-${SPLIT_SLOT[n]}`) : css('--other'));
  const groups = [...names, 'multiple', 'unassigned'].filter(n => counts[n]);
  // draw the training set first so the smaller held-out sets sit on top
  const traces = groups.map(n => trace(m, idx(m).filter(i => (labels[i] ?? 'unassigned') === n), color(n), n, hover));
  plotScatter('result-plot', traces, `${sl.key}:${S.resultAxis}`);
  $('result-legend').innerHTML = legendHtml(groups.map(n => [color(n), n, counts[n]]));
  $('result-meta').textContent = `${techName(S.resultTech)} · ${m.embedding} → t-SNE · ${fmtInt(m.n_shown)} ${S.run[S.resultAxis].name} entities`;
  show('result-meta-2d', !!S.run.e2);
  renderMethodPanel(groups.map(n => [n, counts[n]]), m);
}

// Side panel: what the selected method is and how its split scores.
function renderMethodPanel(counts, m) {
  const t = S.resultTech, info = TECH[t] || {}, cmp = S.comparison || {};
  const r = (cmp.techniques || {})[t] || {};
  const axes = S.run.e2 ? ['e1', 'e2'] : ['e1'];
  const pct = v => (v == null ? '—' : (v * 100).toFixed(1) + '%');
  const ent = S.run[S.resultAxis].name;
  let rows = `<div class="mp-sec">${esc(ent)}s on this map${m.n_shown < m.n_entities ? ' (sample)' : ''}</div>` +
    counts.map(([n, c]) => `<dt>${esc(n)}</dt><dd>${fmtInt(c)}</dd>`).join('');
  rows += '<div class="mp-sec">Split quality</div>' +
    `<dt>Size deviation</dt><dd>${pct(r.size_dev)}</dd>`;
  axes.forEach(ax => {
    const name = esc(S.run[ax].name), rel = sepIsRelative(cmp, ax);
    const sep = rel ? r[`nn_rel_${ax}`] : r[`nn_sep_${ax}`];
    if (axes.length > 1) rows += `<dt>Overlap · ${name}</dt><dd>${pct(r[`overlap_frac_${ax}`])}</dd>`;
    rows += `<dt>NN separation · ${name}</dt><dd>${sep == null ? '—' : sig3(sep) + (rel ? '×' : '')}</dd>`;
  });
  if (S.timings && S.timings[t] != null) rows += `<dt>Time</dt><dd>${S.timings[t].toFixed(1)} s</dd>`;
  rows += `<dt>Pareto-optimal</dt><dd>${r.pareto_optimal ? 'yes' : 'no'}</dd>`;
  $('method-panel').innerHTML = `<h4>${esc(info.name || t)} <span class="tag">${esc(t)}</span></h4>` +
    `<div class="mp-fam">${esc(info.family || '')}${t === cmp.recommendation ? ' · recommended' : ''}</div>` +
    (info.desc ? `<p class="mp-desc">${esc(info.desc)}</p>` : '') + `<dl>${rows}</dl>`;
}

// ── Boot ────────────────────────────────────────────────────────────────────

function initNav() {
  document.querySelectorAll('.step').forEach(b => b.addEventListener('click', () => {
    const k = +b.dataset.step;
    if (k <= 2 && k <= maxStep) goStep(k);
    else if (k === 3 && S.runId) goStep(3);
    else if (k === 4 && S.runId && S.comparison) goStep(4);
  }));
  document.querySelectorAll('[data-goto]').forEach(b => b.addEventListener('click', () => goStep(+b.dataset.goto)));
  $('to-results').addEventListener('click', () => {
    if (S.comparison) goStep(4); else loadResults().then(() => goStep(4));
  });
  $('dl-all').addEventListener('click', () => { window.location.href = `/api/runs/${S.runId}/download?client_id=${encodeURIComponent(CLIENT_ID)}`; });
  $('restart').addEventListener('click', () => {
    S.jobId = null; S.demo = null; maxStep = 1;
    show('dataset-card', false); $('file-input').value = ''; $('folder-input').value = '';
    resetExplore(); goStep(1);
  });
  $('map-btn').addEventListener('click', computeMap);
}

window.addEventListener('DOMContentLoaded', async () => {
  initTheme(); initNav(); initUpload(); initConfig(); initRunsPanel(); setMode('1d');
  loadDemos();
  refreshRuns();
  try {
    S.info = await api('/api/info');
    $('retention').textContent = `Uploaded data and runs are deleted after ${S.info.job_max_age_hours} h without activity.`;
  } catch (_) { /* optional */ }
});

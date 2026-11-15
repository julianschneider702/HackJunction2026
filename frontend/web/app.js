'use strict';
// Read-only dashboard for the inspection cell. Polls GET /api/state (docs/CONTRACT.md); no buttons, no setup.

const $ = (id) => document.getElementById(id);
const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ESCAPES[c]);
const has = (v) => v != null && Number.isFinite(Number(v));
const fmt = (v, digits = 1) => (has(v) ? Number(v).toFixed(digits) : '–');
const clock = (ts) => { const d = new Date(ts); return Number.isNaN(d.getTime()) ? '–' : d.toLocaleTimeString('en-GB'); };

const STEPS = [
  { phase: 'perceive', title: 'Perceive', sub: 'Cameras' },
  { phase: 'detect', title: 'Detect', sub: 'Anomaly model' },
  { phase: 'reason', title: 'Reason', sub: 'Decision' },
  { phase: 'execute', title: 'Execute', sub: 'SO-101 arm' },
];
const DEVICES = [
  { id: 'CPU', name: 'CPU', util: 'cpu' },
  { id: 'GPU', name: 'iGPU', util: 'gpu' },
  { id: 'NPU', name: 'NPU', util: 'npu' },
];
const RECENT_ROWS = 8;
const CHART_CYCLES = 30;

let S = null;
let seq = null;
let online = null;
let clockOffset = 0;
let newestSeen = null;
const memo = {};

// ---------------------------------------------------------------- polling

async function refresh() {
  try {
    const response = await fetch('/api/state' + (S && seq != null ? `?since=${seq}` : ''), { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    online = true;
    if (!data.unchanged) {
      S = data;
      seq = data.seq;
      clockOffset = Date.now() - Date.parse(data.server_time);
    }
  } catch (err) {
    online = false;
  }
  render();
}

async function loop() {
  await refresh();
  setTimeout(loop, online ? 500 : 2000);
}

// ---------------------------------------------------------------- rendering

function panel(key, deps, draw) {
  const signature = JSON.stringify(deps);
  if (memo[key] === signature) return;
  memo[key] = signature;
  draw();
}

function render() {
  renderStatus();
  if (!S) return;
  panel('steps', [S.phase.phase], renderSteps);
  renderPhaseNow();
  panel('kpis', S.stats, renderKpis);
  panel('charts', [S.cycles.slice(0, CHART_CYCLES), S.stats.total], renderCharts);
  panel('hw', [S.telemetry.latest, S.models, S.inference.policy.mean_ms, S.inference.anomaly.mean_ms], renderHardware);
  panel('recent', [S.cycles.slice(0, RECENT_ROWS), S.stats.total], renderRecent);
  panel('bench', S.benchmarks, renderBench);
}

function renderStatus() {
  const robot = S ? S.robot.state : null;
  const paused = Boolean(S && S.control.stop_requested) || robot === 'stopped';
  let cls = '';
  let text = 'Connecting';
  if (online === false) { cls = 'bad'; text = 'Offline · reconnecting'; }
  else if (S && robot === 'error') { cls = 'bad'; text = 'Needs attention'; }
  else if (S && paused) { cls = 'warn'; text = 'Paused'; }
  else if (S && robot === 'running') { cls = 'live'; text = 'Sorting'; }
  else if (S) { text = 'Idle'; }
  $('state').className = 'state ' + cls;
  $('state-text').textContent = text;
  $('demo').hidden = !(S && S.demo);
}

function renderSteps() {
  const active = STEPS.findIndex((s) => s.phase === S.phase.phase);
  $('steps').innerHTML = STEPS.map((s, i) => `
    <li class="${i === active ? 'active' : ''}"${i === active ? ' aria-current="step"' : ''}>
      <b>${i + 1} ${esc(s.title)}</b><span>${esc(s.sub)}</span>
    </li>`).join('');
}

function renderPhaseNow() {
  const p = S.phase;
  let html;
  if (p.phase === 'idle') {
    html = `Waiting for the next brick<small>&nbsp;</small>`;
  } else {
    const seconds = Math.max(0, (Date.now() - clockOffset - Date.parse(p.since)) / 1000);
    const elapsed = Number.isFinite(seconds) ? `${seconds < 100 ? seconds.toFixed(1) : Math.round(seconds)} s` : '–';
    const title = `${p.phase[0].toUpperCase()}${p.phase.slice(1)}`;
    html = `${esc(title)} · <b>${elapsed}</b><small>${esc(p.detail || '') || '&nbsp;'}</small>`;
  }
  if (memo.phaseNow !== html) {
    memo.phaseNow = html;
    $('phase-now').innerHTML = html;
  }
}

function kpi(label, value, unit, sub, swatch) {
  return `<div class="kpi">
    <div class="kpi-label">${swatch ? `<i style="background:var(${swatch})"></i>` : ''}${esc(label)}</div>
    <div class="kpi-value">${value}${unit ? `<small>${esc(unit)}</small>` : ''}</div>
    <div class="kpi-sub">${sub || '&nbsp;'}</div>
  </div>`;
}

function renderKpis() {
  const s = S.stats;
  const tiles = [
    kpi('Bricks sorted', s.total, '', s.lost ? `${s.lost} lost on the way` : ''),
    kpi('Good', s.good, '', '', '--good'),
    kpi('Defective', s.defective, '', '', '--bad'),
    kpi('Cycle time', fmt(s.avg_cycle_s_last10), has(s.avg_cycle_s_last10) ? 's' : '', 'average of the last 10'),
    kpi('Throughput', fmt(s.bricks_per_min), has(s.bricks_per_min) ? '/min' : '', ''),
  ];
  if (has(s.accuracy)) {
    tiles.push(kpi('Accuracy', Math.round(s.accuracy * 100), '%', `${s.correct} of ${s.judged} checked`));
  }
  $('kpis').innerHTML = tiles.join('');
}

// ---- charts (plain SVG)

function niceTicks(lo, hi, count = 4) {
  const raw = (hi - lo || 1) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(Number(v.toFixed(8)));
  return out;
}

function chartFrame(id, emptyText) {
  const el = $(id);
  const width = el.clientWidth;
  const height = el.clientHeight;
  if (emptyText || width < 40) {
    el.innerHTML = `<div class="empty">${esc(emptyText || '')}</div>`;
    return null;
  }
  return { el, width, height, left: 36, right: width - 8, top: 10, bottom: height - 22 };
}

function yAxis(f, lo, hi, digits) {
  const y = (v) => f.bottom - ((v - lo) / (hi - lo)) * (f.bottom - f.top);
  const parts = niceTicks(lo, hi).map((v) => `
    <line class="gridline" x1="${f.left}" x2="${f.right}" y1="${y(v)}" y2="${y(v)}"/>
    <text x="${f.left - 6}" y="${y(v) + 4}" text-anchor="end">${Number(v.toFixed(digits))}</text>`);
  parts.push(`<line class="axis" x1="${f.left}" x2="${f.right}" y1="${f.bottom}" y2="${f.bottom}"/>`);
  return { y, svg: parts.join('') };
}

function refLine(f, y, label) {
  return `<line class="ref" x1="${f.left}" x2="${f.right}" y1="${y}" y2="${y}"/>
    <text class="ref-label" x="${f.right}" y="${y - 4}" text-anchor="end">${esc(label)}</text>`;
}

function ordinalLabels(f, count, firstNo, cx) {
  const every = Math.max(1, Math.ceil(count / 8));
  let out = '';
  for (let i = 0; i < count; i += 1) {
    if ((count - 1 - i) % every === 0) out += `<text x="${cx(i)}" y="${f.height - 5}" text-anchor="middle">#${firstNo + i}</text>`;
  }
  return out;
}

function renderCharts() {
  const recent = S.cycles.slice(0, CHART_CYCLES).reverse(); // oldest left
  const firstNo = S.stats.total - recent.length + 1;

  let f = chartFrame('chart-cycles', recent.length ? null : 'Waiting for the first brick');
  if (f) {
    const hi = Math.max(...recent.map((c) => c.duration_s), 1) * 1.12;
    const axis = yAxis(f, 0, hi, 0);
    const band = (f.right - f.left) / Math.max(recent.length, 8);
    const w = Math.max(3, Math.min(26, band - 3));
    const cx = (i) => f.left + band * (i + 0.5);
    const bars = recent.map((c, i) => {
      const top = axis.y(c.duration_s);
      const r = Math.min(4, w / 2, f.bottom - top);
      const x0 = cx(i) - w / 2;
      return `<path class="bar" d="M${x0},${f.bottom}V${top + r}Q${x0},${top} ${x0 + r},${top}H${x0 + w - r}Q${x0 + w},${top} ${x0 + w},${top + r}V${f.bottom}Z"/>`;
    });
    const avg = S.stats.avg_cycle_s_last10;
    f.el.innerHTML = `<svg width="${f.width}" height="${f.height}" role="img" aria-label="Cycle time per brick">
      ${axis.svg}${bars.join('')}${has(avg) ? refLine(f, axis.y(avg), `avg ${fmt(avg)} s`) : ''}
      ${ordinalLabels(f, recent.length, firstNo, cx)}</svg>`;
  }

  const scored = recent.some((c) => has(c.anomaly_score));
  $('card-scores').hidden = !scored && recent.length > 0;
  f = chartFrame('chart-scores', scored ? null : 'Waiting for the first score');
  if (f) {
    const thresholds = recent.map((c) => c.threshold).filter(has);
    const threshold = thresholds.length ? thresholds[thresholds.length - 1] : null;
    const scores = recent.map((c) => c.anomaly_score).filter(has);
    const lo = Math.min(0, ...scores);
    const hi = Math.max(1, (threshold || 0) * 1.2, ...scores) * 1.06;
    const axis = yAxis(f, lo, hi, 1);
    const band = (f.right - f.left) / Math.max(recent.length, 8);
    const cx = (i) => f.left + band * (i + 0.5);
    const marks = recent.map((c, i) => {
      if (!has(c.anomaly_score)) return '';
      const x = cx(i);
      const y = axis.y(c.anomaly_score);
      return c.decision === 'defective'
        ? `<rect class="bad" x="${x - 4.5}" y="${y - 4.5}" width="9" height="9" transform="rotate(45 ${x} ${y})"/>`
        : `<circle class="good" cx="${x}" cy="${y}" r="5"/>`;
    });
    f.el.innerHTML = `<svg width="${f.width}" height="${f.height}" role="img" aria-label="Anomaly score per brick">
      ${axis.svg}${has(threshold) ? refLine(f, axis.y(threshold), `threshold ${fmt(threshold, 2)}`) : ''}
      ${marks.join('')}${ordinalLabels(f, recent.length, firstNo, cx)}</svg>`;
  }
}

// ---- hardware

function renderHardware() {
  const t = S.telemetry.latest || {};
  const util = t.util || {};
  $('devices').innerHTML = DEVICES.map((d) => {
    const value = util[d.util];
    const models = S.models.filter((m) => m.device === d.id);
    return `<div class="device">
      <div class="device-name"><i class="dev-${d.id}"></i>${d.name}</div>
      <div class="device-util">${has(value) ? `${fmt(value, 0)}<small>%</small>` : '<small>–</small>'}</div>
      <div class="meter"><i class="dev-${d.id}" style="width:${has(value) ? Math.max(0, Math.min(100, value)) : 0}%"></i></div>
      ${models.map((m) => `<div class="device-model"><b>${esc(m.name)}</b><br>${esc(m.role === 'policy' ? 'VLA policy' : 'Anomaly model')}${m.precision ? ' · ' + esc(m.precision) : ''}</div>`).join('')}
    </div>`;
  }).join('');

  const items = [];
  const add = (label, value, unit, digits = 1) => {
    if (has(value)) items.push(`<div><span>${esc(label)}</span><b>${fmt(value, digits)}<small>${esc(unit)}</small></b></div>`);
  };
  add('Policy step', S.inference.policy.mean_ms, 'ms');
  add('Anomaly inference', S.inference.anomaly.mean_ms, 'ms');
  add('Power', t.power_w, 'W');
  add('Temperature', t.temp_c, '°C', 0);
  $('system').innerHTML = items.join('');
}

// ---- recent bricks

function decisionHtml(value) {
  if (value === 'good') return '<span class="is-good">Good</span>';
  if (value === 'defective') return '<span class="is-bad">Defective</span>';
  return '–';
}

function renderRecent() {
  const rows = S.cycles.slice(0, RECENT_ROWS);
  if (!rows.length) {
    $('recent').innerHTML = '<div class="empty">Waiting for the first brick</div>';
    return;
  }
  const fresh = newestSeen !== null && rows[0].id !== newestSeen;
  newestSeen = rows[0].id;
  const body = rows.map((c, i) => `<tr class="${fresh && i === 0 ? 'fresh' : ''}">
      <td class="num">#${S.stats.total - i}</td>
      <td>${clock(c.finished_at)}</td>
      <td>${decisionHtml(c.decision)}</td>
      <td class="num">${has(c.anomaly_score) ? fmt(c.anomaly_score, 2) : '–'}</td>
      <td class="num">${fmt(c.duration_s)} s</td>
    </tr>`).join('');
  $('recent').innerHTML = `<table>
    <thead><tr><th class="num">Brick</th><th>Time</th><th>Decision</th><th class="num">Score</th><th class="num">Cycle</th></tr></thead>
    <tbody>${body}</tbody></table>`;
}

// ---- device comparison (only shown when benchmark results exist)

function renderBench() {
  const rows = S.benchmarks;
  $('card-bench').hidden = !rows.length;
  if (!rows.length) return;
  const slowest = {};
  rows.forEach((r) => { slowest[r.model] = Math.max(slowest[r.model] || 0, r.latency_ms); });
  let previous = null;
  const body = rows.map((r) => {
    const first = r.model !== previous;
    previous = r.model;
    const share = slowest[r.model] ? (r.latency_ms / slowest[r.model]) * 100 : 0;
    const dev = DEVICES.find((d) => d.id === r.device);
    return `<tr>
      <td><b>${first ? esc(r.model) : ''}</b></td>
      <td><span class="tag"><i class="dev-${esc(r.device)}"></i>${esc(dev ? dev.name : r.device)}</span></td>
      <td>${esc(r.precision)}</td>
      <td class="num">${fmt(r.latency_ms)} ms</td>
      <td style="width:35%"><div class="inline-bar dev-${esc(r.device)}" style="width:${share.toFixed(1)}%"></div></td>
      <td class="num">${has(r.fps) ? `${fmt(r.fps)} fps` : ''}</td>
    </tr>`;
  }).join('');
  $('bench').innerHTML = `<table><tbody>${body}</tbody></table>`;
}

// ---------------------------------------------------------------- start

let resizeTimer = null;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { delete memo.charts; render(); }, 150);
});

setInterval(() => { if (S && online) renderPhaseNow(); }, 200);
loop();

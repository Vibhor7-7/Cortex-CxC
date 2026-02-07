import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PointerLockControls } from "three/addons/controls/PointerLockControls.js";
import { fetchChats, searchChats, fetchChatDetails, uploadChatFile, healthCheck, ApiError } from "./api.js";

/**********************************************************************
 * Cortext — split-file Three.js
 * (Logic preserved 1:1 — just moved into this module)
 **********************************************************************/

// ---------- Config ----------
const CFG = {
  N: 100,             // 300–1000
  D: 128,             // vector dim
  CLUSTERS: 8,
  SPACE_SCALE: 34,    // spread in world units
  POINT_SIZE_PX: 2.2, // base px (overridden by UI)
  HOVER_BOOST: 1.3,
  SELECT_BOOST: 1.8,
  EDGE_K: 8,          // top-K neighbors
  EDGE_MIN_SIM: 0.25, // similarity threshold for showing edges
  EDGE_FADE_DIST: 70,
  DRIFT_STRENGTH: 0.40,  // how "alive" motion feels
  SPRING_K: 3.2,         // spring stiffness
  DAMPING: 2.9,          // critical-ish damping factor
  GRAVITY_RADIUS: 16,
  GRAVITY_STRENGTH: 1.8,
  FOCUS_RADIUS: 36,
  WAVE_SPEED: 60,
  WAVE_WIDTH: 14,
  WAVE_STRENGTH: 0.6,
  WAVE_MAX_DIST: 220,
  STAR_COUNT: 2200,
  FOG_NEAR: 35,
  FOG_FAR: 170,
};

// ---------- DOM ----------
const searchEl = document.getElementById("search");
const resultsEl = document.getElementById("results");
const modeBadgeEl = document.getElementById("modeBadge");
const legendModeEl = document.getElementById("legendMode");
const helpEl = document.getElementById("help");

const clusterSel = document.getElementById("clusterFilter");
const pointSizeRange = document.getElementById("pointSize");
const pointSizeLabel = document.getElementById("pointSizeLabel");
const kNNRange = document.getElementById("kNN");
const kNNLabel = document.getElementById("kNNLabel");
const resetViewBtn = document.getElementById("resetView");
const selectedCountEl = document.getElementById("selectedCount");
const genPromptBtn = document.getElementById("genPrompt");
const clearSelectedBtn = document.getElementById("clearSelected");
const promptModal = document.getElementById("promptModal");
const promptText = document.getElementById("promptText");
const copyPromptBtn = document.getElementById("copyPrompt");
const closePromptBtn = document.getElementById("closePrompt");

const pTitle = document.getElementById("pTitle");
const pCluster = document.getElementById("pCluster");
const pTime = document.getElementById("pTime");
const pTags = document.getElementById("pTags");
const pSnippet = document.getElementById("pSnippet");
const pNeighbors = document.getElementById("pNeighbors");

// ---------- Loading / Toast helpers ----------
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingMessage = document.getElementById("loadingMessage");
const toastContainer = document.getElementById("toastContainer");

/** Show the fullscreen loading overlay with a custom message. */
function showLoading(msg = "Loading…") {
  loadingMessage.textContent = msg;
  loadingOverlay.classList.remove("hidden");
  loadingOverlay.style.display = "flex";
}

/** Hide the fullscreen loading overlay. */
function hideLoading() {
  loadingOverlay.classList.add("hidden");
  setTimeout(() => {
    if (loadingOverlay.classList.contains("hidden")) {
      loadingOverlay.style.display = "none";
    }
  }, 350);
}

/**
 * Show a toast notification.
 * @param {string} msg   – Text to show
 * @param {"error"|"info"|"success"} type
 * @param {number} durationMs – Auto-dismiss time (0 = manual)
 */
function showToast(msg, type = "info", durationMs = 4000) {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = msg;
  toastContainer.appendChild(el);
  if (durationMs > 0) {
    setTimeout(() => {
      el.style.opacity = "0";
      el.style.transform = "translateY(12px)";
      el.style.transition = "opacity 280ms, transform 280ms";
      setTimeout(() => el.remove(), 300);
    }, durationMs);
  }
  return el;
}

/** Show a spinner inside the search results dropdown. */
function showSearchLoading() {
  resultsEl.style.display = "block";
  resultsEl.innerHTML = `<div class="resItem loading"><span class="inline-spinner"></span>Searching…</div>`;
}

/** Track whether the API is currently being called (disable interactions). */
let apiLoading = false;

function setApiLoading(on) {
  apiLoading = on;
  searchEl.disabled = on;
}

// ---------- Renderer / Scene ----------
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(new THREE.Color(0x070A12), 1.0);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(new THREE.Color(0x070A12), CFG.FOG_NEAR, CFG.FOG_FAR);

const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 600);
camera.position.set(0, 18, 78);

// Subtle light for edges / glints (points are shader-driven, so this is minimal)
const amb = new THREE.AmbientLight(0x9bb6ff, 0.25);
scene.add(amb);

// ---------- Controls ----------
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.enableDamping = true;
orbit.dampingFactor = 0.07;
orbit.rotateSpeed = 0.55;
orbit.panSpeed = 0.65;
orbit.zoomSpeed = 0.9;
orbit.minDistance = 8;
orbit.maxDistance = 220;

const fly = new PointerLockControls(camera, document.body);
let isFly = false;

const homeCam = {
  pos: camera.position.clone(),
  target: orbit.target.clone()
};

function setModeFly(on) {
  isFly = on;
  if (isFly) {
    orbit.enabled = false;
    modeBadgeEl.textContent = "Fly";
    legendModeEl.textContent = "Fly";
  } else {
    if (document.pointerLockElement) document.exitPointerLock();
    orbit.enabled = true;
    modeBadgeEl.textContent = "Orbit";
    legendModeEl.textContent = "Orbit";
  }
}

// ---------- Utility: seeded RNG ----------
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t ^= t + Math.imul(t ^ t >>> 7, 61 | t);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
const rng = mulberry32(1337);

// ---------- Vectors & semantics ----------
function normalize(vec) {
  let s = 0;
  for (let i=0; i<vec.length; i++) s += vec[i]*vec[i];
  const inv = 1 / Math.sqrt(s + 1e-9);
  for (let i=0; i<vec.length; i++) vec[i] *= inv;
  return vec;
}
function dot(a, b) {
  let s = 0;
  for (let i=0; i<a.length; i++) s += a[i]*b[i];
  return s;
}
// Deterministic "fake query embedding" from text -> 128D unit vector
function hash32(str) {
  // FNV-1a like
  let h = 2166136261 >>> 0;
  for (let i=0; i<str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
function embedText(text, D=CFG.D) {
  const seed = hash32(text.trim().toLowerCase() || "empty");
  const r = mulberry32(seed);
  const v = new Float32Array(D);
  // quasi-gaussian-ish via sum of uniforms
  for (let i=0; i<D; i++) {
    let x = (r()+r()+r()+r()) - 2; // range approx [-2,2]
    v[i] = x;
  }
  return normalize(v);
}

// ---------- Generate clustered memory nodes ----------
const clusters = [];
for (let c=0; c<CFG.CLUSTERS; c++) {
  const center = new Float32Array(CFG.D);
  for (let i=0; i<CFG.D; i++) center[i] = (rng()*2 - 1);
  normalize(center);
  clusters.push(center);
}

const TOPICS = [
  "RAG / hallucinations",
  "Interview prep",
  "Physics homework",
  "Math / calculus",
  "Hackathon ideas",
  "Career / co-op",
  "Audio / Voicemeeter",
  "Random notes"
];

function pickTags(topic) {
  const tagPool = {
    0: ["retrieval", "embeddings", "citations", "eval"],
    1: ["STAR", "stakeholders", "SQL", "Excel"],
    2: ["forces", "doppler", "circuits", "lab"],
    3: ["integrals", "trig", "proofs", "limits"],
    4: ["3D", "three.js", "MCP", "prototype"],
    5: ["Waterloo", "co-op", "resume", "schedule"],
    6: ["audio", "routing", "drivers", "latency"],
    7: ["idea", "todo", "note", "link"]
  };
  const pool = tagPool[topic] || ["note"];
  const a = pool[Math.floor(rng()*pool.length)];
  const b = pool[Math.floor(rng()*pool.length)];
  const c = pool[Math.floor(rng()*pool.length)];
  const set = Array.from(new Set([a,b,c]));
  return set.slice(0, 3);
}

const nodes = new Array(CFG.N);
const vectors = new Array(CFG.N);
const clusterId = new Uint8Array(CFG.N);
const timestamps = new Array(CFG.N);

// Also keep anchor positions (embedding projection -> 3D)
const anchors = new Float32Array(CFG.N * 3);
const pos = new Float32Array(CFG.N * 3);
const vel = new Float32Array(CFG.N * 3);

// Build simple 3D projection per cluster using random basis vectors
const basis = clusters.map(() => {
  const b1 = new Float32Array(CFG.D);
  const b2 = new Float32Array(CFG.D);
  const b3 = new Float32Array(CFG.D);
  for (let i=0; i<CFG.D; i++) {
    b1[i] = (rng()*2 - 1);
    b2[i] = (rng()*2 - 1);
    b3[i] = (rng()*2 - 1);
  }
  normalize(b1); normalize(b2); normalize(b3);
  return [b1,b2,b3];
});

function clamp01(x){ return Math.max(0, Math.min(1, x)); }

// Spread cluster centers in 3D
const clusterCenters3 = new Array(CFG.CLUSTERS).fill(0).map((_, c) => {
  const angle = (c / CFG.CLUSTERS) * Math.PI * 2;
  const ring = CFG.SPACE_SCALE * 0.75;
  const y = (rng()*2-1) * 8;
  return new THREE.Vector3(Math.cos(angle)*ring, y, Math.sin(angle)*ring);
});

// Generate nodes
const now = Date.now();
const dayMs = 24*3600*1000;

for (let i=0; i<CFG.N; i++) {
  const c = Math.floor(rng()*CFG.CLUSTERS);
  clusterId[i] = c;

  const v = new Float32Array(CFG.D);
  // cluster-centered vector with noise
  const center = clusters[c];
  for (let d=0; d<CFG.D; d++) {
    v[d] = center[d] + (rng()*2-1)*0.35;
  }
  normalize(v);
  vectors[i] = v;

  const [b1,b2,b3] = basis[c];
  const x = dot(v, b1);
  const y = dot(v, b2);
  const z = dot(v, b3);

  // anchor position near that cluster's 3D center
  const jitter = new THREE.Vector3(
    (rng()*2-1) * 10,
    (rng()*2-1) * 7,
    (rng()*2-1) * 10
  );
  const base = clusterCenters3[c].clone().add(jitter);
  const anchor = new THREE.Vector3(x, y, z).multiplyScalar(CFG.SPACE_SCALE * 0.55).add(base);

  anchors[i*3+0] = anchor.x;
  anchors[i*3+1] = anchor.y;
  anchors[i*3+2] = anchor.z;

  // start current pos near anchor
  pos[i*3+0] = anchor.x + (rng()*2-1)*0.6;
  pos[i*3+1] = anchor.y + (rng()*2-1)*0.6;
  pos[i*3+2] = anchor.z + (rng()*2-1)*0.6;

  vel[i*3+0] = 0;
  vel[i*3+1] = 0;
  vel[i*3+2] = 0;

  const topic = c % TOPICS.length;
  const title = `${TOPICS[topic]} — Memory #${i}`;
  const tags = pickTags(topic);
  const snippet = `A short snippet about ${TOPICS[topic]} with tags ${tags.join(", ")}.`;
  const full = `Full text: This is a synthetic memory node for "${TOPICS[topic]}". It exists to test search, hover edges, and selection UX.`;

  // Random timestamp within last ~180 days
  const t = new Date(now - Math.floor(rng()*180)*dayMs - Math.floor(rng()*dayMs));
  timestamps[i] = t;

  nodes[i] = {
    id: i,
    title,
    cluster: c,
    tags,
    snippet,
    full,
    time: t
  };
}

// ---------- Build cluster filter UI ----------
function clusterName(c) {
  return `${c} — ${TOPICS[c % TOPICS.length]}`;
}
clusterSel.innerHTML = "";
const optAll = document.createElement("option");
optAll.value = "-1";
optAll.textContent = "All";
clusterSel.appendChild(optAll);
for (let c=0; c<CFG.CLUSTERS; c++) {
  const opt = document.createElement("option");
  opt.value = String(c);
  opt.textContent = clusterName(c);
  clusterSel.appendChild(opt);
}

// ---------- Points: sprite texture (soft circle) ----------
function makeCircleSprite(size=64) {
  const cnv = document.createElement("canvas");
  cnv.width = cnv.height = size;
  const ctx = cnv.getContext("2d");
  const grd = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
  grd.addColorStop(0.0, "rgba(255,255,255,1.0)");
  grd.addColorStop(0.35, "rgba(255,255,255,0.75)");
  grd.addColorStop(1.0, "rgba(255,255,255,0.0)");
  ctx.fillStyle = grd;
  ctx.fillRect(0,0,size,size);
  const tex = new THREE.CanvasTexture(cnv);
  tex.minFilter = THREE.LinearMipMapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  return tex;
}
const spriteTex = makeCircleSprite(64);

// ---------- Point shader ----------
const geom = new THREE.BufferGeometry();
geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
const aColor = new Float32Array(CFG.N * 3);
const aAlpha = new Float32Array(CFG.N);
const aBoost = new Float32Array(CFG.N);
const aBoostTarget = new Float32Array(CFG.N);
const aBoostBase = new Float32Array(CFG.N);
for (let i=0; i<CFG.N; i++) {
  const c = clusterId[i];
  const hue = (c / CFG.CLUSTERS);
  const col = new THREE.Color().setHSL(hue, 0.65, 0.62);
  aColor[i*3+0] = col.r;
  aColor[i*3+1] = col.g;
  aColor[i*3+2] = col.b;
  aAlpha[i] = 0.95;
  aBoost[i] = 1.0;
  aBoostTarget[i] = 1.0;
  aBoostBase[i] = 1.0;
}
geom.setAttribute("aColor", new THREE.BufferAttribute(aColor, 3));
geom.setAttribute("aAlpha", new THREE.BufferAttribute(aAlpha, 1));
geom.setAttribute("aBoost", new THREE.BufferAttribute(aBoost, 1));

const pointMat = new THREE.ShaderMaterial({
  uniforms: {
    uPointSize: { value: CFG.POINT_SIZE_PX },
    uSprite: { value: spriteTex },
    uTime: { value: 0 },
    uFogColor: { value: scene.fog.color },
    uFogNear: { value: scene.fog.near },
    uFogFar: { value: scene.fog.far },
  },
  vertexShader: `
    uniform float uPointSize;
    uniform float uTime;
    attribute vec3 aColor;
    attribute float aAlpha;
    attribute float aBoost;
    varying vec3 vColor;
    varying float vAlpha;
    varying float vDepth;

    void main() {
      vColor = aColor;
      vAlpha = aAlpha;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vDepth = -mv.z;

      float size = uPointSize * aBoost;
      float atten = 300.0 / (vDepth + 30.0);
      gl_PointSize = size * atten;

      gl_Position = projectionMatrix * mv;
    }
  `,
  fragmentShader: `
    uniform sampler2D uSprite;
    uniform vec3 uFogColor;
    uniform float uFogNear;
    uniform float uFogFar;

    varying vec3 vColor;
    varying float vAlpha;
    varying float vDepth;

    void main() {
      vec4 spr = texture2D(uSprite, gl_PointCoord);
      float alpha = spr.a * vAlpha;

      float core = smoothstep(0.0, 0.35, spr.r);
      vec3 col = vColor * (0.75 + core * 0.55);

      col *= (0.95 + spr.r * 0.35);

      float fogFactor = smoothstep(uFogNear, uFogFar, vDepth);
      col = mix(col, uFogColor, fogFactor);
      alpha *= (1.0 - fogFactor*0.65);

      if (alpha < 0.02) discard;
      gl_FragColor = vec4(col, alpha);
    }
  `,
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending
});

const points = new THREE.Points(geom, pointMat);
scene.add(points);

// ---------- Halo layer ----------
const haloGeom = new THREE.BufferGeometry();
const haloPos = new Float32Array(3);
haloGeom.setAttribute("position", new THREE.BufferAttribute(haloPos, 3));
const haloMat = new THREE.ShaderMaterial({
  uniforms: {
    uPointSize: { value: 20.0 },
    uSprite: { value: spriteTex },
    uColor: { value: new THREE.Color(0xb4d7ff) }
  },
  vertexShader: `
    uniform float uPointSize;
    varying float vDepth;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vDepth = -mv.z;
      float atten = 280.0 / (vDepth + 35.0);
      gl_PointSize = uPointSize * atten;
      gl_Position = projectionMatrix * mv;
    }
  `,
  fragmentShader: `
    uniform sampler2D uSprite;
    uniform vec3 uColor;
    void main() {
      vec4 spr = texture2D(uSprite, gl_PointCoord);
      float a = spr.a;
      vec3 col = uColor;
      if (a < 0.02) discard;
      gl_FragColor = vec4(col, a * 0.75);
    }
  `,
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending
});
const halo = new THREE.Points(haloGeom, haloMat);
halo.visible = false;
scene.add(halo);

// ---------- Starfield background ----------
const starsGeom = new THREE.BufferGeometry();
const starPos = new Float32Array(CFG.STAR_COUNT * 3);
const starCol = new Float32Array(CFG.STAR_COUNT * 3);
for (let i=0; i<CFG.STAR_COUNT; i++) {
  const r = 260 + rng()*120;
  const u = rng()*2 - 1;
  const tt = rng()*Math.PI*2;
  const s = Math.sqrt(1 - u*u);
  const x = r * s * Math.cos(tt);
  const y = r * u;
  const z = r * s * Math.sin(tt);
  starPos[i*3+0] = x;
  starPos[i*3+1] = y;
  starPos[i*3+2] = z;

  const c = new THREE.Color().setHSL(0.58 + (rng()*0.06), 0.35, 0.75 + rng()*0.2);
  starCol[i*3+0] = c.r;
  starCol[i*3+1] = c.g;
  starCol[i*3+2] = c.b;
}
starsGeom.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
starsGeom.setAttribute("color", new THREE.BufferAttribute(starCol, 3));

const starsMat = new THREE.PointsMaterial({
  size: 1.2,
  sizeAttenuation: true,
  transparent: true,
  opacity: 0.75,
  vertexColors: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending
});
const stars = new THREE.Points(starsGeom, starsMat);
scene.add(stars);

// ---------- Edges (LineSegments, dynamic) ----------
const edgeGeom = new THREE.BufferGeometry();
let edgePositions = new Float32Array(CFG.EDGE_K * 2 * 3);
let edgeColors = new Float32Array(CFG.EDGE_K * 2 * 3);
edgeGeom.setAttribute("position", new THREE.BufferAttribute(edgePositions, 3));
edgeGeom.setAttribute("color", new THREE.BufferAttribute(edgeColors, 3));
edgeGeom.setDrawRange(0, 0);

const edgeMat = new THREE.LineBasicMaterial({
  vertexColors: true,
  transparent: true,
  opacity: 0.9,
  blending: THREE.AdditiveBlending,
  depthWrite: false
});
const edges = new THREE.LineSegments(edgeGeom, edgeMat);
edges.visible = false;
scene.add(edges);

// ---------- Hover/Select state ----------
let hovered = -1;
let selected = -1;
const selectedSet = new Set();

// Search state
let currentResults = [];
let activeResIndex = 0;

// Shockwave focus
const wave = {
  active: false,
  t: 0,
  origin: new THREE.Vector3()
};

// ---------- Cosine similarity + top-K neighbors ----------
function topKNeighbors(idx, K) {
  const v = vectors[idx];
  const bestSim = new Float32Array(K);
  const bestIdx = new Int32Array(K);
  for (let k=0; k<K; k++) { bestSim[k] = -1e9; bestIdx[k] = -1; }

  for (let j=0; j<CFG.N; j++) {
    if (j === idx) continue;
    const sim = dot(v, vectors[j]);
    let minK = 0;
    for (let k=1; k<K; k++) if (bestSim[k] < bestSim[minK]) minK = k;
    if (sim > bestSim[minK]) {
      bestSim[minK] = sim;
      bestIdx[minK] = j;
    }
  }
  const pairs = [];
  for (let k=0; k<K; k++) pairs.push([bestSim[k], bestIdx[k]]);
  pairs.sort((a,b) => b[0]-a[0]);
  return pairs.filter(p => p[1] >= 0);
}

// ---------- Fast picking (screen-space nearest) ----------
const mouse = new THREE.Vector2(0,0);
let needsPick = false;
let lastPickT = 0;

renderer.domElement.addEventListener("mousemove", (e) => {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
  needsPick = true;
});

renderer.domElement.addEventListener("click", (e) => {
  if (e.target !== renderer.domElement) return;

  if (isFly) {
    if (!document.pointerLockElement) fly.lock();
  } else {
    if (hovered >= 0) setSelected(hovered);
    else setSelected(-1);
  }
});

function pickPoint() {
  const t = performance.now();
  if (t - lastPickT < 45) return;
  lastPickT = t;

  if (isFly && document.pointerLockElement) return;

  const w = renderer.domElement.width;
  const h = renderer.domElement.height;

  const clusterFilter = parseInt(clusterSel.value, 10);
  const maxPx = 14;
  let best = -1;
  let bestD2 = maxPx*maxPx;

  const v = new THREE.Vector3();
  const mx = (mouse.x * 0.5 + 0.5) * w;
  const my = (-(mouse.y) * 0.5 + 0.5) * h;

  for (let i=0; i<CFG.N; i++) {
    if (clusterFilter >= 0 && clusterId[i] !== clusterFilter) continue;

    v.set(pos[i*3+0], pos[i*3+1], pos[i*3+2]);
    v.project(camera);
    if (v.z < -1 || v.z > 1) continue;

    const sx = (v.x * 0.5 + 0.5) * w;
    const sy = (-(v.y) * 0.5 + 0.5) * h;

    const dx = sx - mx;
    const dy = sy - my;
    const d2 = dx*dx + dy*dy;
    if (d2 < bestD2) {
      bestD2 = d2;
      best = i;
    }
  }

  setHovered(best);
}

function setHovered(i) {
  if (hovered === i) return;
  if (hovered >= 0 && hovered !== selected) aBoostTarget[hovered] = 1.0;

  hovered = i;
  if (hovered >= 0 && hovered !== selected) {
    aBoostTarget[hovered] = CFG.HOVER_BOOST;
    showHalo(hovered, false);
  } else {
    if (selected < 0) {
      halo.visible = false;
    } else {
      showHalo(selected, true);
    }
  }
}

function setSelected(i) {
  if (selected === i) return;

  if (selected >= 0) aBoostTarget[selected] = 1.0;
  selected = i;

  if (selected >= 0) {
    aBoostTarget[selected] = CFG.SELECT_BOOST;
    showHalo(selected, true);
    populatePanel(selected);
    focusNode(selected);
    wave.origin.set(pos[selected*3+0], pos[selected*3+1], pos[selected*3+2]);
    wave.t = 0;
    wave.active = true;
  } else {
    halo.visible = false;
    clearPanel();
  }
}

function showHalo(i, strong) {
  haloPos[0] = pos[i*3+0];
  haloPos[1] = pos[i*3+1];
  haloPos[2] = pos[i*3+2];
  haloGeom.attributes.position.needsUpdate = true;

  haloMat.uniforms.uPointSize.value = strong ? 26.0 : 20.0;
  haloMat.uniforms.uColor.value.set(strong ? 0xbfe2ff : 0x9fd0ff);
  halo.visible = true;
}

function buildAllEdges() {
  const K = CFG.EDGE_K;
  const allEdges = [];

  for (let i = 0; i < CFG.N; i++) {
    const neigh = topKNeighbors(i, K);
    for (let k = 0; k < neigh.length; k++) {
      const sim = neigh[k][0];
      const j = neigh[k][1];
      if (sim < CFG.EDGE_MIN_SIM) continue;
      if (i < j) allEdges.push([i, j, sim]);
    }
  }

  const edgeCount = allEdges.length;
  const needed = edgeCount * 2 * 3;
  edgePositions = new Float32Array(needed);
  edgeColors = new Float32Array(needed);

  for (let e = 0; e < edgeCount; e++) {
    const [i, j, sim] = allEdges[e];
    const ax = pos[i*3+0], ay = pos[i*3+1], az = pos[i*3+2];
    const bx = pos[j*3+0], by = pos[j*3+1], bz = pos[j*3+2];

    const dx = ax - bx, dy = ay - by, dz = az - bz;
    const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
    const fade = clamp01(1.0 - dist / CFG.EDGE_FADE_DIST) * clamp01((sim - CFG.EDGE_MIN_SIM) / (1.0 - CFG.EDGE_MIN_SIM));

    const o = e * 2 * 3;
    edgePositions[o+0] = ax; edgePositions[o+1] = ay; edgePositions[o+2] = az;
    edgePositions[o+3] = bx; edgePositions[o+4] = by; edgePositions[o+5] = bz;

    const colI = new THREE.Color(aColor[i*3+0], aColor[i*3+1], aColor[i*3+2]);
    const colJ = new THREE.Color(aColor[j*3+0], aColor[j*3+1], aColor[j*3+2]);
    const aMix = colI.clone().lerp(colJ, 0.5);
    aMix.multiplyScalar((0.4 + 0.6*fade) * 0.75);

    edgeColors[o+0] = aMix.r; edgeColors[o+1] = aMix.g; edgeColors[o+2] = aMix.b;
    edgeColors[o+3] = aMix.r; edgeColors[o+4] = aMix.g; edgeColors[o+5] = aMix.b;
  }

  edgeGeom.setAttribute("position", new THREE.BufferAttribute(edgePositions, 3));
  edgeGeom.setAttribute("color", new THREE.BufferAttribute(edgeColors, 3));
  edgeGeom.setDrawRange(0, edgeCount * 2);
  edges.visible = true;
}

function showEdgesFor(i, strong) {
  // Kept as in your file: edges are always visible from buildAllEdges().
  // Optional future: brighten edges connected to i.
}

// ---------- Panel ----------
function fmtDate(d) {
  const y = d.getFullYear();
  const m = String(d.getMonth()+1).padStart(2,"0");
  const day = String(d.getDate()).padStart(2,"0");
  return `${y}-${m}-${day}`;
}

function populatePanel(i) {
  const n = nodes[i];
  pTitle.textContent = n.title;
  pCluster.textContent = clusterName(n.cluster);
  pTime.textContent = fmtDate(n.time);
  pTags.textContent = n.tags.join(", ");
  pSnippet.textContent = n.full || n.snippet || "";

  pNeighbors.innerHTML = "";
  const neigh = topKNeighbors(i, CFG.EDGE_K);
  for (let k=0; k<neigh.length; k++) {
    const sim = neigh[k][0];
    const j = neigh[k][1];
    if (j < 0) continue;
    const item = document.createElement("div");
    item.className = "nbItem";
    item.innerHTML = `<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:240px">${nodes[j].title}</span>
                      <span class="badge">${sim.toFixed(2)}</span>`;
    item.addEventListener("click", () => setSelected(j));
    pNeighbors.appendChild(item);
  }

  // If we have backend data, fetch full details asynchronously
  if (backendDataLoaded && n.backendId) {
    pSnippet.innerHTML = `<span class="inline-spinner"></span> Loading details…`;
    fetchChatDetails(n.backendId).then((details) => {
      // Show summary + first few messages
      let content = "";
      if (details.summary) content += details.summary + "\n\n";
      if (details.messages && details.messages.length) {
        const preview = details.messages.slice(0, 4);
        for (const msg of preview) {
          const role = msg.role === "user" ? "👤" : "🤖";
          const text = msg.content.length > 200 ? msg.content.slice(0, 200) + "…" : msg.content;
          content += `${role} ${text}\n\n`;
        }
        if (details.messages.length > 4) {
          content += `… and ${details.messages.length - 4} more messages`;
        }
      }
      pSnippet.textContent = content || n.full || n.snippet || "";
    }).catch((err) => {
      console.warn("[panel] Failed to load details:", err);
      pSnippet.textContent = n.full || n.snippet || "(details unavailable)";
    });
  }
}

function clearPanel() {
  pTitle.textContent = "None";
  pCluster.textContent = "—";
  pTime.textContent = "—";
  pTags.textContent = "—";
  pSnippet.textContent = "Click a point to inspect. Hover to preview + show neighbor edges.";
  pNeighbors.innerHTML = "";
}

// ---------- Cinematic focus ----------
let flyTo = null;
function focusNode(i) {
  if (i < 0) return;
  const target = new THREE.Vector3(pos[i*3+0], pos[i*3+1], pos[i*3+2]);

  const camDir = camera.position.clone().sub(orbit.target).normalize();
  const dist = 24;
  const newPos = target.clone().add(camDir.multiplyScalar(dist)).add(new THREE.Vector3(0, 8, 0));

  flyTo = {
    t: 0,
    dur: 0.7,
    fromPos: camera.position.clone(),
    toPos: newPos,
    fromTarget: orbit.target.clone(),
    toTarget: target
  };
}

function easeInOutCubic(x) {
  return x < 0.5 ? 4*x*x*x : 1 - Math.pow(-2*x+2, 3)/2;
}

// ---------- Living motion ----------
function flowField(x, y, z, t) {
  const s = 0.06;
  const tx = x*s, ty = y*s, tz = z*s;
  const aa = Math.sin(tx + t*0.7) + Math.cos(ty*1.2 - t*0.4);
  const bb = Math.sin(ty + t*0.6) + Math.cos(tz*1.1 + t*0.5);
  const cc = Math.sin(tz + t*0.5) + Math.cos(tx*1.3 - t*0.35);

  const vx = (bb - cc);
  const vy = (cc - aa);
  const vz = (aa - bb);
  const v = new THREE.Vector3(vx, vy, vz);
  const len = v.length() + 1e-6;
  v.multiplyScalar(1/len);
  return v;
}

function applyLocalGravity(i, dt) {
  if (i < 0) return;
  const cx = pos[i*3+0], cy = pos[i*3+1], cz = pos[i*3+2];

  const R = CFG.GRAVITY_RADIUS;
  const R2 = R*R;
  for (let j=0; j<CFG.N; j++) {
    if (j === i) continue;
    const px = pos[j*3+0], py = pos[j*3+1], pz = pos[j*3+2];
    const dx = cx - px, dy = cy - py, dz = cz - pz;
    const d2 = dx*dx + dy*dy + dz*dz;
    if (d2 > R2) continue;

    const d = Math.sqrt(d2) + 1e-6;
    const strength = CFG.GRAVITY_STRENGTH * (1.0 - d/R);
    const ax = (dx/d) * strength;
    const ay = (dy/d) * strength;
    const az = (dz/d) * strength;

    vel[j*3+0] += ax * dt;
    vel[j*3+1] += ay * dt;
    vel[j*3+2] += az * dt;
  }
}

// ---------- Search ----------
/** Map from conversation UUID → index in the nodes[] array. */
const nodeIdToIndex = new Map();
function rebuildIdIndex() {
  nodeIdToIndex.clear();
  for (let i = 0; i < nodes.length; i++) {
    if (nodes[i] && nodes[i].id != null) nodeIdToIndex.set(String(nodes[i].id), i);
  }
}

/**
 * Local fallback search using cosine similarity (works when backend is unreachable
 * or data was loaded as fake).
 */
function searchNodesLocal(query) {
  const q = query.trim();
  if (!q) return [];
  const qv = embedText(q, CFG.D);

  const clusterFilter = parseInt(clusterSel.value, 10);
  const scored = [];
  for (let i = 0; i < CFG.N; i++) {
    if (clusterFilter >= 0 && clusterId[i] !== clusterFilter) continue;
    if (!vectors[i]) continue;
    const sim = dot(qv, vectors[i]);
    scored.push([sim, i]);
  }
  scored.sort((a, b) => b[0] - a[0]);
  return scored.slice(0, 30);
}

/**
 * Search via the backend API.  Returns the same [score, index] format as
 * the local fallback so the rest of the UI code stays identical.
 */
async function searchNodesBackend(query) {
  const q = query.trim();
  if (!q) return [];
  const cf = parseInt(clusterSel.value, 10);
  try {
    const res = await searchChats(q, 30, cf >= 0 ? cf : null);
    const list = [];
    for (const r of res.results) {
      const idx = nodeIdToIndex.get(String(r.conversation_id));
      if (idx != null) {
        list.push([r.score, idx]);
      }
    }
    return list;
  } catch (err) {
    console.warn("[search] Backend search failed, falling back to local:", err);
    showToast("Search fell back to local mode", "info", 2500);
    return searchNodesLocal(query);
  }
}

/** Whether we have real backend data loaded (vs fake). */
let backendDataLoaded = false;

/** Unified search: prefer backend when available. */
async function searchNodes(query) {
  if (backendDataLoaded) return searchNodesBackend(query);
  return searchNodesLocal(query);
}

function showResults(list) {
  currentResults = list;
  activeResIndex = 0;

  if (!list.length) {
    resultsEl.style.display = "none";
    resultsEl.innerHTML = "";
    return;
  }
  resultsEl.style.display = "block";
  resultsEl.innerHTML = "";

  for (let r=0; r<list.length; r++) {
    const sim = list[r][0];
    const i = list[r][1];
    const div = document.createElement("div");
    div.className = "resItem" + (r===0 ? " active" : "");
    if (selectedSet.has(i)) div.classList.add("selected");
    div.innerHTML = `
      <div class="resTitle">${nodes[i].title}</div>
      <div class="resCheck">
        <input type="checkbox" ${selectedSet.has(i) ? "checked" : ""} />
        <span>${sim.toFixed(2)}</span>
      </div>
    `;
    div.addEventListener("mouseenter", () => setActiveResult(r));
    div.addEventListener("click", (e) => {
      const isCheckbox = e.target && e.target.tagName === "INPUT";
      if (isCheckbox) {
        toggleSelectedChat(i);
        div.classList.toggle("selected", selectedSet.has(i));
        return;
      }
      setSelected(i);
      focusNode(i);
    });
    resultsEl.appendChild(div);
  }
}

function hideResults() {
  resultsEl.style.display = "none";
  resultsEl.innerHTML = "";
  currentResults = [];
  activeResIndex = 0;
}

function setActiveResult(idx) {
  activeResIndex = Math.max(0, Math.min(idx, currentResults.length-1));
  const children = [...resultsEl.children];
  children.forEach((el, i) => el.classList.toggle("active", i === activeResIndex));
  const el = children[activeResIndex];
  if (el) el.scrollIntoView({ block: "nearest" });
}

function selectActiveResult() {
  if (!currentResults.length) return;
  const i = currentResults[activeResIndex][1];
  setSelected(i);
  focusNode(i);
  searchEl.blur();
}

let searchTimer = null;
let searchAbort = null;  // track in-flight searches

async function triggerSearch() {
  const q = searchEl.value.trim();
  if (!q) { hideResults(); return; }

  // Show inline loading only for backend searches
  if (backendDataLoaded) showSearchLoading();
  setApiLoading(true);

  try {
    const list = await searchNodes(q);
    showResults(list);
  } catch (err) {
    console.error("[search]", err);
    showToast(`Search error: ${err.message}`, "error");
    hideResults();
  } finally {
    setApiLoading(false);
  }
}

searchEl.addEventListener("input", () => {
  clearTimeout(searchTimer);
  // Longer debounce for backend (network latency) vs instant local
  const delay = backendDataLoaded ? 350 : 80;
  searchTimer = setTimeout(() => triggerSearch(), delay);
});

searchEl.addEventListener("focus", () => {
  triggerSearch();
});

document.addEventListener("click", (e) => {
  if (!resultsEl.contains(e.target) && e.target !== searchEl) {
    hideResults();
  }
});

clusterSel.addEventListener("change", () => {
  const cf = parseInt(clusterSel.value, 10);
  if (cf >= 0 && selected >= 0 && clusterId[selected] !== cf) setSelected(-1);
  if (document.activeElement === searchEl) {
    triggerSearch();
  }
});

// ---------- UI sliders ----------
function setPointSize(px) {
  CFG.POINT_SIZE_PX = px;
  pointMat.uniforms.uPointSize.value = px;
  pointSizeLabel.textContent = `${px.toFixed(1)}px`;
}
pointSizeRange.addEventListener("input", () => setPointSize(parseFloat(pointSizeRange.value)));
setPointSize(parseFloat(pointSizeRange.value));

function setK(k) {
  CFG.EDGE_K = k;
  kNNLabel.textContent = `K=${k}`;
  if (hovered >= 0) showEdgesFor(hovered, hovered === selected);
  else if (selected >= 0) showEdgesFor(selected, true);
}
kNNRange.addEventListener("input", () => setK(parseInt(kNNRange.value, 10)));
setK(parseInt(kNNRange.value, 10));
resetViewBtn.addEventListener("click", () => resetCamera());

function updateSelectedUI() {
  const count = selectedSet.size;
  selectedCountEl.textContent = String(count);
  const ok = count >= 2 && count <= 5;
  genPromptBtn.disabled = !ok;
  clearSelectedBtn.disabled = count === 0;
}

function toggleSelectedChat(i) {
  if (selectedSet.has(i)) selectedSet.delete(i);
  else selectedSet.add(i);
  updateSelectedUI();
}

function buildContextPack(ids) {
  const items = ids.map((id) => nodes[id]);
  const insights = items.map((n) => {
    const tags = n.tags.slice(0, 2).join(", ");
    return `${n.title} (${tags})`;
  });
  const bullets = insights.map((s) => `• ${s}`).join("\n");
  return `Relevant past context:\n${bullets}\n\nInstructions:\n- Use the context pack to continue the thread.\n- Avoid repeating raw chat logs; summarize and reference insights.`;
}

function openPromptModal() {
  const ids = Array.from(selectedSet).slice(0, 5);
  promptText.value = buildContextPack(ids);
  promptModal.style.display = "block";
}

function closePromptModal() {
  promptModal.style.display = "none";
}

genPromptBtn.addEventListener("click", () => openPromptModal());
clearSelectedBtn.addEventListener("click", () => {
  selectedSet.clear();
  updateSelectedUI();
  showResults(currentResults);
});

copyPromptBtn.addEventListener("click", async () => {
  const text = promptText.value;
  try {
    await navigator.clipboard.writeText(text);
    copyPromptBtn.textContent = "Copied";
    setTimeout(() => (copyPromptBtn.textContent = "Copy"), 900);
  } catch {
    promptText.select();
  }
});

closePromptBtn.addEventListener("click", () => closePromptModal());
promptModal.addEventListener("click", (e) => {
  if (e.target === promptModal) closePromptModal();
});

// ---------- Keyboard ----------
const keys = new Set();
let shiftDown = false;

function toggleHelp() {
  helpEl.style.display = (helpEl.style.display === "block") ? "none" : "block";
}

document.addEventListener("keydown", (e) => {
  keys.add(e.code);
  shiftDown = e.shiftKey;

  if (e.key === "/" && document.activeElement !== searchEl) {
    e.preventDefault();
    searchEl.focus();
    searchEl.select();
    showResults(searchNodes(searchEl.value));
    return;
  }

  if (e.key === "?") {
    e.preventDefault();
    toggleHelp();
    return;
  }

  if (e.key === "Escape") {
    e.preventDefault();
    if (document.pointerLockElement) document.exitPointerLock();
    setSelected(-1);
    setHovered(-1);
    hideResults();
    helpEl.style.display = "none";
    return;
  }

  if (e.code === "Space" && e.shiftKey) {
    e.preventDefault();
    setModeFly(!isFly);
    return;
  }

  if (e.code === "KeyR") {
    e.preventDefault();
    resetCamera();
    return;
  }
  if (e.code === "KeyF") {
    e.preventDefault();
    if (selected >= 0) focusNode(selected);
    return;
  }

  if (resultsEl.style.display === "block") {
    if (e.key === "ArrowDown") { e.preventDefault(); setActiveResult(activeResIndex + 1); }
    if (e.key === "ArrowUp") { e.preventDefault(); setActiveResult(activeResIndex - 1); }
    if (e.key === "Enter") { e.preventDefault(); selectActiveResult(); }
  }
});

document.addEventListener("keyup", (e) => {
  keys.delete(e.code);
  shiftDown = e.shiftKey;
});

function resetCamera() {
  setModeFly(false);
  flyTo = {
    t: 0,
    dur: 0.7,
    fromPos: camera.position.clone(),
    toPos: homeCam.pos.clone(),
    fromTarget: orbit.target.clone(),
    toTarget: homeCam.target.clone()
  };
}

helpEl.addEventListener("click", (e) => {
  if (e.target === helpEl) helpEl.style.display = "none";
});

// ---------- Movement (Orbit keyboard pan/zoom) ----------
function orbitKeyboard(dt) {
  if (!orbit.enabled) return;

  const panSpeed = 20 * dt;
  const dollySpeed = 26 * dt;

  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  forward.y = 0; forward.normalize();

  const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0,1,0)).normalize();
  const up = new THREE.Vector3(0,1,0);

  const pan = new THREE.Vector3();

  if (keys.has("KeyA")) pan.addScaledVector(right, -panSpeed);
  if (keys.has("KeyD")) pan.addScaledVector(right, panSpeed);
  if (keys.has("KeyQ")) pan.addScaledVector(up, panSpeed);
  if (keys.has("KeyE")) pan.addScaledVector(up, -panSpeed);

  if (pan.lengthSq() > 0) {
    orbit.target.add(pan);
    camera.position.add(pan);
  }

  if (keys.has("KeyW")) camera.position.addScaledVector(forward, dollySpeed);
  if (keys.has("KeyS")) camera.position.addScaledVector(forward, -dollySpeed);
}

// ---------- Fly mode movement ----------
const flyVel = new THREE.Vector3(0,0,0);
function flyKeyboard(dt) {
  if (!isFly) return;

  const base = 18;
  const boost = shiftDown ? 2.4 : 1.0;
  const brake = keys.has("Space") ? 0.25 : 1.0;
  const speed = base * boost;

  const move = new THREE.Vector3();
  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  forward.normalize();
  const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0,1,0)).normalize();
  const up = new THREE.Vector3(0,1,0);

  if (keys.has("KeyW")) move.add(forward);
  if (keys.has("KeyS")) move.addScaledVector(forward, -1);
  if (keys.has("KeyA")) move.addScaledVector(right, -1);
  if (keys.has("KeyD")) move.add(right);
  if (keys.has("KeyE")) move.add(up);
  if (keys.has("KeyQ")) move.addScaledVector(up, -1);

  if (move.lengthSq() > 0) move.normalize();

  const accel = speed * 3.2;
  flyVel.addScaledVector(move, accel * dt);

  flyVel.multiplyScalar(Math.pow(0.08, dt));
  flyVel.multiplyScalar(brake);

  camera.position.addScaledVector(flyVel, dt);

  orbit.target.copy(camera.position).addScaledVector(forward, 12);
}

// ---------- Resize ----------
addEventListener("resize", () => {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
});

// ---------- Time / animation ----------
const clock = new THREE.Clock();

// ---------- Motion update ----------
const tmp = new THREE.Vector3();
const aa = new THREE.Vector3();
const pp = new THREE.Vector3();
const vv = new THREE.Vector3();

function updateNodes(dt, t) {
  if (selected >= 0) applyLocalGravity(selected, dt);

  const drift = CFG.DRIFT_STRENGTH;
  const k = CFG.SPRING_K;
  const damp = CFG.DAMPING;
  const boostEase = 1.0 - Math.exp(-dt * 10.0);
  const waveFront = wave.active ? (wave.t * CFG.WAVE_SPEED) : -1;

  const cf = parseInt(clusterSel.value, 10);
  const selX = (selected >= 0) ? pos[selected*3+0] : 0;
  const selY = (selected >= 0) ? pos[selected*3+1] : 0;
  const selZ = (selected >= 0) ? pos[selected*3+2] : 0;

  for (let i=0; i<CFG.N; i++) {
    if (cf >= 0) {
      const match = (clusterId[i] === cf);
      aAlpha[i] = match ? 0.95 : 0.06;
    } else {
      aAlpha[i] = 0.95;
    }

    aa.set(anchors[i*3+0], anchors[i*3+1], anchors[i*3+2]);
    pp.set(pos[i*3+0], pos[i*3+1], pos[i*3+2]);
    vv.set(vel[i*3+0], vel[i*3+1], vel[i*3+2]);

    const ff = flowField(pp.x, pp.y, pp.z, t);
    const driftScale = drift * (0.65 + 0.7*((i*97)%100)/100);

    tmp.copy(pp).sub(aa).multiplyScalar(-k);
    tmp.addScaledVector(vv, -damp);
    tmp.addScaledVector(ff, driftScale);

    vv.addScaledVector(tmp, dt);
    pp.addScaledVector(vv, dt);

    aBoostBase[i] += (aBoostTarget[i] - aBoostBase[i]) * boostEase;

    if (selected >= 0 && i !== selected) {
      const dx = pp.x - selX;
      const dy = pp.y - selY;
      const dz = pp.z - selZ;
      const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
      const focus = 1.0 - clamp01((d - 6) / CFG.FOCUS_RADIUS);
      aAlpha[i] *= (0.28 + 0.72 * focus);
    }

    let waveBoost = 0;
    if (wave.active) {
      const dx = pp.x - wave.origin.x;
      const dy = pp.y - wave.origin.y;
      const dz = pp.z - wave.origin.z;
      const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
      const diff = Math.abs(d - waveFront);
      if (diff < CFG.WAVE_WIDTH) {
        waveBoost = (1.0 - diff / CFG.WAVE_WIDTH) * CFG.WAVE_STRENGTH;
      }
    }
    aBoost[i] = aBoostBase[i] + waveBoost;

    pos[i*3+0] = pp.x;
    pos[i*3+1] = pp.y;
    pos[i*3+2] = pp.z;
    vel[i*3+0] = vv.x;
    vel[i*3+1] = vv.y;
    vel[i*3+2] = vv.z;
  }

  geom.attributes.position.needsUpdate = true;
  geom.attributes.aAlpha.needsUpdate = true;
  geom.attributes.aBoost.needsUpdate = true;

  if (halo.visible) {
    const i = (selected >= 0) ? selected : hovered;
    if (i >= 0) {
      haloPos[0] = pos[i*3+0];
      haloPos[1] = pos[i*3+1];
      haloPos[2] = pos[i*3+2];
      haloGeom.attributes.position.needsUpdate = true;
    }
  }

  if (edges.visible) {
    buildAllEdges();
  }
}

function updateFlyTo(dt) {
  if (!flyTo) return;
  flyTo.t += dt;
  const x = Math.min(1, flyTo.t / flyTo.dur);
  const e = easeInOutCubic(x);
  camera.position.lerpVectors(flyTo.fromPos, flyTo.toPos, e);
  orbit.target.lerpVectors(flyTo.fromTarget, flyTo.toTarget, e);
  if (x >= 1) flyTo = null;
}

// ---------- Run loop ----------
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(0.03, clock.getDelta());
  const t = clock.elapsedTime;

  if (wave.active) {
    wave.t += dt;
    if (wave.t * CFG.WAVE_SPEED > CFG.WAVE_MAX_DIST) wave.active = false;
  }

  if (needsPick) {
    needsPick = false;
    pickPoint();
  }

  if (!isFly) {
    orbitKeyboard(dt);
    orbit.update();
  } else {
    flyKeyboard(dt);
  }

  updateFlyTo(dt);
  updateNodes(dt, t);
  if (halo.visible) {
    const base = (selected >= 0) ? 26.0 : 20.0;
    haloMat.uniforms.uPointSize.value = base * (1.0 + 0.08 * Math.sin(t * 4.2));
  }

  renderer.render(scene, camera);
}
animate();

// ---------- UX: pointer lock change ----------
document.addEventListener("pointerlockchange", () => {
  // preserved as-is
});

// ---------- Initial panel ----------
clearPanel();
updateSelectedUI();

// ---------- Build all edges at startup ----------
buildAllEdges();

// ---------- Backend data bootstrap ----------
/**
 * Try to load real data from the Cortex backend.
 * If it succeeds and there are nodes, replace the fake data in-place.
 * If it fails (backend offline, no data, etc.) keep the demo data.
 */
async function tryLoadBackendData() {
  showLoading("Connecting to Cortex backend…");

  try {
    // 1. Health check
    const health = await healthCheck();
    if (!health.ollama_connected) {
      showToast("Ollama is not running — using demo data", "info", 5000);
      hideLoading();
      return;
    }

    showLoading("Loading conversations…");

    // 2. Fetch visualization data
    const viz = await fetchChats();

    if (!viz.nodes || viz.nodes.length === 0) {
      showToast("No conversations yet — upload a chat export to get started", "info", 5000);
      hideLoading();
      return;
    }

    showToast(`Loaded ${viz.nodes.length} conversations from backend`, "success", 3000);
    backendDataLoaded = true;

    // Build the lookup index
    rebuildIdIndex();
    hideLoading();
  } catch (err) {
    console.warn("[init] Backend unavailable, keeping demo data:", err.message);
    showToast("Backend offline — showing demo data", "info", 5000);
    hideLoading();
  }
}

// Kick off backend loading (non-blocking — animation already runs)
tryLoadBackendData();

// Hide loading overlay after a safety timeout (in case something hangs)
setTimeout(() => hideLoading(), 12000);

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
  N: 0,               // populated from backend
  D: 128,             // vector dim
  CLUSTERS: 8,
  SPACE_SCALE: 8,     // spread in world units (UMAP ~[-10,10] * 8 = [-80,80])
  POINT_SIZE_PX: 4.5, // base px — big enough to see clusters
  HOVER_BOOST: 1.5,
  SELECT_BOOST: 2.2,
  EDGE_K: 6,          // top-K neighbors
  EDGE_MIN_SIM: 0.20, // similarity threshold for showing edges
  EDGE_FADE_DIST: 50,
  DRIFT_STRENGTH: 0.25,  // gentle motion
  SPRING_K: 4.0,         // spring stiffness
  DAMPING: 3.5,          // critical-ish damping factor
  GRAVITY_RADIUS: 20,
  GRAVITY_STRENGTH: 1.5,
  FOCUS_RADIUS: 50,
  WAVE_SPEED: 60,
  WAVE_WIDTH: 14,
  WAVE_STRENGTH: 0.6,
  WAVE_MAX_DIST: 220,
  STAR_COUNT: 1200,
  FOG_NEAR: 80,
  FOG_FAR: 350,
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

// Upload modal DOM refs
const uploadBtn = document.getElementById("uploadBtn");
const uploadModal = document.getElementById("uploadModal");
const closeUploadBtn = document.getElementById("closeUpload");
const dropZone = document.getElementById("dropZone");
const uploadFileInput = document.getElementById("uploadFileInput");
const uploadList = document.getElementById("uploadList");
const uploadActions = document.getElementById("uploadActions");
const uploadStartBtn = document.getElementById("uploadStart");
const uploadClearBtn = document.getElementById("uploadClear");
const uploadSummary = document.getElementById("uploadSummary");

// ---------- Loading / Toast helpers ----------
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingMessage = document.getElementById("loadingMessage");
const toastContainer = document.getElementById("toastContainer");
const retryButton = document.getElementById("retryButton");

/** Show the fullscreen loading overlay with a custom message. */
function showLoading(msg = "Loading…", showRetry = false) {
  loadingMessage.textContent = msg;
  loadingOverlay.classList.remove("hidden");
  loadingOverlay.style.display = "flex";

  // Show/hide retry button
  if (retryButton) {
    retryButton.style.display = showRetry ? "block" : "none";
  }
}

/** Hide the fullscreen loading overlay. */
function hideLoading() {
  loadingOverlay.classList.add("hidden");
  if (retryButton) {
    retryButton.style.display = "none";
  }
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
  // Don't disable the search input — let the user keep editing while
  // a search is in flight.  The debounce timer will re-trigger if they
  // change the query, and the loading spinner gives visual feedback.
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
orbit.maxDistance = 400;

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

function clamp01(x) { return Math.max(0, Math.min(1, x)); }

// ---------- Data arrays (start empty — populated by backend) ----------
let nodes = [];
let vectors = [];
let clusterId = new Uint8Array(0);
let timestamps = [];

let anchors = new Float32Array(0);
let pos = new Float32Array(0);
let vel = new Float32Array(0);

/** True until the first successful backend data load positions the camera. */
let firstLoad = true;

// ---------- Empty-state + new DOM refs ----------
const emptyState = document.getElementById("emptyState");
const pMsgCount = document.getElementById("pMsgCount");
const emptyUploadInput = document.getElementById("emptyUploadInput");

function showEmptyState() {
  if (emptyState) emptyState.style.display = "flex";
}
function hideEmptyState() {
  if (emptyState) emptyState.style.display = "none";
}

// ---------- Build cluster filter UI ----------
let backendClusterMetadata = null;

function clusterName(c) {
  if (backendClusterMetadata) {
    const meta = backendClusterMetadata.find(m => m.cluster_id === c);
    if (meta && meta.cluster_name) {
      return `${c} — ${meta.cluster_name}`;
    }
  }
  return `Cluster ${c}`;
}
// Start with just the "All" option (populated after backend load)
clusterSel.innerHTML = "";
const optAll = document.createElement("option");
optAll.value = "-1";
optAll.textContent = "All";
clusterSel.appendChild(optAll);

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
let aColor = new Float32Array(CFG.N * 3);
let aAlpha = new Float32Array(CFG.N);
let aBoost = new Float32Array(CFG.N);
let aBoostTarget = new Float32Array(CFG.N);
let aBoostBase = new Float32Array(CFG.N);
// Initial colors — will be overwritten by rebuildPointsGeometry after backend load
for (let i=0; i<CFG.N; i++) {
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
  const r = 500 + rng()*300;  // far away: 500-800 radius
  const u = rng()*2 - 1;
  const tt = rng()*Math.PI*2;
  const s = Math.sqrt(1 - u*u);
  const x = r * s * Math.cos(tt);
  const y = r * u;
  const z = r * s * Math.sin(tt);
  starPos[i*3+0] = x;
  starPos[i*3+1] = y;
  starPos[i*3+2] = z;

  const c = new THREE.Color().setHSL(0.58 + (rng()*0.06), 0.20, 0.55 + rng()*0.15);
  starCol[i*3+0] = c.r;
  starCol[i*3+1] = c.g;
  starCol[i*3+2] = c.b;
}
starsGeom.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
starsGeom.setAttribute("color", new THREE.BufferAttribute(starCol, 3));

const starsMat = new THREE.PointsMaterial({
  size: 0.8,
  sizeAttenuation: true,
  transparent: true,
  opacity: 0.35,
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
  const bestSim = new Float32Array(K);
  const bestIdx = new Int32Array(K);
  for (let k=0; k<K; k++) { bestSim[k] = -1e9; bestIdx[k] = -1; }

  // Use vector similarity if vectors are available, otherwise use spatial distance
  const useVectors = vectors[idx] != null;

  if (useVectors) {
    const v = vectors[idx];
    for (let j=0; j<CFG.N; j++) {
      if (j === idx) continue;
      if (!vectors[j]) continue;
      const sim = dot(v, vectors[j]);
      let minK = 0;
      for (let k=1; k<K; k++) if (bestSim[k] < bestSim[minK]) minK = k;
      if (sim > bestSim[minK]) {
        bestSim[minK] = sim;
        bestIdx[minK] = j;
      }
    }
  } else {
    // Use spatial proximity (inverse distance)
    const px = pos[idx*3+0], py = pos[idx*3+1], pz = pos[idx*3+2];
    for (let j=0; j<CFG.N; j++) {
      if (j === idx) continue;
      const dx = pos[j*3+0] - px;
      const dy = pos[j*3+1] - py;
      const dz = pos[j*3+2] - pz;
      const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
      // Convert distance to similarity (closer = higher score)
      const sim = dist > 0 ? 1.0 / (1.0 + dist * 0.1) : 1.0;
      let minK = 0;
      for (let k=1; k<K; k++) if (bestSim[k] < bestSim[minK]) minK = k;
      if (sim > bestSim[minK]) {
        bestSim[minK] = sim;
        bestIdx[minK] = j;
      }
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
  edges.visible = edgeCount > 0;
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
  if (pMsgCount) pMsgCount.textContent = n.messageCount != null ? n.messageCount : "—";
  pTags.textContent = n.tags.join(", ");
  pSnippet.textContent = n.full || n.snippet || "";

  pNeighbors.innerHTML = "";

  // Use backend semantic search for real similarity scores
  if (backendDataLoaded && n.backendId) {
    pNeighbors.innerHTML = `<div class="nbItem" style="opacity:0.5"><span class="inline-spinner"></span> Finding similar…</div>`;
    searchChats(n.title, CFG.EDGE_K + 1).then((res) => {
      pNeighbors.innerHTML = "";
      for (const r of res.results) {
        // Skip the selected conversation itself
        if (r.conversation_id === n.backendId) continue;
        const j = nodeIdToIndex.get(String(r.conversation_id));
        if (j == null) continue;
        const item = document.createElement("div");
        item.className = "nbItem";
        item.innerHTML = `<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:240px">${nodes[j].title}</span>
                          <span class="badge">${r.score.toFixed(2)}</span>`;
        item.addEventListener("click", () => setSelected(j));
        pNeighbors.appendChild(item);
      }
      if (!pNeighbors.children.length) {
        pNeighbors.innerHTML = `<div class="nbItem" style="opacity:0.5">No similar conversations found</div>`;
      }
    }).catch(() => {
      // Fallback to spatial neighbors if backend search fails
      pNeighbors.innerHTML = "";
      const neigh = topKNeighbors(i, CFG.EDGE_K);
      for (let k = 0; k < neigh.length; k++) {
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
    });
  } else {
    // Local fallback (no backend)
    const neigh = topKNeighbors(i, CFG.EDGE_K);
    for (let k = 0; k < neigh.length; k++) {
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

      // Show error with fallback to summary
      const fallback = n.full || n.snippet || "";
      pSnippet.innerHTML = `
        <div style="color: #f87171; font-size: 12px; margin-bottom: 8px;">
          ⚠️ Failed to load full details: ${err.message}
        </div>
        <div style="opacity: 0.8;">${fallback || "(No summary available)"}</div>
      `;
    });
  }
}

function clearPanel() {
  pTitle.textContent = "None";
  pCluster.textContent = "—";
  pTime.textContent = "—";
  if (pMsgCount) pMsgCount.textContent = "—";
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

    // Show "No results" message if search returned nothing
    if (list.length === 0) {
      resultsEl.style.display = "block";
      resultsEl.innerHTML = `<div class="resItem" style="cursor: default; opacity: 0.6;">No results found for "${q}"</div>`;
    }
  } catch (err) {
    console.error("[search]", err);
    showToast(`Search error: ${err.message}`, "error", 5000);

    // Show error in results dropdown with retry option
    resultsEl.style.display = "block";
    resultsEl.innerHTML = `
      <div class="resItem" style="cursor: default; color: #f87171;">
        <div style="font-weight: 500;">Search failed</div>
        <div style="font-size: 11px; opacity: 0.8; margin-top: 4px;">${err.message}</div>
      </div>
    `;
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

async function openPromptModal() {
  const ids = Array.from(selectedSet).slice(0, 5);
  // Collect backend conversation IDs
  const conversationIds = ids.map(i => nodes[i]?.backendId).filter(Boolean);
  if (conversationIds.length === 0) {
    showToast("No valid conversations selected", "error", 3000);
    return;
  }

  // Show modal immediately with loading state
  promptText.value = "Generating system prompt…";
  promptModal.style.display = "block";
  copyPromptBtn.disabled = true;

  try {
    const { generatePrompt } = await import("./api.js");
    const result = await generatePrompt(conversationIds);
    promptText.value = result.prompt;
    showToast(
      `Prompt generated from ${result.conversations_used} chats (${(result.processing_time_ms / 1000).toFixed(1)}s)`,
      "success",
      3000
    );
  } catch (err) {
    console.error("[prompt]", err);
    promptText.value = `Error generating prompt: ${err.message}\n\nFallback context pack:\n\n${buildContextPack(ids)}`;
    showToast("Prompt generation failed — showing fallback", "error", 4000);
  } finally {
    copyPromptBtn.disabled = false;
  }
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
  // Check if user is typing in an input field FIRST — before adding to keys
  const isTyping = document.activeElement &&
    (document.activeElement.tagName === "INPUT" ||
     document.activeElement.tagName === "TEXTAREA" ||
     document.activeElement.tagName === "SELECT" ||
     document.activeElement.isContentEditable);

  // Only track movement keys when NOT typing in an input
  if (!isTyping) {
    keys.add(e.code);
    shiftDown = e.shiftKey;
  }

  if (e.key === "/" && !isTyping) {
    e.preventDefault();
    searchEl.focus();
    searchEl.select();
    showResults(searchNodes(searchEl.value));
    return;
  }

  if (e.key === "?" && !isTyping) {
    e.preventDefault();
    toggleHelp();
    return;
  }

  if (e.key === "Escape") {
    e.preventDefault();
    if (document.pointerLockElement) document.exitPointerLock();
    if (isTyping) {
      document.activeElement.blur();
      hideResults();
    } else {
      setSelected(-1);
      setHovered(-1);
      hideResults();
      helpEl.style.display = "none";
    }
    return;
  }

  // Skip game-style bindings when typing in an input
  if (isTyping) {
    // Still allow arrow keys and Enter for search results navigation
    if (resultsEl.style.display === "block") {
      if (e.key === "ArrowDown") { e.preventDefault(); setActiveResult(activeResIndex + 1); }
      if (e.key === "ArrowUp") { e.preventDefault(); setActiveResult(activeResIndex - 1); }
      if (e.key === "Enter") { e.preventDefault(); selectActiveResult(); }
    }
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

// Clear all keys when window loses focus to prevent stuck camera drift
window.addEventListener("blur", () => { keys.clear(); shiftDown = false; });
document.addEventListener("visibilitychange", () => {
  if (document.hidden) { keys.clear(); shiftDown = false; }
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
      aAlpha[i] = match ? 1.0 : 0.04;
      // Boost matching cluster nodes so they "light up"
      aBoostTarget[i] = match ? 1.8 : 0.5;
    } else {
      aAlpha[i] = 0.95;
      aBoostTarget[i] = 1.0;
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

// ---------- Helper functions for dynamic data loading ----------

/**
 * Auto-fit camera to enclose all loaded nodes with comfortable padding.
 */
function autoFitCamera() {
  if (CFG.N === 0) return;

  // Compute bounding sphere from anchor positions
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < CFG.N; i++) {
    cx += anchors[i * 3 + 0];
    cy += anchors[i * 3 + 1];
    cz += anchors[i * 3 + 2];
  }
  cx /= CFG.N; cy /= CFG.N; cz /= CFG.N;

  let maxR = 0;
  for (let i = 0; i < CFG.N; i++) {
    const dx = anchors[i * 3 + 0] - cx;
    const dy = anchors[i * 3 + 1] - cy;
    const dz = anchors[i * 3 + 2] - cz;
    const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (r > maxR) maxR = r;
  }

  // Position camera to see the whole bounding sphere
  const fov = camera.fov * (Math.PI / 180);
  const dist = (maxR / Math.sin(fov / 2)) * 1.2; // 1.2x padding

  orbit.target.set(cx, cy, cz);
  camera.position.set(cx, cy + dist * 0.25, cz + dist);
  orbit.update();

  // Update home position for reset
  homeCam.pos.copy(camera.position);
  homeCam.target.copy(orbit.target);

  // Also update fog to match the data spread
  const fogNear = dist * 0.15;
  const fogFar = dist * 2.5;
  scene.fog.near = fogNear;
  scene.fog.far = fogFar;
  pointMat.uniforms.uFogNear.value = fogNear;
  pointMat.uniforms.uFogFar.value = fogFar;
}

/**
 * Rebuild cluster filter dropdown with backend cluster data
 */
function rebuildClusterFilter(clusterIds, clusterMetadata = null) {
  clusterSel.innerHTML = "";

  const optAll = document.createElement("option");
  optAll.value = "-1";
  optAll.textContent = "All";
  clusterSel.appendChild(optAll);

  for (const cid of clusterIds) {
    const opt = document.createElement("option");
    opt.value = String(cid);

    // Use backend cluster name if available, otherwise fallback
    if (clusterMetadata) {
      const meta = clusterMetadata.find(m => m.cluster_id === cid);
      opt.textContent = meta ? `${cid} — ${meta.cluster_name || 'Cluster ' + cid}` : `Cluster ${cid}`;
    } else {
      opt.textContent = `Cluster ${cid}`;
    }

    clusterSel.appendChild(opt);
  }

  // Build visual cluster legend with colored dots
  buildClusterLegend(clusterIds, clusterMetadata);
}

/** Hex palette matching the shader CLUSTER_PALETTE */
const CLUSTER_HEX = [
  "#00D4FF", "#FF6B9D", "#7CFF6B", "#FFA64D", "#B48CFF",
  "#FFE44D", "#FF4D6A", "#4DFFF0", "#FF8CDB", "#8CFFB4",
  "#6BAAFF", "#FFB86B"
];

/**
 * Build an interactive cluster color legend in the bottom-left panel.
 */
function buildClusterLegend(clusterIds, clusterMetadata) {
  const legend = document.getElementById("clusterLegend");
  if (!legend) return;
  legend.innerHTML = "";

  for (const cid of clusterIds) {
    const color = CLUSTER_HEX[cid % CLUSTER_HEX.length];
    let name = `Cluster ${cid}`;
    let count = 0;

    if (clusterMetadata) {
      const meta = clusterMetadata.find(m => m.cluster_id === cid);
      if (meta) {
        name = meta.cluster_name || name;
        count = meta.count || 0;
      }
    }

    const item = document.createElement("div");
    item.className = "clusterLegendItem";
    item.innerHTML = `
      <span class="clusterDot" style="background:${color}; color:${color}"></span>
      <span>${name}</span>
      ${count ? `<span class="clusterLegendCount">${count}</span>` : ""}
    `;
    item.addEventListener("click", () => {
      clusterSel.value = String(cid);
      clusterSel.dispatchEvent(new Event("change"));
    });
    legend.appendChild(item);
  }
}

/**
 * Rebuild points geometry with new data
 */
function rebuildPointsGeometry() {
  const N = nodes.length;

  // Update position attribute
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));

  // Rebuild ALL per-point arrays (reassign outer-scope variables so the
  // animation loop, hover/select handlers, and edge builder always
  // reference the correctly-sized arrays).
  aColor = new Float32Array(N * 3);
  aAlpha = new Float32Array(N);
  aBoost = new Float32Array(N);
  aBoostTarget = new Float32Array(N);
  aBoostBase = new Float32Array(N);

  // Curated cluster color palette — maximally distinct, vibrant, accessible
  const CLUSTER_PALETTE = [
    new THREE.Color(0x00D4FF),  // 0  cyan
    new THREE.Color(0xFF6B9D),  // 1  pink
    new THREE.Color(0x7CFF6B),  // 2  lime green
    new THREE.Color(0xFFA64D),  // 3  warm orange
    new THREE.Color(0xB48CFF),  // 4  lavender
    new THREE.Color(0xFFE44D),  // 5  golden yellow
    new THREE.Color(0xFF4D6A),  // 6  coral red
    new THREE.Color(0x4DFFF0),  // 7  turquoise
    new THREE.Color(0xFF8CDB),  // 8  hot pink
    new THREE.Color(0x8CFFB4),  // 9  mint
    new THREE.Color(0x6BAAFF),  // 10 sky blue
    new THREE.Color(0xFFB86B),  // 11 peach
  ];

  for (let i = 0; i < N; i++) {
    const c = clusterId[i];
    const col = CLUSTER_PALETTE[c % CLUSTER_PALETTE.length];
    aColor[i * 3 + 0] = col.r;
    aColor[i * 3 + 1] = col.g;
    aColor[i * 3 + 2] = col.b;
    aAlpha[i] = 0.95;
    aBoost[i] = 1.0;
    aBoostTarget[i] = 1.0;
    aBoostBase[i] = 1.0;
  }

  geom.setAttribute("aColor", new THREE.BufferAttribute(aColor, 3));
  geom.setAttribute("aAlpha", new THREE.BufferAttribute(aAlpha, 1));
  geom.setAttribute("aBoost", new THREE.BufferAttribute(aBoost, 1));

  // Mark attributes as needing update
  geom.attributes.position.needsUpdate = true;
  geom.attributes.aColor.needsUpdate = true;
  geom.attributes.aAlpha.needsUpdate = true;
  geom.attributes.aBoost.needsUpdate = true;
}

// ---------- Upload modal logic ----------
/** @type {File[]} Files staged for upload */
let stagedFiles = [];
let uploadInProgress = false;

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB

function openUploadModal() {
  if (uploadModal) {
    uploadModal.style.display = "flex";
    resetUploadState();
  }
}

function closeUploadModal() {
  if (uploadModal) {
    uploadModal.style.display = "none";
    if (!uploadInProgress) resetUploadState();
  }
}

function resetUploadState() {
  stagedFiles = [];
  renderFileList();
  if (uploadSummary) { uploadSummary.style.display = "none"; uploadSummary.textContent = ""; }
  if (uploadActions) uploadActions.style.display = "none";
}

function fmtSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

/** Validate and add files (dedup by name). */
function addFiles(fileListOrArray) {
  const newFiles = Array.from(fileListOrArray);
  const existing = new Set(stagedFiles.map(f => f.name));
  let rejected = 0;

  for (const f of newFiles) {
    if (existing.has(f.name)) continue;
    if (!f.name.endsWith(".html")) {
      rejected++;
      showToast(`Skipped "${f.name}" — only .html files accepted`, "info", 3000);
      continue;
    }
    if (f.size > MAX_FILE_SIZE) {
      rejected++;
      showToast(`Skipped "${f.name}" — exceeds 10 MB limit`, "info", 3000);
      continue;
    }
    stagedFiles.push(f);
    existing.add(f.name);
  }

  renderFileList();
  if (uploadActions) uploadActions.style.display = stagedFiles.length ? "flex" : "none";
  if (uploadSummary) uploadSummary.style.display = "none";
}

function removeFile(idx) {
  stagedFiles.splice(idx, 1);
  renderFileList();
  if (uploadActions) uploadActions.style.display = stagedFiles.length ? "flex" : "none";
}

function renderFileList() {
  if (!uploadList) return;
  uploadList.innerHTML = "";

  for (let i = 0; i < stagedFiles.length; i++) {
    const f = stagedFiles[i];
    const item = document.createElement("div");
    item.className = "uploadItem";
    item.id = `uploadItem-${i}`;
    item.innerHTML = `
      <div>
        <div class="fileName">${f.name}</div>
        <div class="fileSize">${fmtSize(f.size)}</div>
      </div>
      <div class="fileStatus">
        <button class="removeFile" data-idx="${i}" title="Remove">&times;</button>
      </div>
    `;
    uploadList.appendChild(item);
  }

  // Attach remove handlers
  uploadList.querySelectorAll(".removeFile").forEach(btn => {
    btn.addEventListener("click", (e) => {
      if (uploadInProgress) return;
      removeFile(parseInt(e.currentTarget.dataset.idx, 10));
    });
  });
}

/** Set per-file status (uploading / success / error). */
function setFileStatus(idx, status, detail = "") {
  const item = document.getElementById(`uploadItem-${idx}`);
  if (!item) return;

  item.classList.remove("success", "error");
  const statusEl = item.querySelector(".fileStatus");

  if (status === "uploading") {
    statusEl.innerHTML = `<span class="inline-spinner"></span>`;
    // Add progress bar
    let prog = item.querySelector(".uploadProgress");
    if (!prog) {
      prog = document.createElement("div");
      prog.className = "uploadProgress";
      prog.innerHTML = `<div class="bar"></div>`;
      item.appendChild(prog);
    }
    prog.querySelector(".bar").style.width = "30%";
  } else if (status === "success") {
    item.classList.add("success");
    statusEl.innerHTML = `<span style="color:var(--good)">✓</span>`;
    const prog = item.querySelector(".uploadProgress");
    if (prog) prog.querySelector(".bar").style.width = "100%";
  } else if (status === "error") {
    item.classList.add("error");
    statusEl.innerHTML = `<span style="color:var(--bad)" title="${detail}">✗</span>`;
    const prog = item.querySelector(".uploadProgress");
    if (prog) prog.querySelector(".bar").style.width = "100%";
    if (prog) prog.querySelector(".bar").style.background = "var(--bad)";
  }
}

/** Upload all staged files sequentially. */
async function startUpload() {
  if (!stagedFiles.length || uploadInProgress) return;
  uploadInProgress = true;

  if (uploadStartBtn) uploadStartBtn.disabled = true;
  if (uploadClearBtn) uploadClearBtn.disabled = true;

  let success = 0;
  let failed = 0;

  for (let i = 0; i < stagedFiles.length; i++) {
    setFileStatus(i, "uploading");
    try {
      const result = await uploadChatFile(stagedFiles[i], i === stagedFiles.length - 1);
      if (result.success) {
        success++;
        setFileStatus(i, "success");
      } else {
        failed++;
        setFileStatus(i, "error", result.error || "Unknown error");
      }
    } catch (err) {
      failed++;
      setFileStatus(i, "error", err.message);
    }
  }

  uploadInProgress = false;
  if (uploadStartBtn) uploadStartBtn.disabled = false;
  if (uploadClearBtn) uploadClearBtn.disabled = false;

  // Show summary
  if (uploadSummary) {
    uploadSummary.style.display = "block";
    if (failed === 0) {
      uploadSummary.className = "uploadSummary allGood";
      uploadSummary.textContent = `✓ ${success} file${success !== 1 ? "s" : ""} uploaded successfully`;
    } else {
      uploadSummary.className = "uploadSummary hasErrors";
      uploadSummary.textContent = `${success} succeeded, ${failed} failed`;
    }
  }

  // Refresh 3D visualization with new data
  if (success > 0) {
    // Close modal immediately and show loading overlay
    closeUploadModal();
    showLoading("Processing conversations — clustering & positioning…");

    // Poll the backend until the node count stabilises and positions are set
    // (the ingest endpoint processes all convs + UMAP before returning, but
    //  the browser may have already gotten a partial response on slow networks)
    let attempts = 0;
    const maxAttempts = 60;  // up to ~60 s
    let lastCount = -1;
    let stableRounds = 0;

    while (attempts < maxAttempts) {
      await new Promise(r => setTimeout(r, 1000));
      attempts++;
      try {
        const check = await fetchChats();
        const count = check.nodes ? check.nodes.length : 0;
        // Check that nodes exist AND have real positions (not all at origin)
        const hasPositions = count > 0 && check.nodes.some(
          n => n.position && (n.position[0] !== 0 || n.position[1] !== 0 || n.position[2] !== 0)
        );
        if (hasPositions && count === lastCount) {
          stableRounds++;
        } else {
          stableRounds = 0;
        }
        lastCount = count;
        showLoading(`Processing conversations… ${count} ready`);
        // Data is stable for 2 consecutive checks and has real positions
        if (hasPositions && stableRounds >= 2) break;
      } catch {
        // Backend still busy — keep waiting
      }
    }

    // Force first-load so camera fits to the new data
    firstLoad = true;
    await tryLoadBackendData();
  }
}

// Wire up upload UI events
if (uploadBtn) uploadBtn.addEventListener("click", openUploadModal);
if (closeUploadBtn) closeUploadBtn.addEventListener("click", closeUploadModal);

if (uploadFileInput) {
  uploadFileInput.addEventListener("change", (e) => {
    addFiles(e.target.files);
    uploadFileInput.value = ""; // reset so same file can be re-selected
  });
}

if (uploadStartBtn) uploadStartBtn.addEventListener("click", startUpload);
if (uploadClearBtn) uploadClearBtn.addEventListener("click", resetUploadState);

// Close modal on backdrop click
if (uploadModal) {
  uploadModal.addEventListener("click", (e) => {
    if (e.target === uploadModal && !uploadInProgress) closeUploadModal();
  });
}

// Drag-and-drop on the drop zone
if (dropZone) {
  dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
  dropZone.addEventListener("dragleave", () => { dropZone.classList.remove("dragover"); });
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
  });
}

// Global drag-and-drop: open modal when dragging files over the page
let bodyDragCounter = 0;
document.addEventListener("dragenter", (e) => {
  e.preventDefault();
  bodyDragCounter++;
  if (bodyDragCounter === 1 && uploadModal && uploadModal.style.display !== "flex") {
    openUploadModal();
  }
});
document.addEventListener("dragleave", (e) => {
  e.preventDefault();
  bodyDragCounter--;
  if (bodyDragCounter <= 0) bodyDragCounter = 0;
});
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => {
  e.preventDefault();
  bodyDragCounter = 0;
  // If dropped outside the dropZone but modal is open, add files anyway
  if (uploadModal && uploadModal.style.display === "flex" && e.dataTransfer.files.length) {
    addFiles(e.dataTransfer.files);
  }
});

// ---------- Backend data bootstrap ----------
/**
 * Try to load real data from the Cortex backend.
 * If it succeeds and there are nodes, replace the empty data arrays.
 * If backend returns 0 nodes, show the empty-state overlay.
 * If backend is unreachable, show empty-state with retry option.
 */
async function tryLoadBackendData() {
  showLoading("Connecting to Cortex backend…");
  hideEmptyState();

  try {
    // 1. Health check
    const health = await healthCheck();
    if (!health.ollama_connected) {
      showToast("Ollama is not running — start Ollama and retry", "info", 5000);
      hideLoading();
      showEmptyState();
      return;
    }

    showLoading("Loading conversations…");

    // 2. Fetch visualization data — but poll until fully ready
    //    (handles page-refresh mid-ingest: nodes committed but UMAP not done yet)
    let viz = await fetchChats();

    if (!viz.nodes || viz.nodes.length === 0) {
      showToast("No conversations yet — upload a chat export to get started", "info", 5000);
      hideLoading();
      showEmptyState();
      return;
    }

    // Check if data is fully processed (UMAP has assigned real positions)
    const allHavePositions = (nodes) =>
      nodes.every(n => n.position && (n.position[0] !== 0 || n.position[1] !== 0 || n.position[2] !== 0));
    const hasClusters = (data) => data.clusters && data.clusters.length > 0;

    if (!allHavePositions(viz.nodes) || !hasClusters(viz)) {
      // Data exists but UMAP/clustering hasn't finished — poll until ready
      showLoading(`Processing ${viz.nodes.length} conversations — waiting for positions…`);
      let attempts = 0;
      const maxAttempts = 120; // up to ~2 minutes
      let lastCount = viz.nodes.length;
      let stableRounds = 0;

      while (attempts < maxAttempts) {
        await new Promise(r => setTimeout(r, 1500));
        attempts++;
        try {
          const check = await fetchChats();
          const count = check.nodes ? check.nodes.length : 0;
          const ready = count > 0 && allHavePositions(check.nodes) && hasClusters(check);

          if (ready && count === lastCount) {
            stableRounds++;
          } else {
            stableRounds = 0;
          }
          lastCount = count;
          showLoading(`Processing conversations… ${count} found${ready ? " — finalising" : ""}`);

          if (ready && stableRounds >= 2) {
            viz = check; // use the fully-ready data
            break;
          }
        } catch {
          // Backend busy — keep waiting
        }
      }
    }

    // 3. Populate data arrays from backend
    const N = viz.nodes.length;

    nodes = new Array(N);
    vectors = new Array(N);
    clusterId = new Uint8Array(N);
    timestamps = new Array(N);
    anchors = new Float32Array(N * 3);
    pos = new Float32Array(N * 3);
    vel = new Float32Array(N * 3);

    const backendClusters = new Set();
    for (let i = 0; i < N; i++) {
      const node = viz.nodes[i];

      nodes[i] = {
        id: node.id,
        title: node.title || "Untitled",
        cluster: node.cluster_id ?? 0,
        tags: node.topics || [],
        snippet: node.summary || "",
        full: node.summary || "",
        time: new Date(node.created_at),
        backendId: node.id,
        messageCount: node.message_count ?? null
      };

      clusterId[i] = node.cluster_id ?? 0;
      timestamps[i] = new Date(node.created_at);
      backendClusters.add(node.cluster_id ?? 0);

      // Anchor = backend-generated 3D coordinate (UMAP projection)
      // Scale up by SPACE_SCALE so nodes fill the 3D world nicely
      const [rx, ry, rz] = node.position || [0, 0, 0];
      const ax = rx * CFG.SPACE_SCALE;
      const ay = ry * CFG.SPACE_SCALE;
      const az = rz * CFG.SPACE_SCALE;
      anchors[i * 3 + 0] = ax;
      anchors[i * 3 + 1] = ay;
      anchors[i * 3 + 2] = az;

      // Start pos near anchor with slight jitter for pop-in effect
      pos[i * 3 + 0] = ax + (Math.random() - 0.5) * 2.0;
      pos[i * 3 + 1] = ay + (Math.random() - 0.5) * 2.0;
      pos[i * 3 + 2] = az + (Math.random() - 0.5) * 2.0;

      vel[i * 3 + 0] = 0;
      vel[i * 3 + 1] = 0;
      vel[i * 3 + 2] = 0;

      // Vectors are null — search is done server-side
      vectors[i] = null;
    }

    CFG.N = N;

    // Store cluster metadata globally
    backendClusterMetadata = viz.clusters || null;

    // Rebuild cluster filter dropdown
    rebuildClusterFilter(Array.from(backendClusters).sort((a, b) => a - b), viz.clusters);

    // Rebuild geometry + edges with real data
    rebuildPointsGeometry();
    buildAllEdges();

    backendDataLoaded = true;
    rebuildIdIndex();

    // Auto-fit camera only on first load — after that the user controls the camera
    if (firstLoad) {
      autoFitCamera();
      firstLoad = false;
    }

    hideLoading();
    hideEmptyState();
    showToast(`Loaded ${N} conversations from backend`, "success", 3000);
  } catch (err) {
    console.warn("[init] Backend unavailable:", err.message);

    // Show loading overlay with retry option
    showLoading(`Connection failed: ${err.message}`, true);
    showToast("Backend offline — click Retry to reconnect", "error", 6000);

    // After timeout, hide loading and show empty state
    setTimeout(() => {
      if (loadingOverlay && !loadingOverlay.classList.contains("hidden")) {
        hideLoading();
        showEmptyState();
      }
    }, 12000);
  }
}

// ---------- Upload handler for empty-state ----------
if (emptyUploadInput) {
  emptyUploadInput.addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    hideEmptyState();
    showLoading("Uploading chat export…");

    try {
      const result = await uploadChatFile(file, true);
      if (!result.success) {
        throw new Error(result.error || "Upload failed");
      }

      showLoading("Processing conversations — clustering & positioning…");

      // Poll until data is fully ready (UMAP + clustering done)
      let attempts = 0;
      let lastCount = -1;
      let stableRounds = 0;
      while (attempts < 60) {
        await new Promise(r => setTimeout(r, 1000));
        attempts++;
        try {
          const check = await fetchChats();
          const count = check.nodes ? check.nodes.length : 0;
          const hasPositions = count > 0 && check.nodes.some(
            n => n.position && (n.position[0] !== 0 || n.position[1] !== 0 || n.position[2] !== 0)
          );
          if (hasPositions && count === lastCount) stableRounds++;
          else stableRounds = 0;
          lastCount = count;
          showLoading(`Processing conversations… ${count} ready`);
          if (hasPositions && stableRounds >= 2) break;
        } catch { /* keep waiting */ }
      }

      firstLoad = true;
      await tryLoadBackendData();
    } catch (err) {
      console.error("[upload] Failed:", err);
      showToast(`Upload failed: ${err.message}`, "error", 5000);
      hideLoading();
      showEmptyState();
    } finally {
      emptyUploadInput.value = "";
    }
  });
}

// Retry button event listener
if (retryButton) {
  retryButton.addEventListener("click", () => {
    console.log("[retry] User requested retry");
    hideLoading();
    tryLoadBackendData();
  });
}

// Kick off backend loading (non-blocking — animation loop already runs)
// Show empty state initially while we wait for backend response
showEmptyState();
tryLoadBackendData();

// Safety timeout in case something hangs (3 min — ingest can take a while)
setTimeout(() => {
  if (loadingOverlay && !loadingOverlay.classList.contains("hidden")) {
    hideLoading();
    if (CFG.N === 0) showEmptyState();
  }
}, 180000);

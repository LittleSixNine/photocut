const CLIENT_NAME_PREFIX = `photocut-confirmation:${location.pathname}:`;
const MAGNIFIER_BUFFER_SIZE = 768;
const MAGNIFIER_REFRESH_MS = 100;
const MAGNIFIER_CROSSHAIR_GAP = 12;
const MAGNIFIER_CROSSHAIR_EXTENT = 30;
const MAGNIFIER_EDGE_LINE_WIDTH = 3;
const MAGNIFIER_CENTER_RADIUS = 1.5;
const MAIN_CROSSHAIR_GAP_RATIO = 0.7;
const MAIN_CROSSHAIR_EXTENT_RATIO = 1.6;
const MAIN_CENTER_DOT_RATIO = 0.12;
const CROSSHAIR_APEX_ANGLE_DEGREES = 20;
const elements = Object.fromEntries(
  [
    "workbench", "filename", "progress", "gui-identity", "lease-state",
    "image-size", "dirty-state", "photo-stage", "photo-preview",
    "corner-overlay", "corner-polygon", "corner-points", "stage-loading",
    "candidate-id", "candidate-status", "algorithm-version", "risk-list",
    "magnifier", "zoom-label", "corner-coordinates", "technical-details",
    "details-status", "details-list", "candidate-audit", "action-error", "error-banner", "error-message",
    "previous-button", "next-button", "candidate-button", "v52-button",
    "reset-button", "skip-button", "pause-button", "quit-button", "confirm-button",
    "zoom-2-button", "zoom-4-button", "zoom-8-button",
  ].map((id) => [id, document.getElementById(id)])
);

let state = null;
let writer = false;
let previewUrl = null;
let magnifierUrl = null;
let prefetchedUrl = null;
let previewRequest = null;
let prefetchRequest = null;
let magnifierRequest = null;
let heartbeatTimer = null;
let previewTransform = null;
let actionPending = false;
let imageFailed = false;
let ready = false;
let drag = null;
let dragFrame = null;
let magnifierImage = null;
let magnifierCenter = null;
let magnifierIndex = null;
let magnifierDrag = null;
let magnifierFrame = null;
let magnifierRefreshTimer = null;
let magnifierRefreshTarget = null;
let magnifierLastRefresh = 0;

const CANDIDATE_LABELS = {
  "v8:draft": "V8 主候选",
  "v52:gui": "v5.2 备选",
  primary: "主候选",
  top1: "主候选",
};
const STATUS_LABELS = {
  automatic: "自动检测",
  manual_review: "建议人工检查",
  "v8:draft": "等待确认",
  v8_recommended: "V8 建议候选",
  legacy: "兼容模式",
};
const RISK_LABELS = {
  outside_background_unverifiable: "外侧背景不足，请核对边界",
  insufficient_support: "边缘依据不足，请重点检查",
};

function clientId() {
  if (window.name.startsWith(CLIENT_NAME_PREFIX)) {
    const existing = window.name.slice(CLIENT_NAME_PREFIX.length);
    if (/^[0-9a-f-]{36}$/.test(existing)) return existing;
  }
  const value = crypto.randomUUID();
  window.name = CLIENT_NAME_PREFIX + value;
  return value;
}

const client = clientId();

function api(name, query = null) {
  const url = new URL(`api/${name}`, document.baseURI);
  if (query) url.search = query.toString();
  return url;
}

async function readResponse(response) {
  const type = response.headers.get("Content-Type") || "";
  const value = type.startsWith("application/json") ? await response.json() : null;
  if (!response.ok) {
    const detail = value?.error || {};
    const error = new Error(detail.message || detail.code || `http_${response.status}`);
    error.code = detail.code || `http_${response.status}`;
    error.scope = detail.scope || "session";
    throw error;
  }
  return value;
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return readResponse(response);
}

function showError(error) {
  const code = error.code || error.message || "unknown";
  if (error.scope === "action") {
    const messages = {
      reset_required: "已调整角点；切换候选前请先重置。",
    };
    elements["action-error"].textContent = messages[code] || `动作未执行：${error.message || code}`;
    elements["action-error"].hidden = false;
    return;
  }
  if (error.scope === "image") imageFailed = true;
  if (error.scope === "session") writer = false;
  elements["error-message"].textContent = `错误代码：${code}。请查看终端并刷新页面。`;
  elements["error-banner"].hidden = false;
  updateControls();
}

function clearError() {
  elements["error-banner"].hidden = true;
  elements["action-error"].hidden = true;
}

function setText(id, value) {
  elements[id].textContent = value ?? "—";
}

function candidateLabel(value) {
  if (CANDIDATE_LABELS[value]) return CANDIDATE_LABELS[value];
  if (String(value).includes("alternate")) return "备选候选";
  return value ? "检测候选" : "—";
}

function statusLabel(value) {
  if (!value) return "等待确认";
  return String(value).split("/").map((part) => {
    const normalized = part.trim();
    return STATUS_LABELS[normalized] || "等待确认";
  }).join(" · ");
}

function actionAllowed(kind) {
  if (!ready || !state || !writer || state.session.status !== "active") return false;
  const capabilities = state.editor.capabilities || {};
  if (kind === "previous") return state.progress.index > 0;
  if (kind === "next") return state.progress.index + 1 < state.progress.total;
  if (kind === "confirm") return !imageFailed && Boolean(capabilities.confirm);
  if (["candidate", "v52", "reset", "skip"].includes(kind)) {
    return Boolean(capabilities[kind]);
  }
  if (["move", "set_corner", "select_corner"].includes(kind)) return Boolean(capabilities.move);
  return true;
}

function updateControls() {
  const actions = {
    "previous-button": "previous",
    "next-button": "next",
    "candidate-button": "candidate",
    "v52-button": "v52",
    "reset-button": "reset",
    "skip-button": "skip",
    "pause-button": "pause",
    "quit-button": "quit",
    "confirm-button": "confirm",
  };
  for (const [id, kind] of Object.entries(actions)) {
    elements[id].disabled = actionPending || !actionAllowed(kind);
  }
  for (const zoom of [2, 4, 8]) {
    const button = elements[`zoom-${zoom}-button`];
    button.disabled = actionPending || !actionAllowed("set_zoom") || state?.editor.zoom === zoom;
    button.setAttribute("aria-pressed", String(state?.editor.zoom === zoom));
  }
}

function renderState(nextState) {
  state = nextState;
  elements.workbench.dataset.revision = String(state.revision);
  const editor = state.editor;
  const candidate = editor.candidate || {};
  const selectedCorner = editor.selected_corner >= 0 ? editor.selected_corner : 0;
  const readonly = Boolean(state.session.readonly);
  writer = !readonly;
  setText("filename", state.image.filename);
  elements.filename.title = state.image.filename;
  setText("progress", `${state.progress.number} / ${state.progress.total}`);
  setText("gui-identity", `GUI ${state.identity.gui || "—"}`);
  setText("image-size", `${state.image.width} × ${state.image.height} px`);
  setText("candidate-id", candidateLabel(candidate.id));
  elements["candidate-id"].title = candidate.id || "";
  setText("candidate-status", statusLabel(candidate.status));
  elements["candidate-status"].title = candidate.status || "";
  setText("algorithm-version", `算法 ${candidate.algorithm_version || state.identity.algorithm || "—"}`);
  setText("zoom-label", `角 ${selectedCorner + 1} · ${editor.zoom}×`);
  elements["lease-state"].dataset.writer = String(writer);
  elements["lease-state"].textContent = writer ? "当前页面编辑" : "只读：另一页面正在编辑";
  elements["dirty-state"].dataset.dirty = String(Boolean(editor.dirty));
  elements["dirty-state"].textContent = editor.dirty
    ? "已调整，尚未确认"
    : state.storage.formal ? "正式确认" : "已保存草稿";
  if (editor.dirty) {
    elements["action-error"].textContent = "已调整角点；切换候选前请先重置。";
    elements["action-error"].hidden = false;
  } else if (!actionPending) {
    elements["action-error"].hidden = true;
  }

  elements["risk-list"].replaceChildren();
  const risks = editor.risks || [];
  for (const risk of risks.length ? risks : ["未报告额外风险"]) {
    const item = document.createElement("li");
    item.textContent = RISK_LABELS[risk] || (risks.length ? "需要额外人工核对" : risk);
    if (risks.length) item.title = risk;
    if (!risks.length) item.dataset.empty = "true";
    elements["risk-list"].append(item);
  }

  elements["corner-coordinates"].replaceChildren();
  editor.corners.forEach(([x, y], index) => {
    const item = document.createElement("li");
    const label = document.createElement("span");
    const value = document.createElement("span");
    label.textContent = `角 ${index + 1}`;
    value.textContent = `${x}, ${y}`;
    if (index === selectedCorner) {
      item.classList.add("is-selected");
      item.setAttribute("aria-current", "true");
    }
    item.append(label, value);
    elements["corner-coordinates"].append(item);
  });
  elements.workbench.dataset.readonly = String(readonly);
  drawOverlay(editor, state.image);
  updateControls();
}

function drawOverlay(editor, image) {
  const selectedCorner = editor.selected_corner >= 0 ? editor.selected_corner : 0;
  elements["corner-overlay"].setAttribute("viewBox", `0 0 ${image.width} ${image.height}`);
  elements["corner-polygon"].setAttribute("points", editor.corners.map(([x, y]) => `${x},${y}`).join(" "));
  elements["corner-points"].replaceChildren();
  editor.corners.forEach(([x, y], index) => {
    const size = Math.max(image.width, image.height) * 0.01;
    const gap = size * MAIN_CROSSHAIR_GAP_RATIO;
    const extent = size * MAIN_CROSSHAIR_EXTENT_RATIO;
    const halfBase = crosshairHalfBase(extent - gap);
    const marker = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const hit = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    const crosshair = document.createElementNS("http://www.w3.org/2000/svg", "path");
    const center = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    marker.setAttribute("class", `corner-point${selectedCorner === index ? " is-selected" : ""}`);
    marker.dataset.cornerIndex = String(index);
    hit.setAttribute("class", "corner-hit");
    hit.setAttribute("cx", x);
    hit.setAttribute("cy", y);
    hit.setAttribute("r", size * 1.3);
    crosshair.setAttribute("class", "corner-crosshair");
    crosshair.setAttribute("d", [
      `M ${x - gap} ${y} L ${x - extent} ${y - halfBase} L ${x - extent} ${y + halfBase} Z`,
      `M ${x + gap} ${y} L ${x + extent} ${y - halfBase} L ${x + extent} ${y + halfBase} Z`,
      `M ${x} ${y - gap} L ${x - halfBase} ${y - extent} L ${x + halfBase} ${y - extent} Z`,
      `M ${x} ${y + gap} L ${x - halfBase} ${y + extent} L ${x + halfBase} ${y + extent} Z`,
    ].join(" "));
    center.setAttribute("class", "corner-center");
    center.setAttribute("cx", x);
    center.setAttribute("cy", y);
    center.setAttribute("r", size * MAIN_CENTER_DOT_RATIO);
    marker.append(hit, crosshair, center);
    elements["corner-points"].append(marker);
  });
}

function crosshairHalfBase(length) {
  return length * Math.tan(CROSSHAIR_APEX_ANGLE_DEGREES * Math.PI / 360);
}

function viewportQuery(extra = {}) {
  const rect = elements["photo-stage"].getBoundingClientRect();
  const dpr = Math.min(4, Math.max(0.5, window.devicePixelRatio || 1));
  return new URLSearchParams({
    css_width: String(Math.max(1, Math.round(rect.width))),
    css_height: String(Math.max(1, Math.round(rect.height))),
    dpr: String(dpr),
    ...extra,
  });
}

function replaceObjectUrl(previous, next) {
  if (previous) URL.revokeObjectURL(previous);
  return next;
}

async function loadPreview() {
  previewRequest?.abort();
  previewRequest = new AbortController();
  elements["stage-loading"].hidden = false;
  const response = await fetch(api("preview", viewportQuery({ image_token: state.image.token })), {
    signal: previewRequest.signal,
  });
  if (!response.ok) await readResponse(response);
  const transform = JSON.parse(response.headers.get("X-PhotoCut-Transform") || "{}");
  if (!Number.isFinite(transform.original_width) || !Number.isFinite(transform.original_height)) {
    const error = new Error("invalid_preview_transform");
    error.code = "invalid_preview_transform";
    error.scope = "image";
    throw error;
  }
  previewTransform = transform;
  previewUrl = replaceObjectUrl(previewUrl, URL.createObjectURL(await response.blob()));
  elements["photo-preview"].src = previewUrl;
  elements["photo-preview"].alt = `待确认照片：${state.image.filename}`;
  await elements["photo-preview"].decode();
  elements["stage-loading"].hidden = true;
  imageFailed = false;
  drawOverlay(state.editor, state.image);
  updateControls();
}

function clampMagnifierCenter([x, y]) {
  return [
    Math.max(0, Math.min(state.image.width - 1, roundTiesToEven(x))),
    Math.max(0, Math.min(state.image.height - 1, roundTiesToEven(y))),
  ];
}

async function loadMagnifier(center = null, requestedIndex = null) {
  magnifierRequest?.abort();
  const request = new AbortController();
  magnifierRequest = request;
  const imageToken = state.image.token;
  const index = requestedIndex ?? (
    state.editor.selected_corner >= 0 ? state.editor.selected_corner : 0
  );
  const requestedCenter = clampMagnifierCenter(
    center ?? state.editor.corners[index]
  );
  const query = new URLSearchParams({
    image_token: state.image.token,
    corner_index: String(index),
    zoom: String(state.editor.zoom),
    size: String(MAGNIFIER_BUFFER_SIZE),
    center_x: String(requestedCenter[0]),
    center_y: String(requestedCenter[1]),
  });
  const response = await fetch(api("magnifier", query), { signal: request.signal });
  if (!response.ok) await readResponse(response);
  const nextUrl = URL.createObjectURL(await response.blob());
  const image = new Image();
  image.src = nextUrl;
  await image.decode();
  if (magnifierRequest !== request || state.image.token !== imageToken) {
    URL.revokeObjectURL(nextUrl);
    return;
  }
  magnifierUrl = replaceObjectUrl(magnifierUrl, nextUrl);
  magnifierImage = image;
  magnifierCenter = requestedCenter;
  magnifierIndex = index;
  drawActiveMagnifier();
}

function drawMagnifier(offsetX = 0, offsetY = 0, editor = state?.editor) {
  if (!magnifierImage || !state || !editor) return;
  const canvas = elements.magnifier;
  const context = canvas.getContext("2d", { alpha: false });
  const index = editor.selected_corner >= 0 ? editor.selected_corner : 0;
  const center = canvas.width / 2;
  const corner = editor.corners[index];
  const styles = getComputedStyle(document.documentElement);
  const guide = styles.getPropertyValue("--color-crosshair").trim();
  const centerColor = styles.getPropertyValue("--color-corner-center").trim();
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.imageSmoothingEnabled = false;
  const bufferOffset = (canvas.width - MAGNIFIER_BUFFER_SIZE) / 2;
  context.drawImage(
    magnifierImage,
    bufferOffset + offsetX,
    bufferOffset + offsetY,
    MAGNIFIER_BUFFER_SIZE,
    MAGNIFIER_BUFFER_SIZE,
  );
  context.strokeStyle = guide;
  context.lineCap = "butt";
  context.lineWidth = MAGNIFIER_EDGE_LINE_WIDTH;
  context.globalAlpha = 0.55;
  for (const neighborIndex of [(index + 3) % 4, (index + 1) % 4]) {
    const neighbor = editor.corners[neighborIndex];
    const dx = neighbor[0] - corner[0];
    const dy = neighbor[1] - corner[1];
    const length = Math.hypot(dx, dy);
    if (!length) continue;
    const ux = dx / length;
    const uy = dy / length;
    context.beginPath();
    context.moveTo(
      center + ux * MAGNIFIER_CROSSHAIR_GAP,
      center + uy * MAGNIFIER_CROSSHAIR_GAP,
    );
    context.lineTo(center + ux * canvas.width * 2, center + uy * canvas.height * 2);
    context.stroke();
  }
  context.globalAlpha = 1;
  context.fillStyle = guide;
  const halfBase = crosshairHalfBase(MAGNIFIER_CROSSHAIR_EXTENT - MAGNIFIER_CROSSHAIR_GAP);
  context.beginPath();
  context.moveTo(center - MAGNIFIER_CROSSHAIR_GAP, center);
  context.lineTo(center - MAGNIFIER_CROSSHAIR_EXTENT, center - halfBase);
  context.lineTo(center - MAGNIFIER_CROSSHAIR_EXTENT, center + halfBase);
  context.closePath();
  context.moveTo(center + MAGNIFIER_CROSSHAIR_GAP, center);
  context.lineTo(center + MAGNIFIER_CROSSHAIR_EXTENT, center - halfBase);
  context.lineTo(center + MAGNIFIER_CROSSHAIR_EXTENT, center + halfBase);
  context.closePath();
  context.moveTo(center, center - MAGNIFIER_CROSSHAIR_GAP);
  context.lineTo(center - halfBase, center - MAGNIFIER_CROSSHAIR_EXTENT);
  context.lineTo(center + halfBase, center - MAGNIFIER_CROSSHAIR_EXTENT);
  context.closePath();
  context.moveTo(center, center + MAGNIFIER_CROSSHAIR_GAP);
  context.lineTo(center - halfBase, center + MAGNIFIER_CROSSHAIR_EXTENT);
  context.lineTo(center + halfBase, center + MAGNIFIER_CROSSHAIR_EXTENT);
  context.closePath();
  context.fill();
  context.globalAlpha = 1;
  context.fillStyle = centerColor;
  context.beginPath();
  context.arc(center, center, MAGNIFIER_CENTER_RADIUS, 0, Math.PI * 2);
  context.fill();
}

function magnifierPoint(event) {
  const canvas = elements.magnifier;
  const rect = canvas.getBoundingClientRect();
  return [
    (event.clientX - rect.left) * canvas.width / rect.width,
    (event.clientY - rect.top) * canvas.height / rect.height,
  ];
}

function roundTiesToEven(value) {
  const sign = Math.sign(value);
  const absolute = Math.abs(value);
  const lower = Math.floor(absolute);
  const rounded = absolute - lower === 0.5
    ? (lower % 2 === 0 ? lower : lower + 1)
    : Math.round(absolute);
  return sign * rounded;
}

function editorWithCorner(index, point) {
  const editor = {
    ...state.editor,
    corners: state.editor.corners.map((corner) => [...corner]),
    selected_corner: index,
  };
  editor.corners[index] = point;
  return editor;
}

function scheduleMagnifierRefresh(point, offsetX, offsetY, index) {
  const threshold = elements.magnifier.width / 3;
  if (
    magnifierIndex === index
    && Math.max(Math.abs(offsetX), Math.abs(offsetY)) < threshold
  ) return;
  magnifierRefreshTarget = { center: clampMagnifierCenter(point), index };
  if (magnifierRefreshTimer !== null) return;
  const delay = Math.max(
    0,
    MAGNIFIER_REFRESH_MS - (performance.now() - magnifierLastRefresh),
  );
  magnifierRefreshTimer = window.setTimeout(() => {
    magnifierRefreshTimer = null;
    magnifierLastRefresh = performance.now();
    const target = magnifierRefreshTarget;
    if (!target) return;
    void loadMagnifier(target.center, target.index).catch(() => {});
  }, delay);
}

function cancelMagnifierRefresh() {
  if (magnifierRefreshTimer !== null) window.clearTimeout(magnifierRefreshTimer);
  magnifierRefreshTimer = null;
  magnifierRefreshTarget = null;
  magnifierRequest?.abort();
}

function drawMagnifierDrag() {
  magnifierFrame = null;
  if (!magnifierDrag || !magnifierCenter) return;
  const corner = state.editor.corners[magnifierDrag.index];
  const point = [
    Math.max(0, Math.min(state.image.width - 1, corner[0] - magnifierDrag.dx / state.editor.zoom)),
    Math.max(0, Math.min(state.image.height - 1, corner[1] - magnifierDrag.dy / state.editor.zoom)),
  ];
  const editor = editorWithCorner(magnifierDrag.index, point);
  const offsetX = (magnifierCenter[0] - point[0]) * state.editor.zoom;
  const offsetY = (magnifierCenter[1] - point[1]) * state.editor.zoom;
  drawMagnifier(offsetX, offsetY, editor);
  scheduleMagnifierRefresh(point, offsetX, offsetY, magnifierDrag.index);
}

function drawActiveMagnifier() {
  if (!magnifierCenter) return;
  if (drag) {
    const editor = editorWithCorner(drag.index, drag.point);
    drawMagnifierForMainDrag(editor);
  } else if (magnifierDrag) {
    drawMagnifierDrag();
  } else {
    const index = state.editor.selected_corner >= 0 ? state.editor.selected_corner : 0;
    const point = state.editor.corners[index];
    const offsetX = magnifierIndex === index
      ? (magnifierCenter[0] - point[0]) * state.editor.zoom : 0;
    const offsetY = magnifierIndex === index
      ? (magnifierCenter[1] - point[1]) * state.editor.zoom : 0;
    drawMagnifier(offsetX, offsetY);
  }
}

function cancelMagnifierDrag() {
  if (!magnifierDrag) return;
  magnifierDrag = null;
  cancelMagnifierRefresh();
  if (magnifierFrame !== null) cancelAnimationFrame(magnifierFrame);
  magnifierFrame = null;
  elements.magnifier.classList.remove("is-dragging");
  drawActiveMagnifier();
}

elements.magnifier.addEventListener("pointerdown", (event) => {
  if (!magnifierImage || !actionAllowed("set_corner")) return;
  const [x, y] = magnifierPoint(event);
  const index = state.editor.selected_corner >= 0 ? state.editor.selected_corner : 0;
  magnifierDrag = { pointerId: event.pointerId, index, startX: x, startY: y, dx: 0, dy: 0 };
  elements.magnifier.setPointerCapture(event.pointerId);
  elements.magnifier.classList.add("is-dragging");
  event.preventDefault();
});

elements.magnifier.addEventListener("pointermove", (event) => {
  if (!magnifierDrag || magnifierDrag.pointerId !== event.pointerId) return;
  const [x, y] = magnifierPoint(event);
  magnifierDrag.dx = x - magnifierDrag.startX;
  magnifierDrag.dy = y - magnifierDrag.startY;
  if (magnifierFrame === null) magnifierFrame = requestAnimationFrame(drawMagnifierDrag);
  event.preventDefault();
});

async function finishMagnifierDrag(event) {
  if (!magnifierDrag || magnifierDrag.pointerId !== event.pointerId) return;
  const [x, y] = magnifierPoint(event);
  magnifierDrag.dx = x - magnifierDrag.startX;
  magnifierDrag.dy = y - magnifierDrag.startY;
  drawMagnifierDrag();
  const completed = magnifierDrag;
  magnifierDrag = null;
  cancelMagnifierRefresh();
  if (magnifierFrame !== null) cancelAnimationFrame(magnifierFrame);
  magnifierFrame = null;
  if (elements.magnifier.hasPointerCapture(event.pointerId)) {
    elements.magnifier.releasePointerCapture(event.pointerId);
  }
  elements.magnifier.classList.remove("is-dragging");
  event.preventDefault();
  const corner = state.editor.corners[completed.index];
  const dx = roundTiesToEven(completed.dx / state.editor.zoom);
  const dy = roundTiesToEven(completed.dy / state.editor.zoom);
  if (!dx && !dy) {
    drawActiveMagnifier();
    return;
  }
  await sendAction("set_corner", {
    index: completed.index,
    x: corner[0] - dx,
    y: corner[1] - dy,
  });
}

elements.magnifier.addEventListener("pointerup", (event) => void finishMagnifierDrag(event));
elements.magnifier.addEventListener("pointercancel", cancelMagnifierDrag);
elements.magnifier.addEventListener("lostpointercapture", cancelMagnifierDrag);

async function prefetchNext() {
  prefetchRequest?.abort();
  prefetchRequest = new AbortController();
  try {
    const response = await fetch(api("prefetch-next", viewportQuery()), {
      signal: prefetchRequest.signal,
    });
    if (!response.ok) return;
    prefetchedUrl = replaceObjectUrl(prefetchedUrl, URL.createObjectURL(await response.blob()));
  } catch (error) {
    if (error.name !== "AbortError") return;
  }
}

async function loadDetails() {
  const status = elements["details-status"];
  status.hidden = false;
  delete status.dataset.state;
  status.textContent = "正在加载技术详情…";
  try {
    const query = new URLSearchParams({ client_id: client });
    const response = await fetch(api("details", query));
    const details = await readResponse(response);
    const labels = {
      path: "完整路径",
      source_sha256: "Source SHA",
      detection_id: "Detection ID",
      previous_gui: "上次 GUI",
    };
    elements["details-list"].replaceChildren();
    for (const [key, labelText] of Object.entries(labels)) {
      const label = document.createElement("dt");
      const value = document.createElement("dd");
      label.textContent = labelText;
      value.textContent = details[key] ?? "—";
      elements["details-list"].append(label, value);
    }
    const audit = details.candidate_audit || [];
    elements["candidate-audit"].hidden = !audit.length;
    elements["candidate-audit"].textContent = audit.length ? JSON.stringify(audit, null, 2) : "";
    status.hidden = true;
    elements["details-list"].scrollIntoView({ block: "nearest" });
  } catch (error) {
    delete elements["technical-details"].dataset.loaded;
    status.dataset.state = "error";
    status.textContent = "技术详情加载失败，请重试。";
    throw error;
  }
}

async function acquireLease() {
  const lease = await postJson(api("lease"), { client_id: client });
  writer = Boolean(lease.writer);
}

async function loadState() {
  const query = new URLSearchParams({ client_id: client });
  const response = await fetch(api("state", query));
  renderState(await readResponse(response));
}

async function heartbeat() {
  const lease = await postJson(api("heartbeat"), { client_id: client });
  const nextWriter = Boolean(lease.writer);
  if (nextWriter !== writer) await loadState();
  writer = nextWriter;
  updateControls();
}

async function sendAction(kind, payload = {}) {
  if (actionPending || !actionAllowed(kind)) return;
  const previousToken = state.image.token;
  actionPending = true;
  updateControls();
  elements["action-error"].hidden = true;
  try {
    const result = await postJson(api("action"), {
      client_id: client,
      action_id: crypto.randomUUID(),
      expected_revision: state.revision,
      kind,
      payload,
    });
    clearError();
    renderState(result);
    if (result.session.status !== "active") return;
    if (result.image.token !== previousToken) {
      elements["technical-details"].open = false;
      delete elements["technical-details"].dataset.loaded;
      elements["details-list"].replaceChildren();
      elements["candidate-audit"].hidden = true;
      previewTransform = null;
      await Promise.all([loadPreview(), loadMagnifier()]);
      void prefetchNext();
    } else if (["select_corner", "move", "set_corner", "set_zoom", "candidate", "v52", "reset"].includes(kind)) {
      await loadMagnifier();
    }
  } catch (error) {
    if (error.name !== "AbortError") showError(error);
    if (state) {
      drawOverlay(state.editor, state.image);
      drawActiveMagnifier();
    }
  } finally {
    actionPending = false;
    updateControls();
  }
}

function pointerToOriginal(event) {
  if (!previewTransform) return null;
  const rect = elements["corner-overlay"].getBoundingClientRect();
  const width = previewTransform.original_width;
  const height = previewTransform.original_height;
  const scale = Math.min(rect.width / width, rect.height / height);
  const offsetX = (rect.width - width * scale) / 2;
  const offsetY = (rect.height - height * scale) / 2;
  return [
    Math.max(0, Math.min(width - 1, Math.round((event.clientX - rect.left - offsetX) / scale))),
    Math.max(0, Math.min(height - 1, Math.round((event.clientY - rect.top - offsetY) / scale))),
  ];
}

function drawDraggedCorner() {
  dragFrame = null;
  if (!drag) return;
  const editor = editorWithCorner(drag.index, drag.point);
  drawOverlay(editor, state.image);
  drawMagnifierForMainDrag(editor);
}

function drawMagnifierForMainDrag(editor) {
  if (!drag || !magnifierImage || !magnifierCenter) return;
  const point = editor.corners[drag.index];
  const dx = (magnifierCenter[0] - point[0]) * state.editor.zoom;
  const dy = (magnifierCenter[1] - point[1]) * state.editor.zoom;
  drawMagnifier(dx, dy, editor);
  scheduleMagnifierRefresh(point, dx, dy, drag.index);
}

elements["corner-overlay"].addEventListener("pointerdown", (event) => {
  const marker = event.target.closest(".corner-point");
  const index = Number(marker?.dataset.cornerIndex);
  if (!Number.isInteger(index) || !actionAllowed("set_corner")) return;
  const point = pointerToOriginal(event);
  if (!point) return;
  drag = {
    pointerId: event.pointerId,
    index,
    point,
  };
  elements["corner-overlay"].setPointerCapture(event.pointerId);
  event.preventDefault();
  drawDraggedCorner();
});

elements["corner-overlay"].addEventListener("pointermove", (event) => {
  if (!drag || drag.pointerId !== event.pointerId) return;
  drag.point = pointerToOriginal(event) || drag.point;
  if (dragFrame === null) dragFrame = requestAnimationFrame(drawDraggedCorner);
  event.preventDefault();
});

async function finishDrag(event) {
  if (!drag || drag.pointerId !== event.pointerId) return;
  drag.point = pointerToOriginal(event) || drag.point;
  drawDraggedCorner();
  const completed = drag;
  drag = null;
  cancelMagnifierRefresh();
  if (dragFrame !== null) cancelAnimationFrame(dragFrame);
  dragFrame = null;
  if (elements["corner-overlay"].hasPointerCapture(event.pointerId)) {
    elements["corner-overlay"].releasePointerCapture(event.pointerId);
  }
  event.preventDefault();
  await sendAction("set_corner", {
    index: completed.index,
    x: completed.point[0],
    y: completed.point[1],
  });
}

function cancelDrag(event) {
  if (!drag || drag.pointerId !== event.pointerId) return;
  drag = null;
  cancelMagnifierRefresh();
  if (dragFrame !== null) cancelAnimationFrame(dragFrame);
  dragFrame = null;
  drawOverlay(state.editor, state.image);
  drawActiveMagnifier();
}

elements["corner-overlay"].addEventListener("pointerup", (event) => void finishDrag(event));
elements["corner-overlay"].addEventListener("pointercancel", cancelDrag);
elements["corner-overlay"].addEventListener("lostpointercapture", cancelDrag);

const BUTTON_ACTIONS = {
  "previous-button": ["previous", {}],
  "next-button": ["next", {}],
  "candidate-button": ["candidate", {}],
  "v52-button": ["v52", {}],
  "reset-button": ["reset", {}],
  "skip-button": ["skip", { reason: "user_skip" }],
  "pause-button": ["pause", {}],
  "quit-button": ["quit", {}],
  "confirm-button": ["confirm", {}],
  "zoom-2-button": ["set_zoom", { zoom: 2 }],
  "zoom-4-button": ["set_zoom", { zoom: 4 }],
  "zoom-8-button": ["set_zoom", { zoom: 8 }],
};

for (const [id, [kind, payload]] of Object.entries(BUTTON_ACTIONS)) {
  elements[id].addEventListener("click", () => void sendAction(kind, payload));
}

const KEY_BINDINGS = {
  KeyC: ["candidate", {}],
  KeyB: ["v52", {}],
  KeyR: ["reset", {}],
  KeyX: ["skip", { reason: "user_skip" }],
  KeyP: ["pause", {}],
  KeyQ: ["quit", {}],
  KeyW: ["move", { dx: 0, dy: -50 }],
  KeyA: ["move", { dx: -50, dy: 0 }],
  KeyS: ["move", { dx: 0, dy: 50 }],
  KeyD: ["move", { dx: 50, dy: 0 }],
  ArrowLeft: ["move", { dx: -5, dy: 0 }],
  ArrowRight: ["move", { dx: 5, dy: 0 }],
  ArrowUp: ["move", { dx: 0, dy: -5 }],
  ArrowDown: ["move", { dx: 0, dy: 5 }],
  Digit1: ["select_corner", { index: 0 }],
  Digit2: ["select_corner", { index: 1 }],
  Digit3: ["select_corner", { index: 2 }],
  Digit4: ["select_corner", { index: 3 }],
  Enter: ["confirm", {}],
  Space: ["confirm", {}],
  Equal: ["adjust_zoom", { delta: 1 }],
  Minus: ["adjust_zoom", { delta: -1 }],
  NumpadAdd: ["adjust_zoom", { delta: 1 }],
  NumpadSubtract: ["adjust_zoom", { delta: -1 }],
};

document.addEventListener("keydown", (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey) return;
  if (document.activeElement !== document.body && !elements.workbench.contains(document.activeElement)) return;
  if (event.code === "Escape" && magnifierDrag) {
    event.preventDefault();
    cancelMagnifierDrag();
    return;
  }
  if (["Enter", "Space"].includes(event.code) && event.target.closest("button, summary")) return;
  let action = KEY_BINDINGS[event.code];
  if (!action) return;
  event.preventDefault();
  if (action[0] === "adjust_zoom") {
    const zooms = [2, 4, 8];
    const current = zooms.indexOf(state?.editor.zoom);
    const target = Math.max(0, Math.min(zooms.length - 1, current + action[1].delta));
    if (current < 0 || target === current) return;
    action = ["set_zoom", { zoom: zooms[target] }];
  }
  void sendAction(action[0], action[1]);
});

async function start() {
  try {
    await acquireLease();
    clearError();
    await loadState();
    heartbeatTimer = window.setInterval(() => void heartbeat().catch(showError), 5000);
  } catch (error) {
    if (error.name !== "AbortError") showError(error);
    elements.workbench.setAttribute("aria-busy", "false");
    return;
  }
  try {
    await Promise.all([loadPreview(), loadMagnifier()]);
    void prefetchNext();
  } catch (error) {
    if (error.name !== "AbortError") showError(error);
  } finally {
    ready = true;
    elements.workbench.setAttribute("aria-busy", "false");
    elements.workbench.focus({ preventScroll: true });
    updateControls();
  }
}

elements["technical-details"].addEventListener("toggle", () => {
  if (elements["technical-details"].open && !elements["technical-details"].dataset.loaded) {
    elements["technical-details"].dataset.loaded = "true";
    void loadDetails().catch(showError);
  }
});

window.addEventListener("pagehide", () => {
  window.clearInterval(heartbeatTimer);
  if (magnifierRefreshTimer !== null) window.clearTimeout(magnifierRefreshTimer);
  previewRequest?.abort();
  prefetchRequest?.abort();
  magnifierRequest?.abort();
  cancelMagnifierDrag();
  for (const url of [previewUrl, magnifierUrl, prefetchedUrl]) {
    if (url) URL.revokeObjectURL(url);
  }
});

void start();

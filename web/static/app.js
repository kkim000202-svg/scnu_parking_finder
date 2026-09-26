// 빈자리 주차 내비 — 카카오맵 화면
// 흐름: 목적지 검색 → 경로 + 목적지 주변 주차장 → (AI 주차장 선택) → 입구까지 경로 → 도착하면 빈칸 안내

const $ = (id) => document.getElementById(id);
const DEFAULT_CENTER = [34.9696, 127.4790]; // 국립순천대학교 부근

const state = {
  map: null,
  origin: null,   // {name, lat, lng}
  dest: null,     // {name, lat, lng, lotId?}
  route: null,
  overlays: { route: [], places: [], parking: [] },
  car: null,
  sim: null,
  activeField: null,
  parkingTimer: null,
};

// ---------- 공통 ----------

function toast(msg, ms = 3200) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), ms);
}

// 서버 주소: 화면을 서버와 따로 올릴 때(Genspark) config.js 의 window.API_BASE, 같은 곳이면 빈 값
const API_BASE = (window.API_BASE || "").replace(/\/$/, "");

async function api(path, tries = 4) {
  let r;
  try {
    r = await fetch(API_BASE + path);
  } catch (err) {
    r = null;  // 서버가 잠들어 있거나 네트워크 오류
  }
  // 서버가 준 오류(JSON)는 그대로 보여 주고, 서버가 아직 안 깨어 호스팅이 대신 준 오류(HTML)만 다시 시도
  const waking = !r || ([502, 503, 504].includes(r.status) && !(r.headers.get("content-type") || "").includes("json"));
  if (waking && tries > 1) {
    toast("서버를 깨우는 중이에요. 처음 한 번은 1분쯤 걸려요…", 6000);
    await new Promise((ok) => setTimeout(ok, 8000));
    return api(path, tries - 1);
  }
  if (!r) throw new Error("서버에 연결하지 못했어요. 잠시 후 다시 시도해 주세요.");
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof body.detail === "string" ? body.detail : `요청 실패 (${r.status})`);
  return body;
}

const fmtDist = (m) => (m >= 1000 ? `${(m / 1000).toFixed(1)}km` : `${Math.round(m)}m`);
const fmtTime = (s) => (s >= 3600 ? `${Math.floor(s / 3600)}시간 ${Math.round((s % 3600) / 60)}분` : `${Math.max(1, Math.round(s / 60))}분`);
const arriveAt = (s) => {
  const d = new Date(Date.now() + s * 1000);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")} 도착`;
};

function haversine(a, b) {
  const R = 6371000, toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(b[0] - a[0]), dLng = toRad(b[1] - a[1]);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(a[0])) * Math.cos(toRad(b[0])) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

const LL = (lat, lng) => new kakao.maps.LatLng(lat, lng);

function overlay(lat, lng, html, { yAnchor = 1, zIndex = 3, onClick } = {}) {
  const el = document.createElement("div");
  el.innerHTML = html;
  const node = el.firstElementChild;
  if (onClick) node.addEventListener("click", onClick);
  const o = new kakao.maps.CustomOverlay({ position: LL(lat, lng), content: node, yAnchor, zIndex });
  o.setMap(state.map);
  return o;
}

function clear(kind) {
  state.overlays[kind].forEach((o) => o.setMap(null));
  state.overlays[kind] = [];
}

// 지도를 target으로 옮기되, 아래 패널(폰)이나 왼쪽 패널(데스크톱)에 가리지 않는 자리로
function focusMap(lat, lng) {
  const sheet = $("sheet");
  sheet.classList.remove("collapsed");
  const map = state.map, proj = map.getProjection();
  map.setCenter(LL(lat, lng));
  const pt = proj.containerPointFromCoords(LL(lat, lng));
  const top = $("search").hidden ? 0 : $("search").getBoundingClientRect().bottom;
  let dx = 0, dy = 0;
  if (window.innerWidth < 900) {
    const bottom = window.innerHeight - Math.min(Math.max(sheet.offsetHeight, window.innerHeight * 0.45), window.innerHeight * 0.58);
    dy = Math.round(window.innerHeight / 2 - (top + bottom) / 2);  // 검색창과 패널 사이 한가운데로
  } else {
    dx = -Math.round((sheet.getBoundingClientRect().right + 12) / 2);
  }
  map.setCenter(proj.coordsFromContainerPoint(new kakao.maps.Point(pt.x + dx, pt.y + dy)));
}

const escapeHtml = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const ICON_PLAY = '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
const ICON_CAR = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M5 11l1.5-4.5A2 2 0 0 1 8.4 5h7.2a2 2 0 0 1 1.9 1.5L19 11M4 11h16a1 1 0 0 1 1 1v5h-2M3 17v-5a1 1 0 0 1 1-1M5 17h14"/><circle cx="7.5" cy="17" r="1.8" fill="currentColor"/><circle cx="16.5" cy="17" r="1.8" fill="currentColor"/></svg>';
// 회전 안내 아이콘 (안내 문구로 방향을 고른다)
function turnIcon(text) {
  const t = text || "";
  const arrow = (d) => `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">${d}</svg>`;
  if (/유턴/.test(t)) return arrow('<path d="M17 20V9a5 5 0 0 0-10 0v3"/><path d="M4 9l3 3 3-3"/>');
  if (/좌회전|왼쪽|11시|10시|9시/.test(t)) return arrow('<path d="M18 20V11a4 4 0 0 0-4-4H5"/><path d="M9 3L5 7l4 4"/>');
  if (/우회전|오른쪽|1시|2시|3시/.test(t)) return arrow('<path d="M6 20V11a4 4 0 0 1 4-4h9"/><path d="M15 3l4 4-4 4"/>');
  if (/직진|방면/.test(t)) return arrow('<path d="M12 20V5"/><path d="M6 11l6-6 6 6"/>');
  return arrow('<circle cx="12" cy="12" r="3" fill="currentColor" stroke="none"/>');
}

const ICON_NAV = '<svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l8 20-8-5-8 5z"/></svg>';

// ---------- 시작 ----------

function boot() {
  if (!window.KAKAO_JS_KEY || window.KAKAO_JS_KEY.startsWith("__")) {
    $("nokey").hidden = false;
    return;
  }
  const s = document.createElement("script");
  s.src = `https://dapi.kakao.com/v2/maps/sdk.js?appkey=${encodeURIComponent(window.KAKAO_JS_KEY)}&autoload=false`;
  s.onload = () => kakao.maps.load(start);
  s.onerror = () => {
    $("nokey").hidden = false;
    document.querySelector("#nokey h1").textContent = "지도를 불러오지 못했어요";
    document.querySelector("#nokey .nokey-card p").textContent = "잠시 후 다시 시도해 주세요.";
    document.querySelector("#nokey pre").hidden = true;
  };
  document.head.appendChild(s);
}

// 아래 패널: 손잡이를 끌어내리면 접히고, 끌어올리면 펼쳐진다 (짧게 누르면 전환)
function bindSheetDrag() {
  const sheet = $("sheet"), grip = $("grip");
  let startY = null, dy = 0, moved = false;
  grip.addEventListener("pointerdown", (e) => {
    startY = e.clientY; dy = 0; moved = false;
    grip.setPointerCapture(e.pointerId);
    sheet.style.transition = "none";
  });
  grip.addEventListener("pointermove", (e) => {
    if (startY === null) return;
    dy = e.clientY - startY;
    if (Math.abs(dy) > 6) moved = true;
    const collapsed = sheet.classList.contains("collapsed");
    // 접힌 상태에서는 위로만, 펼친 상태에서는 아래로만 따라 움직인다
    const shift = collapsed ? Math.min(0, dy) : Math.max(0, dy);
    sheet.style.transform = `translateY(${shift}px)`;
  });
  const end = () => {
    if (startY === null) return;
    startY = null;
    sheet.style.transition = "";
    sheet.style.transform = "";
    if (!moved) sheet.classList.toggle("collapsed");
    else if (dy > 30) sheet.classList.add("collapsed");
    else if (dy < -30) sheet.classList.remove("collapsed");
  };
  grip.addEventListener("pointerup", end);
  grip.addEventListener("pointercancel", end);
}

function start() {
  state.map = new kakao.maps.Map($("map"), { center: LL(...DEFAULT_CENTER), level: 4 });
  if (window.innerWidth >= 900) state.map.addControl(new kakao.maps.ZoomControl(), kakao.maps.ControlPosition.RIGHT); // 폰은 손가락 확대
  bindSheetDrag();
  $("quick-lot")?.addEventListener("click", () => {
    const p = state.parking?.ours?.[0];
    p ? showLot(p.id) : toast("근처에 AI 주차장이 없어요.");
  });
  $("quick-gps")?.addEventListener("click", () => locate(true));
  bindSearch();
  $("use-gps").addEventListener("click", () => locate(true));
  $("swap").addEventListener("click", swap);
  $("banner-stop").addEventListener("click", stopSim);
  locate(false);
  // 처음 화면에서도 근처 AI 주차장이 보이게
  loadParking({ lat: DEFAULT_CENTER[0], lng: DEFAULT_CENTER[1] }, false);
}

function locate(explicit) {
  if (!navigator.geolocation) {
    if (explicit) toast("이 브라우저는 위치 기능을 지원하지 않아요.");
    return;
  }
  navigator.geolocation.getCurrentPosition(
    (p) => {
      setOrigin({ name: "내 위치", lat: p.coords.latitude, lng: p.coords.longitude });
      if (explicit) state.map.panTo(LL(p.coords.latitude, p.coords.longitude));
    },
    () => explicit && toast("위치 권한이 없어요. 출발지를 직접 검색해 주세요."),
    { enableHighAccuracy: true, timeout: 8000 },
  );
}

// ---------- 검색 ----------

function bindSearch() {
  for (const id of ["origin", "dest"]) {
    const input = $(id);
    let timer;
    input.addEventListener("input", () => {
      clearTimeout(timer);
      state.activeField = id;
      const q = input.value.trim();
      if (!q) return ($("suggest").hidden = true);
      timer = setTimeout(() => suggest(q), 250);
    });
    input.addEventListener("focus", () => (state.activeField = id));
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        const first = $("suggest").querySelector("li");
        if (first) first.click();
      }
    });
  }
  document.addEventListener("click", (e) => {
    if (!$("search").contains(e.target)) $("suggest").hidden = true;
  });
}

async function suggest(q) {
  const c = state.map.getCenter();
  try {
    const items = await api(`/api/search?q=${encodeURIComponent(q)}&x=${c.getLng()}&y=${c.getLat()}`);
    const ul = $("suggest");
    ul.innerHTML = items.length ? "" : `<li><span class="addr">검색 결과가 없어요.</span></li>`;
    for (const it of items) {
      const li = document.createElement("li");
      li.innerHTML = `<div class="name">${escapeHtml(it.name)}</div><div class="addr">${escapeHtml(it.address)}${it.category ? " · " + escapeHtml(it.category) : ""}</div>`;
      li.addEventListener("click", () => choose({ name: it.name, lat: it.y, lng: it.x }));
      ul.appendChild(li);
    }
    ul.hidden = false;
  } catch (err) {
    toast(err.message);
  }
}

function choose(place) {
  $("suggest").hidden = true;
  if (state.activeField === "origin") setOrigin(place);
  else setDest(place);
}

function setOrigin(place) {
  state.origin = place;
  $("origin").value = place.name;
  if (state.dest) route();
}

function setDest(place) {
  if (!place.lotId) {
    const lot = (state.parking?.ours || []).find((o) => haversine([place.lat, place.lng], [o.lat, o.lng]) < 60);
    if (lot) place = { name: lot.name, lat: lot.entrance[0], lng: lot.entrance[1], lotId: lot.id, lot };
  }
  state.dest = place;
  $("dest").value = place.name;
  clear("places");
  state.overlays.places.push(overlay(place.lat, place.lng, `<div class="pin-end">도착</div>`, { zIndex: 5 }));
  state.map.setLevel(4);
  focusMap(place.lat, place.lng);
  if (!place.lotId) loadParking(place, true);
  if (state.origin) route();
  else sheetHint("출발지를 입력하거나 ◎ 버튼으로 내 위치를 쓰세요.");
}

function swap() {
  if (!state.origin || !state.dest) return;
  const o = state.origin;
  state.origin = { ...state.dest };
  delete state.origin.lotId;
  $("origin").value = state.origin.name;
  setDest(o);
}

// ---------- 길찾기 ----------

async function route() {
  stopSim();
  state.approach = null;
  try {
    const r = await api(`/api/route?ox=${state.origin.lng}&oy=${state.origin.lat}&dx=${state.dest.lng}&dy=${state.dest.lat}`);
    state.route = r;
    drawRoute(r);
    renderRouteSheet();
  } catch (err) {
    toast(err.message);
  }
}

function drawRoute(r) {
  clear("route");
  const path = r.path.map(([lat, lng]) => LL(lat, lng));
  const line = new kakao.maps.Polyline({ path, strokeWeight: 7, strokeColor: "#2F6FED", strokeOpacity: 0.9 });
  line.setMap(state.map);
  state.overlays.route.push(line);
  state.routeLine = line;
  state.overlays.route.push(overlay(state.origin.lat, state.origin.lng, `<div class="pin-start">출발</div>`, { zIndex: 5 }));
  state.guidePins = turnGuides(r).map((g, i) => overlay(g.lat, g.lng, `<div class="pin-guide">${i + 1}</div>`, { yAnchor: 0.5, zIndex: 4 }));
  state.overlays.route.push(...state.guidePins);
  // 목록 순서와 지도 번호를 맞추기 위해 목록에도 작은 번호를 같이 둔다 (turn 아이콘 옆)
  const bounds = new kakao.maps.LatLngBounds();
  path.forEach((p) => bounds.extend(p));
  state.map.setBounds(bounds, 170, 40, window.innerWidth < 900 ? 320 : 40, window.innerWidth < 900 ? 40 : 480);
}

const turnGuides = (r) => r.guides.slice(1, -1).slice(0, 30); // 출발·도착 안내는 빼고 회전 지점만

function renderRouteSheet() {
  const r = state.route, lot = state.dest.lot;
  const guides = turnGuides(r);
  let html = `
    <div class="summary"><span class="time">${fmtTime(r.duration)}</span><span class="dist">${fmtDist(r.distance)}</span>
      <span class="arrive">${arriveAt(r.duration)}</span></div>
    <div class="muted">${escapeHtml(state.origin.name)} → ${escapeHtml(state.dest.name)}</div>`;
  if (lot) {
    html += `<div class="callout">주차장 입구에 도착하면 <b>${escapeHtml(lot.recommended)}번 칸</b>까지 이어서 안내해요. (지금 빈칸 ${lot.empty}/${lot.total})</div>`;
  }
  html += `<div class="row"><button class="btn primary" id="sim">${ICON_PLAY} 모의 주행으로 안내 보기</button><button class="btn ghost" id="gps" title="폰 GPS를 따라 안내 (https 또는 localhost)">${ICON_CAR} 실제 주행</button></div>`;
  html += `<h3>경로 안내</h3><ul class="list">` +
    (guides.length ? guides.map((g, i) => `<li class="guide" data-i="${i}"><span class="turn">${turnIcon(g.text)}</span><div class="grow"><div class="name">${escapeHtml(g.text || "경로 안내")}</div>
      <div class="muted"><span class="mini-num">${i + 1}</span>${escapeHtml(g.name)}${g.distance ? " · " + fmtDist(g.distance) : ""}</div></div></li>`).join("")
      : `<li class="muted">회전 없이 직진하면 돼요.</li>`) + `</ul>`;
  html += `<div id="parking-list"></div>`;
  $("sheet-body").innerHTML = html;
  $("sim").addEventListener("click", startSim);
  $("gps").addEventListener("click", startGps);
  renderParkingList();
}

function sheetHint(text) {
  $("sheet-body").innerHTML = `<div class="hint"><b>${escapeHtml(text)}</b></div><div id="parking-list"></div>`;
  renderParkingList();
}

// ---------- 주변 주차장 ----------

async function loadParking(center, showList) {
  try {
    state.parking = await api(`/api/parking?x=${center.lng}&y=${center.lat}`);
  } catch (err) {
    state.parking = { ours: [], others: [] };
    if (showList) toast(err.message);
  }
  drawParking();
  if (showList) renderParkingList();
  clearInterval(state.parkingTimer);
  state.parkingTimer = setInterval(refreshOurs, 5000); // AI 주차장 빈칸 수는 5초마다 새로 고침 (카카오 검색은 안 부름)
}

async function refreshOurs() {
  if (!state.parking) return;
  try {
    const c = state.dest || { lat: DEFAULT_CENTER[0], lng: DEFAULT_CENTER[1] };
    const fresh = await api(`/api/parking?x=${c.lng}&y=${c.lat}&ours_only=true`);
    state.parking.ours = fresh.ours;
    drawParking();
    renderParkingList();
  } catch { /* 다음에 다시 시도 */ }
}

function drawParking() {
  clear("parking");
  const { ours = [], others = [] } = state.parking || {};
  for (const p of others) {
    state.overlays.parking.push(overlay(p.lat, p.lng,
      `<div style="display:flex;flex-direction:column;align-items:center"><div class="pin-p">P</div><div class="pin-label">${escapeHtml(p.name)}</div></div>`,
      { yAnchor: 0.3, zIndex: 2, onClick: () => showOther(p) }));
  }
  for (const p of ours) {
    const full = p.empty === 0;
    state.overlays.parking.push(overlay(p.lat, p.lng,
      `<div class="pin-ai${full ? " full" : ""}">${p.live ? '<span class="live-dot"></span>' : ""}P ${full ? "만차" : `빈칸 ${p.empty}/${p.total}`}</div>`,
      { zIndex: 6, onClick: () => showLot(p.id) }));
  }
}

function renderParkingList() {
  const box = $("parking-list");
  if (!box || !state.parking) return;
  const { ours = [], others = [] } = state.parking;
  if (!ours.length && !others.length) return (box.innerHTML = "");
  const from = state.dest ? [state.dest.lat, state.dest.lng] : DEFAULT_CENTER;
  const items = [
    ...ours.map((p) => ({ ...p, ai: true, d: haversine(from, [p.lat, p.lng]) })),
    ...others.map((p) => ({ ...p, ai: false, d: p.distance || haversine(from, [p.lat, p.lng]) })),
  ].sort((a, b) => (b.ai - a.ai) || a.d - b.d);
  box.innerHTML = `<h3>${state.dest ? "목적지 주변 주차장" : "주변 주차장"}</h3><ul class="list">` + items.slice(0, 12).map((p, i) => `
    <li class="click" data-i="${i}"><div class="grow"><div class="name">${escapeHtml(p.name)}</div>
      <div class="muted">${fmtDist(p.d)}${p.ai ? " · AI 카메라" : ""}</div></div>
      ${p.ai ? `<span class="badge ${p.empty ? "ai" : "full"}">${p.empty ? `빈칸 ${p.empty}/${p.total}` : "만차"}</span>` : `<span class="badge plain">정보 없음</span>`}
    </li>`).join("") + `</ul>`;
  box.querySelectorAll("li.click").forEach((li) => li.addEventListener("click", () => {
    const p = items[+li.dataset.i];
    p.ai ? showLot(p.id) : showOther(p);
  }));
}

async function showLot(id) {
  try {
    // 이 주차장으로 가는 경로가 있으면, 그 경로가 들어오는 쪽 입구 기준으로 안내
    let q = "";
    if (state.dest?.lotId === id && state.route) {
      const path = state.route.path, cum = [0];
      for (let i = 1; i < path.length; i++) cum.push(cum[i - 1] + haversine(path[i - 1], path[i]));
      const i = cum.findIndex((c) => c >= cum[cum.length - 1] - 40);
      q = "?" + approachQuery(path[Math.max(0, i)]).slice(1);
    }
    const lot = await api(`/api/lots/${id}${q}`);
    $("sheet-body").innerHTML = `
      <div class="lot-card">
        <div class="muted">${lot.live ? '<span class="badge full">● 실시간 카메라</span>' : "AI 카메라 주차장"}</div>
        <div class="name" style="font-weight:700;font-size:17px">${escapeHtml(lot.name)}</div>
        <div class="big">빈칸 ${lot.empty} / ${lot.total}</div>
        ${lot.recommended ? `<div class="callout">입구에서 가장 가까운 빈칸은 <b>${escapeHtml(lot.recommended)}번 칸</b>이에요. 노란 선이 입구에서 칸까지 가는 길이에요.</div>`
          : `<div class="callout" style="background:#FDE8E7;color:#8A1F19">지금은 빈칸이 없어요. 다른 주차장을 골라 주세요.</div>`}
        <img src="${lot.image}" alt="AI가 분석한 주차장 사진">
        <div class="muted">초록 = 빈칸 · 노란 테두리 = 추천 칸 · AI가 ${new Date(lot.updated * 1000).toLocaleTimeString("ko-KR")}에 확인</div>
        <div class="row stack">
          ${lot.recommended ? `<button class="btn primary" id="go-lot">${ICON_NAV} 빈칸까지 안내 시작</button>` : ""}
          ${lot.plan ? `<button class="btn ghost" id="show-plan">도면 보기</button>` : ""}
          <button class="btn ghost" id="back">경로로 돌아가기</button>
        </div>
      </div>`;
    const go = $("go-lot");
    if (go) go.addEventListener("click", () => {
      state.activeField = "dest";
      setDest({ name: lot.name, lat: lot.entrance[0], lng: lot.entrance[1], lotId: lot.id, lot });
    });
    if ($("show-plan")) $("show-plan").addEventListener("click", () => showPlan(lot));
    focusMap(lot.lat, lot.lng);
    $("back").addEventListener("click", () => (state.route && state.dest ? renderRouteSheet() : sheetHint("목적지를 검색해 보세요.")));
  } catch (err) {
    toast(err.message);
  }
}

function showOther(p) {
  $("sheet-body").innerHTML = `
    <div class="lot-card">
      <div class="name" style="font-weight:700;font-size:17px">${escapeHtml(p.name)}</div>
      <div class="muted">${escapeHtml(p.address || "")}</div>
      <div class="callout" style="background:#F2F4F7;color:#4A5465">이 주차장에는 AI 카메라가 없어서 빈칸 정보를 알 수 없어요.</div>
      <div class="row">
        <button class="btn primary" id="go-other">여기로 길안내</button>
        ${p.url ? `<a class="btn ghost" href="${escapeHtml(p.url)}" target="_blank" rel="noopener">카카오맵 정보</a>` : ""}
      </div>
    </div>`;
  focusMap(p.lat, p.lng);
  $("go-other").addEventListener("click", () => {
    state.activeField = "dest";
    setDest({ name: p.name, lat: p.lat, lng: p.lng });
  });
}

// ---------- 모의 주행 (발표·시연용) ----------

// ---------- 주행 안내 (모의 주행 · 실제 GPS 주행이 같은 코드를 쓴다) ----------

function prepareDrive() {
  const path = state.route.path;
  const cum = [0];
  for (let i = 1; i < path.length; i++) cum.push(cum[i - 1] + haversine(path[i - 1], path[i]));
  const guides = turnGuides(state.route).map((g) => ({ ...g, at: nearestIndex(path, [g.lat, g.lng]) }));
  state.drive = { path, cum, total: cum[cum.length - 1], guides, arrived: false };
  state.snap = null;
  $("search").hidden = true;
  $("banner").hidden = false;
  state.car = overlay(path[0][0], path[0][1], `<div class="car"></div>`, { yAnchor: 0.5, zIndex: 10 });
  state.map.setLevel(3);
}

// 경로 위 s(m) 지점까지 왔을 때 화면 갱신. 끝에 닿으면 arrive().
function progress(s, posOverride) {
  const d = state.drive;
  if (!d || d.arrived) return;
  s = Math.min(d.total, Math.max(0, s));
  let i = d.cum.findIndex((c) => c >= s);
  if (i < 1) i = 1;
  const k = (s - d.cum[i - 1]) / Math.max(1e-6, d.cum[i] - d.cum[i - 1]);
  const p = d.path;
  const pos = posOverride || [p[i - 1][0] + (p[i][0] - p[i - 1][0]) * k, p[i - 1][1] + (p[i][1] - p[i - 1][1]) * k];
  state.car.setPosition(LL(...pos));
  state.map.setCenter(LL(...pos));
  if (state.routeLine) state.routeLine.setPath([LL(...pos), ...p.slice(i).map((q) => LL(...q))]); // 지나간 길은 지운다
  const nextIdx = d.guides.findIndex((g) => g.at >= i);
  const next = nextIdx >= 0 ? d.guides[nextIdx] : null;
  const passed = nextIdx >= 0 ? nextIdx : d.guides.length;
  document.querySelectorAll("li.guide").forEach((li) => li.classList.toggle("passed", +li.dataset.i < passed));
  (state.guidePins || []).forEach((pin, idx) => pin.setMap(idx < passed ? null : state.map));
  const left = d.total - s;
  if (state.dest.lotId && left <= 10 && !state.snap) snapshotLot(state.dest.lotId);
  if (state.snap) {
    $("banner-dist").textContent = fmtDist(left);
    $("banner-text").textContent = state.snap.msg;
  } else if (next) {
    $("banner-dist").textContent = fmtDist(Math.max(0, d.cum[next.at] - s));
    $("banner-text").textContent = next.text || "안내";
  } else {
    $("banner-dist").textContent = fmtDist(left);
    $("banner-text").textContent = state.dest.lotId ? "주차장 입구까지 직진" : "목적지까지 직진";
  }
  if (s >= d.total) {
    d.arrived = true;
    arrive();
  }
}

function startSim() {
  if (!state.route) return;
  stopSim();
  prepareDrive();
  const duration = Math.min(40000, Math.max(12000, state.drive.total * 6)); // 실제 시간과 무관하게 12~40초 안에 재생
  const t0 = performance.now();
  const step = () => {
    const s = Math.min(1, (performance.now() - t0) / duration) * state.drive.total;
    progress(s);
    if (state.drive && !state.drive.arrived) state.sim = requestAnimationFrame(step);
  };
  state.sim = requestAnimationFrame(step);
}

// ---------- 실제 주행: 폰 GPS를 따라간다 ----------

const GPS_ARRIVE_M = 15;     // 목적지에서 이만큼 안이면 도착
const GPS_OFFROUTE_M = 60;   // 경로에서 이만큼 벗어난 상태가 이어지면 다시 길찾기

function startGps() {
  if (!state.route) return;
  if (!navigator.geolocation) return toast("이 브라우저는 위치 기능이 없어요.");
  if (!window.isSecureContext) return toast("실제 주행은 https 주소(배포 서버)나 localhost에서만 돼요.");
  stopSim();
  prepareDrive();
  state.gps = { offCount: 0, bad: 0 };
  $("banner-text").textContent = "GPS 신호를 기다리는 중…";
  $("banner-dist").textContent = "";
  state.gpsWatch = navigator.geolocation.watchPosition(
    (pos) => onGpsPosition(pos.coords.latitude, pos.coords.longitude, pos.coords.accuracy),
    (err) => {
      const msg = { 1: "위치 권한을 허용해 주세요.", 2: "위치를 잡지 못했어요. 하늘이 보이는 곳에서 다시 해 보세요.", 3: "위치 신호가 늦어요." }[err.code] || err.message;
      $("banner-text").textContent = msg;
    },
    { enableHighAccuracy: true, maximumAge: 1000, timeout: 15000 });
}

function onGpsPosition(lat, lng, accuracy = 10) {
  const d = state.drive;
  if (!d || d.arrived || !state.gps) return;
  const here = [lat, lng];
  if (accuracy > 100) {  // 너무 부정확한 위치(실내 등)는 건너뛴다
    $("banner-text").textContent = `GPS 정확도가 낮아요 (±${Math.round(accuracy)}m)`;
    return;
  }
  // 경로에서 가장 가까운 지점과 그 지점까지의 경로 거리
  let best = { dist: Infinity, s: 0 };
  const p = d.path;
  for (let i = 1; i < p.length; i++) {
    const { t, dist } = projectOnSegment(here, p[i - 1], p[i]);
    if (dist < best.dist) best = { dist, s: d.cum[i - 1] + (d.cum[i] - d.cum[i - 1]) * t };
  }
  const toDest = haversine(here, [state.dest.lat, state.dest.lng]);
  if (toDest <= GPS_ARRIVE_M || (best.dist <= GPS_OFFROUTE_M && d.total - best.s <= GPS_ARRIVE_M)) {  // 목적지 근처 또는 경로 끝
    progress(d.total, here);
    return;
  }
  if (best.dist > GPS_OFFROUTE_M) {
    state.gps.offCount++;
    $("banner-text").textContent = `경로에서 ${fmtDist(best.dist)} 벗어남 · 다시 찾는 중…`;
    if (state.gps.offCount >= 3) reroute(here);
    return;
  }
  state.gps.offCount = 0;
  progress(best.s, here);
}

function projectOnSegment(p, a, b) {
  // 짧은 거리라 위경도를 평면으로 보고 계산 (경도는 위도에 맞춰 줄임)
  const kx = Math.cos((a[0] * Math.PI) / 180);
  const ax = 0, ay = 0, bx = (b[1] - a[1]) * kx, by = b[0] - a[0], px = (p[1] - a[1]) * kx, py = p[0] - a[0];
  const len2 = bx * bx + by * by;
  const t = len2 ? Math.min(1, Math.max(0, (px * bx + py * by) / len2)) : 0;
  const q = [a[0] + by * t, a[1] + (bx * t) / kx];
  return { t, dist: haversine(p, q) };
}

async function reroute(here) {
  state.gps.offCount = 0;
  try {
    const r = await api(`/api/route?ox=${here[1]}&oy=${here[0]}&dx=${state.dest.lng}&dy=${state.dest.lat}`);
    state.route = r;
    state.origin = { name: "현재 위치", lat: here[0], lng: here[1] };
    $("origin").value = "현재 위치";
    drawRoute(r);
    renderRouteSheet();
    const car = state.car;
    prepareDrive();
    if (car) car.setMap(null);
    toast("경로를 다시 찾았어요.");
  } catch (err) {
    toast(err.message);
  }
}

function nearestIndex(path, p) {
  let best = 0, bd = Infinity;
  path.forEach((q, i) => {
    const d = haversine(p, q);
    if (d < bd) { bd = d; best = i; }
  });
  return best;
}

function stopSim() {
  if (state.sim) cancelAnimationFrame(state.sim);
  if (state.gpsWatch != null) navigator.geolocation.clearWatch(state.gpsWatch);
  state.gpsWatch = null;
  state.gps = null;
  state.drive = null;
  if (state.routeLine && state.route) state.routeLine.setPath(state.route.path.map((q) => LL(...q)));
  document.querySelectorAll("li.guide.passed").forEach((li) => li.classList.remove("passed"));
  (state.guidePins || []).forEach((pin) => pin.setMap(state.map));
  state.sim = null;
  if (state.car) state.car.setMap(null);
  state.car = null;
  $("banner").hidden = true;
  $("search").hidden = false;
}

// 주차장 10m 전: CCTV 최신 화면을 새로 분석해서 들어가기 전에 빈칸을 미리 알려 준다
// 차가 주차장으로 들어오는 방향: 경로 끝에서 약 40m 전 지점 (어느 입구로 들어오는지 고르는 데 쓴다)
function approachPoint() {
  const d = state.drive;
  if (!d) return state.approach || null;
  const i = d.cum.findIndex((c) => c >= d.total - 40);
  const p = d.path[Math.max(0, i)];
  state.approach = p;
  return p;
}
const approachQuery = (p) => (p ? `&alat=${p[0]}&alng=${p[1]}` : "");

function snapshotLot(id) {
  state.snap = { msg: "주차장 10m 전 · CCTV로 빈칸 확인 중…" };
  state.snap.promise = api(`/api/lots/${id}?fresh=true${approachQuery(approachPoint())}`).then((lot) => {
    state.snap.lot = lot;
    state.snap.msg = lot.recommended
      ? `CCTV 확인: 빈칸 ${lot.empty}/${lot.total} → ${lot.recommended}번 칸으로 안내할게요`
      : "CCTV 확인: 지금 만차예요";
    return lot;
  }).catch((err) => {
    state.snap.msg = "CCTV 확인 실패 · 입구에서 다시 확인할게요";
    throw err;
  });
}

async function arrive() {
  $("banner-dist").textContent = "도착";
  if (!state.dest.lotId) {
    $("banner-text").textContent = `${state.dest.name}에 도착했어요.`;
    setTimeout(stopSim, 2500);
    return;
  }
  // 주차장 입구 도착 → 지금 이 순간의 빈칸을 다시 확인해서 칸까지 안내
  $("banner-text").textContent = "주차장 입구 도착! 빈칸을 다시 확인하는 중…";
  try {
    const q = approachQuery(approachPoint());
    const lot = state.snap?.promise ? await state.snap.promise.catch(() => api(`/api/lots/${state.dest.lotId}?${q.slice(1)}`))
      : await api(`/api/lots/${state.dest.lotId}?fresh=true${q}`);
    stopSim();
    await showLot(lot.id);
    showPlan(lot);
  } catch (err) {
    toast(err.message);
    setTimeout(stopSim, 3000);
  }
}

// ---------- 주차장 도면: 입구 → 빈칸 (지나간 길은 사라짐) ----------

const SVG_NS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs, parent) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}
const ptsAttr = (pts) => pts.map((p) => p.join(",")).join(" ");

function showPlan(lot) {
  const plan = lot.plan;
  if (!plan) return toast("이 주차장은 도면 정보가 없어요.");
  $("plan").hidden = false;
  $("plan-meta").textContent = `${lot.name} · ${lot.live ? "실시간 카메라" : "CCTV"} ${new Date(lot.updated * 1000).toLocaleTimeString("ko-KR")} 촬영 · AI 판정 빈칸 ${lot.empty}/${lot.total}`;
  const view = (mode) => {
    $("view-cctv").classList.toggle("on", mode === "cctv");
    $("view-plan").classList.toggle("on", mode === "plan");
    drawPlan(plan, mode === "cctv" ? lot.photo : null);
    animatePlan(plan);
  };
  $("view-cctv").onclick = () => view("cctv");
  $("show-occ").checked = !!state.showOccupied;
  $("show-occ").onchange = () => {
    state.showOccupied = $("show-occ").checked;
    view($("view-cctv").classList.contains("on") ? "cctv" : "plan");
  };
  $("view-plan").onclick = () => view("plan");
  $("plan-close").onclick = closePlan;
  $("plan-replay").onclick = () => animatePlan(plan);
  view(lot.photo ? "cctv" : "plan");
}

function closePlan() {
  cancelAnimationFrame(state.planAnim);
  clearTimeout(state.planWait);
  $("plan").hidden = true;
}

function drawPlan(plan, photo) {
  const svg = $("plan-svg");
  svg.innerHTML = "";
  svg.classList.toggle("photo", !!photo);
  // 칸·통로·입구가 있는 부분만 보이게 화면 범위를 잡는다
  const all = [...plan.spaces.flatMap((s) => s.points), ...plan.aisles.flat(), plan.entrance];
  const xs = all.map((p) => Math.min(Math.max(p[0], 0), plan.width)), ys = all.map((p) => Math.min(Math.max(p[1], 0), plan.height));
  const pad = 40;
  // CCTV 화면이면 사진 전체, 도면이면 칸이 있는 부분만
  const x0 = photo ? 0 : Math.min(...xs) - pad, y0 = photo ? 0 : Math.min(...ys) - pad;
  const w = photo ? plan.width : Math.max(...xs) - x0 + pad, h = photo ? plan.height : Math.max(...ys) - y0 + pad;
  svg.setAttribute("viewBox", `${x0} ${y0} ${w} ${h}`);
  svg.style.aspectRatio = `${w} / ${h}`;
  const unit = Math.max(w, h) / 100; // 선 굵기·글자 크기 기준

  const clip = svgEl("clipPath", { id: "plan-clip" }, svgEl("defs", {}, svg));
  svgEl("rect", { x: 0, y: 0, width: plan.width, height: plan.height }, clip);
  const g = svgEl("g", { "clip-path": "url(#plan-clip)" }, svg);
  if (photo) svgEl("image", { href: photo, x: 0, y: 0, width: plan.width, height: plan.height, preserveAspectRatio: "none" }, g);
  else plan.aisles.forEach((a) => svgEl("polygon", { points: ptsAttr(a), fill: "#DDE2EA" }, g));
  for (const s of plan.spaces) {
    const best = s.id === plan.recommended;
    if (photo && s.occupied && !state.showOccupied) continue;  // CCTV 화면: 빈칸만 표시 (주차된 차는 사진에 그대로 보임)
    svgEl("polygon", {
      points: ptsAttr(s.points), "stroke-linejoin": "round", "stroke-width": best ? unit * 0.9 : unit * 0.45,
      fill: best ? "#FFE9A8" : s.occupied ? "#F6C9C6" : "#BFE8D2", "fill-opacity": photo ? (s.occupied ? 0.25 : 0.45) : 1,
      stroke: best ? "#E0A800" : s.occupied ? "#D6453D" : "#0E8A5F",
      class: best ? "plan-best" : "",
    }, g);
    const t = svgEl("text", {
      x: s.center[0], y: s.center[1], "text-anchor": "middle", "dominant-baseline": "central",
      "font-size": unit * 3.2, "font-weight": 800, fill: s.occupied ? "#8A1F19" : "#075A3D",
      stroke: "#fff", "stroke-width": unit * 0.5, "paint-order": "stroke",
    }, g);
    t.textContent = s.occupied && !photo ? `${s.id} 🚗` : s.id;
  }
  if (plan.route) {
    state.planRouteCasing = svgEl("polyline", { points: ptsAttr(plan.route), fill: "none", stroke: "#fff",
      "stroke-width": unit * 2.2, "stroke-linecap": "round", "stroke-linejoin": "round" }, g);
    state.planRoute = svgEl("polyline", { points: ptsAttr(plan.route), fill: "none", stroke: "#F2B705",
      "stroke-width": unit * 1.4, "stroke-linecap": "round", "stroke-linejoin": "round" }, g);
  }
  const [ex, ey] = plan.entrance;
  svgEl("circle", { cx: ex, cy: ey, r: unit * 2.2, fill: "#1C2330" }, svg);
  const et = svgEl("text", { x: ex, y: ey - unit * 3.4, "text-anchor": "middle", "font-size": unit * 3, "font-weight": 800,
    fill: "#1C2330", stroke: "#fff", "stroke-width": unit * 0.5, "paint-order": "stroke" }, svg);
  et.textContent = "입구";
  state.planCar = svgEl("circle", { cx: ex, cy: ey, r: unit * 2, fill: "#2F6FED", stroke: "#fff", "stroke-width": unit * 0.8 }, svg);
}

// 칸 짧은 변 ≈ 2.5m 로 보고 사진 픽셀을 대략 미터로 바꾼다 (원근 때문에 추천 칸 크기를 기준으로)
function metersPerPixel(plan) {
  const target = plan.spaces.filter((s) => s.id === plan.recommended);
  const sides = (target.length ? target : plan.spaces).map((s) => {
    const d = s.points.map((p, i) => Math.hypot(p[0] - s.points[(i + 1) % s.points.length][0], p[1] - s.points[(i + 1) % s.points.length][1]));
    return Math.min(...d);
  }).filter((v) => v > 0);
  return sides.length ? 2.5 / (sides.reduce((a, b) => a + b, 0) / sides.length) : 0.05;
}

function animatePlan(plan) {
  cancelAnimationFrame(state.planAnim);
  clearTimeout(state.planWait);
  if (!plan.route) {
    $("plan-dist").textContent = "만차";
    $("plan-text").textContent = "빈칸이 없어요. 지도로 돌아가 다른 주차장을 골라 주세요.";
    return;
  }
  const route = plan.route, cum = [0];
  for (let i = 1; i < route.length; i++) cum.push(cum[i - 1] + Math.hypot(route[i][0] - route[i - 1][0], route[i][1] - route[i - 1][1]));
  const total = cum[cum.length - 1], mpp = metersPerPixel(plan);
  const duration = 7000, t0 = performance.now();
  $("plan-dist").textContent = `${plan.recommended}번 칸`;
  const step = () => {
    const s = Math.min(1, (performance.now() - t0) / duration) * total;
    let i = cum.findIndex((c) => c >= s);
    if (i < 1) i = 1;
    const k = (s - cum[i - 1]) / Math.max(1e-6, cum[i] - cum[i - 1]);
    const pos = [route[i - 1][0] + (route[i][0] - route[i - 1][0]) * k, route[i - 1][1] + (route[i][1] - route[i - 1][1]) * k];
    const rest = ptsAttr([pos, ...route.slice(i)]); // 지나간 길은 지우고 남은 길만
    state.planRoute.setAttribute("points", rest);
    state.planRouteCasing.setAttribute("points", rest);
    state.planCar.setAttribute("cx", pos[0]);
    state.planCar.setAttribute("cy", pos[1]);
    if (s < total) {
      $("plan-text").textContent = `노란 길을 따라 ${plan.recommended}번 칸까지 약 ${Math.max(1, Math.round((total - s) * mpp))}m`;
      state.planAnim = requestAnimationFrame(step);
    } else {
      $("plan-dist").textContent = "도착";
      $("plan-text").textContent = `${plan.recommended}번 칸에 주차하세요.`;
      state.planRoute.setAttribute("points", "");
      state.planRouteCasing.setAttribute("points", "");
    }
  };
  $("plan-text").textContent = "주차장 입구에 도착했어요. 빈칸까지 안내할게요.";
  state.planWait = setTimeout(() => (state.planAnim = requestAnimationFrame(step)), 1200);
}

boot();

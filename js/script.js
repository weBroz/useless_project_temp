/**
 * ProctorAI — Productivity Enforcer v3.1
 * script.js
 *
 * Facial trigger tracking via MediaPipe FaceLandmarker.
 * Triggers:
 *  - Yawning (mouth open)  → sepia + shake + gangeee.mp3 + river bg
 *  - Looking Away          → red border flash + eda_mone.mp3 (loop)
 */

import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ───────────────────────────────────────────────────────────────────────────
// Constants — Landmark indices
// ───────────────────────────────────────────────────────────────────────────
const LM = {
  UPPER_LIP:   13,
  LOWER_LIP:   14,
  NOSE_TIP:     1,
  LEFT_CHEEK:  234,
  RIGHT_CHEEK: 454,
};

// Iris & eye-corner landmarks (indices 468-477 from the 478-point model)
const LM_EYE = {
  // Left eye corners
  L_OUTER:  33,
  L_INNER: 133,
  // Right eye corners
  R_INNER: 362,
  R_OUTER: 263,
  // Iris centres (landmark 468 = left, 473 = right)
  L_IRIS: 468,
  R_IRIS: 473,
  // Temple landmarks for anger-vein emoji placement
  L_TEMPLE: 109,
  R_TEMPLE: 338,
  // Forehead for sweat-drop
  FOREHEAD: 10,
};

// Thresholds (normalised 0-1, relative to face bounding box height/width)
const THRESH = {
  MOUTH_OPEN_RATIO: 0.06,   // lip gap / face height
  HEAD_TURN_RATIO:  0.30,   // nose-x offset from face midpoint / face width
};

// Hysteresis delay in ms (prevents rapid on/off flicker)
const DEBOUNCE_MS = 300;

// ───────────────────────────────────────────────────────────────────────────
// DOM refs
// ───────────────────────────────────────────────────────────────────────────
const video          = document.getElementById("webcam");
const canvas         = document.getElementById("landmark-canvas");
const ctx            = canvas.getContext("2d");
const startOverlay   = document.getElementById("start-overlay");
const btnStart       = document.getElementById("btn-start");
const statusDot      = document.getElementById("status-dot");
const statusLabel    = document.getElementById("status-label");
const fpsCounter     = document.getElementById("fps-counter");
const faceCounter    = document.getElementById("face-counter");
const sessionTimeEl  = document.getElementById("session-time");
const violationCount = document.getElementById("violation-count");
const complianceEl   = document.getElementById("compliance-score");
const complianceFill = document.getElementById("compliance-fill");
const currentStateEl = document.getElementById("current-state");
const mouthApertureEl= document.getElementById("mouth-aperture");
const headRotationEl = document.getElementById("head-rotation");
const logFeed        = document.getElementById("log-feed");
const banner         = document.getElementById("violation-banner");
const bannerTitle    = document.getElementById("violation-title");
const bannerSub      = document.getElementById("violation-sub");
const bannerIcon     = document.getElementById("violation-icon");
const badgeYawn      = document.getElementById("badge-yawn");
const badgeLook      = document.getElementById("badge-look");
const badgeWork      = document.getElementById("badge-work");
const badgeDizzy     = document.getElementById("badge-dizzy");
const badgeCursor    = document.getElementById("badge-cursor");
const badgeTilt      = document.getElementById("badge-tilt");
const badgePip       = document.getElementById("badge-pip");
const badgeTab       = document.getElementById("badge-tab");
const badgeGhost     = document.getElementById("badge-ghost");
const btnPipToggle   = document.getElementById("btn-pip-toggle");
const pipVideo       = document.getElementById("pip-video");
const resolutionLabel= document.getElementById("resolution-label");
const comicPopup     = document.getElementById("comic-popup");
const comicPopupInner= document.getElementById("comic-popup-inner");
const workAlertBanner= document.getElementById("work-alert-banner");
const calWrap        = document.getElementById("calibration-bar-wrap");
const calStatus      = document.getElementById("cal-status");
const calPercentage  = document.getElementById("cal-percentage");
const calPostureAngle= document.getElementById("cal-posture-angle");
const calFill        = document.getElementById("cal-fill");

// Audio objects — created in JS so paths are explicit and reloadable
// (HTML <audio> elements are kept as fallback but we don't use them for playback)
let audioGangeee      = null;
let audioEdaMone      = null;
let audioMosquito     = null; // plays during compliance ("you can't win")
let audioDashamoolam  = null; // plays on tab switch treason
let audioSecond       = null; // plays on tab switch treason
let audioGhost        = null; // plays on window close attempt
let audioKeyboard     = null; // plays on typing
let audioMouse        = null; // plays on mouse motion
let audioStay         = null; // plays if user stays after close prompt
let audioTilt         = null; // plays on head tilt >1.8°
let audioIntro        = new Audio("assets/intro.mp3"); // plays exactly once when website opens
audioIntro.volume     = 1.0;
audioIntro.preload    = "auto";
let introPlayed       = false;
let audioCtx          = null; // Web AudioContext — unlocked on first user gesture

// Tab Treason, PiP & Escape State
let tabPunishmentActive  = false;
let originalDocTitle     = document.title;
let titleBlinkInterval   = null;
let pipStreamActive      = false;
let lastGhostPlayTime    = 0;
let ghostAudioActive     = false;
let ghostLockActive      = false;
let ghostLockTimeout     = null;
let triedToClose         = false;
let stayTimer            = null;
let stayLockActive       = false;
let stayLockTimeout      = null;
let lastMouseAudioTime   = 0;

// Spatial Audio & Mechanics State
let mosquitoPanner        = null;
let mosquitoSourceNode    = null;
let mosquitoOrbit         = 0;
let lastMosquitoEvadeTime = 0;

let calProgress           = 58.42;
let calTiltAngle          = 0;
let lastTiltSpillTime     = 0;

// ───────────────────────────────────────────────────────────────────────────
// State
// ───────────────────────────────────────────────────────────────────────────
let faceLandmarker   = null;
let rafId            = null;
let sessionStart     = null;
let totalViolations  = 0;
let totalSeconds     = 0;
let violationSeconds = 0;

const state = {
  yawn:     { active: false, timer: null },
  lookAway: { active: false, timer: null },
  work:     { active: false, _timeout: null },
  dizzy:    { active: false, _timeout: null },
};

// FPS tracking
let frameCount = 0;
let lastFpsTime = performance.now();

// ───────────────────────────────────────────────────────────────────────────
// Logging
// ───────────────────────────────────────────────────────────────────────────
function log(msg, type = "info") {
  const el = document.createElement("div");
  el.className = `log-entry log-${type}`;
  const ts = new Date().toLocaleTimeString("en-US", { hour12: false });
  el.textContent = `[${ts}] ${msg}`;
  logFeed.prepend(el);
  // Keep log short
  while (logFeed.children.length > 20) {
    logFeed.removeChild(logFeed.lastChild);
  }
}

// ───────────────────────────────────────────────────────────────────────────
// Update Status Bar
// ───────────────────────────────────────────────────────────────────────────
function setStatus(label, dotClass) {
  statusLabel.textContent = label;
  statusDot.className = `status-dot ${dotClass}`;
}

// ───────────────────────────────────────────────────────────────────────────
// Session Clock
// ───────────────────────────────────────────────────────────────────────────
function startClock() {
  sessionStart = Date.now();
  setInterval(() => {
    const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
    const h = String(Math.floor(elapsed / 3600)).padStart(2, "0");
    const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, "0");
    const s = String(elapsed % 60).padStart(2, "0");
    sessionTimeEl.textContent = `${h}:${m}:${s}`;
    totalSeconds = elapsed;
    updateCompliance();
  }, 1000);
}

function updateCompliance() {
  if (totalSeconds === 0) return;
  const score = Math.max(0, Math.round(100 - (violationSeconds / totalSeconds) * 100));
  complianceEl.innerHTML = `${score}<span class="metric-unit">%</span>`;
  complianceFill.style.width = `${score}%`;
  if (score < 60) {
    complianceFill.style.background = "linear-gradient(90deg, #ff3355, #ff6600)";
  } else if (score < 80) {
    complianceFill.style.background = "linear-gradient(90deg, #ffaa00, #ffcc44)";
  } else {
    complianceFill.style.background = "linear-gradient(90deg, #00cc6a, #00ff88)";
  }
}

// ───────────────────────────────────────────────────────────────────────────
// MediaPipe Initialisation
// ───────────────────────────────────────────────────────────────────────────
async function initMediaPipe() {
  setStatus("LOADING MODEL", "");
  log("Loading MediaPipe WASM runtime...", "info");

  try {
    const filesetResolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU",
      },
      outputFaceBlendshapes: false,
      runningMode: "VIDEO",
      numFaces: 1,
    });

    log("FaceLandmarker loaded (GPU delegate)", "info");
    setStatus("READY", "");
    return true;
  } catch (err) {
    log(`Model load failed: ${err.message}`, "error");
    setStatus("MODEL ERROR", "error");
    return false;
  }
}

// ───────────────────────────────────────────────────────────────────────────
// Webcam Start
// ───────────────────────────────────────────────────────────────────────────
async function startWebcam() {
  log("Requesting webcam permission...", "info");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;

    await new Promise((res) => {
      video.onloadedmetadata = () => {
        video.play();
        res();
      };
    });

    // Size canvas to match video
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    resolutionLabel.textContent = `${video.videoWidth}×${video.videoHeight}`;
    log(`Webcam active: ${video.videoWidth}×${video.videoHeight}`, "info");
    return true;
  } catch (err) {
    log(`Webcam denied: ${err.message}`, "error");
    setStatus("CAM ERROR", "error");
    return false;
  }
}

// ───────────────────────────────────────────────────────────────────────────
// Audio System
// ───────────────────────────────────────────────────────────────────────────

/**
 * Call this SYNCHRONOUSLY inside a click handler (user gesture).
 * Creating an AudioContext + playing a silent buffer through it
 * permanently unlocks ALL audio on the page — even from timers / RAF —
 * regardless of whether the actual mp3 files have loaded yet.
 */
function unlockAudioContext() {
  try {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    // Play a 1-sample silent buffer to satisfy the user-gesture requirement
    const silentBuf = audioCtx.createBuffer(1, 1, 22050);
    const silentSrc = audioCtx.createBufferSource();
    silentSrc.buffer = silentBuf;
    silentSrc.connect(audioCtx.destination);
    silentSrc.start(0);
    audioCtx.resume(); // Chrome sometimes starts suspended
    log("AudioContext unlocked ✓", "info");
  } catch (e) {
    log("AudioContext failed: " + e.message, "warn");
  }
}

/**
 * Build fresh Audio objects so paths are explicit.
 * Called once after unlockAudioContext(), still inside the click handler.
 */
function initAudioObjects() {
  audioGangeee = new Audio("assets/gangeee.mp3");
  audioGangeee.volume = 1.0;
  audioGangeee.loop   = true;   // keep punishing until mouth closes
  audioGangeee.preload = "auto";

  audioEdaMone = new Audio("assets/eda_mone.mp3");
  audioEdaMone.volume = 1.0;
  audioEdaMone.loop   = true;
  audioEdaMone.preload = "auto";

  // Log load errors so the user can see them in the on-page log
  audioGangeee.addEventListener("error", () =>
    log("gangeee.mp3 not found — add it to assets/", "error"));
  audioEdaMone.addEventListener("error", () =>
    log("eda_mone.mp3 not found — add it to assets/", "error"));

  audioGangeee.addEventListener("canplaythrough", () =>
    log("gangeee.mp3 ready ✓", "info"));
  audioEdaMone.addEventListener("canplaythrough", () =>
    log("eda_mone.mp3 ready ✓", "info"));

  // Kick off preloading
  audioGangeee.load();
  audioEdaMone.load();

  // Mosquito — plays during compliance (no violation = high-pitch torture)
  audioMosquito = new Audio('assets/mosquito.mp3');
  audioMosquito.volume  = 0.55;
  audioMosquito.loop    = true;
  audioMosquito.preload = 'auto';
  audioMosquito.addEventListener('error', () =>
    log('mosquito.mp3 not found — add it to assets/', 'error'));
  audioMosquito.addEventListener('canplaythrough', () =>
    log('mosquito.mp3 ready ✓', 'info'));
  audioMosquito.load();

  // Dashamoolam1 & Second — blast simultaneously on tab switch treason
  audioDashamoolam = new Audio('assets/dashamoolam1.mp3');
  audioDashamoolam.volume  = 1.0;
  audioDashamoolam.loop    = true;
  audioDashamoolam.preload = 'auto';
  audioDashamoolam.addEventListener('error', () =>
    log('dashamoolam1.mp3 not found — add it to assets/', 'error'));
  audioDashamoolam.addEventListener('canplaythrough', () =>
    log('dashamoolam1.mp3 ready ✓', 'info'));
  audioDashamoolam.load();

  audioSecond = new Audio('assets/second.mp3');
  audioSecond.volume  = 1.0;
  audioSecond.loop    = true;
  audioSecond.preload = 'auto';
  audioSecond.addEventListener('error', () =>
    log('second.mp3 not found — add it to assets/', 'error'));
  audioSecond.addEventListener('canplaythrough', () =>
    log('second.mp3 ready ✓', 'info'));
  audioSecond.load();

  // Ghost — plays once on window close attempt
  audioGhost = new Audio('assets/ghost.mp3');
  audioGhost.volume  = 1.0;
  audioGhost.preload = 'auto';
  audioGhost.addEventListener('error', () =>
    log('ghost.mp3 not found — add it to assets/', 'error'));
  audioGhost.addEventListener('canplaythrough', () =>
    log('ghost.mp3 ready ✓', 'info'));
  audioGhost.load();

  // Keyboard — plays on keystroke
  audioKeyboard = new Audio('assets/keyboard.mp3');
  audioKeyboard.volume  = 1.0;
  audioKeyboard.preload = 'auto';
  audioKeyboard.addEventListener('error', () =>
    log('keyboard.mp3 not found — add it to assets/', 'error'));
  audioKeyboard.addEventListener('canplaythrough', () =>
    log('keyboard.mp3 ready ✓', 'info'));
  audioKeyboard.load();

  // Mouse — plays on moving mouse
  audioMouse = new Audio('assets/mouse.mp3');
  audioMouse.volume  = 1.0;
  audioMouse.preload = 'auto';
  audioMouse.addEventListener('error', () =>
    log('mouse.mp3 not found — add it to assets/', 'error'));
  audioMouse.addEventListener('canplaythrough', () =>
    log('mouse.mp3 ready ✓', 'info'));
  audioMouse.load();

  // Stay — plays when user stays after close prompt
  audioStay = new Audio('assets/stay.mp3');
  audioStay.volume  = 1.0;
  audioStay.preload = 'auto';
  audioStay.addEventListener('error', () =>
    log('stay.mp3 not found — add it to assets/', 'error'));
  audioStay.addEventListener('canplaythrough', () =>
    log('stay.mp3 ready ✓', 'info'));
  audioStay.load();

  // Tilt — plays on head tilt
  audioTilt = new Audio('assets/tilt.mp3');
  audioTilt.volume  = 1.0;
  audioTilt.preload = 'auto';
  audioTilt.addEventListener('error', () =>
    log('tilt.mp3 not found — add it to assets/', 'error'));
  audioTilt.addEventListener('canplaythrough', () =>
    log('tilt.mp3 ready ✓', 'info'));
  audioTilt.load();

  initSpatialMosquito();
}

/**
 * 3D Spatial Audio Panning for Mosquito:
 * Routes the real organic mosquito.mp3 through a StereoPannerNode.
 * Continuously orbits user's head, jumping away if user turns to face it.
 */
function initSpatialMosquito() {
  if (!audioCtx || !audioMosquito) return;
  try {
    mosquitoPanner = audioCtx.createStereoPanner();
    mosquitoPanner.connect(audioCtx.destination);

    if (!mosquitoSourceNode) {
      mosquitoSourceNode = audioCtx.createMediaElementSource(audioMosquito);
      mosquitoSourceNode.connect(mosquitoPanner);
      log("Real mosquito.mp3 linked to 3D Spatial Panner ✓", "info");
    }
  } catch (e) {
    console.warn("Spatial panner routing fallback (playing direct):", e);
  }
}

let currentPlayingAudio = null;
let headTiltActive       = false;

function getAllAudios() {
  return [
    audioIntro,
    audioGangeee,
    audioEdaMone,
    audioMosquito,
    audioDashamoolam,
    audioSecond,
    audioGhost,
    audioKeyboard,
    audioMouse,
    audioStay,
    audioTilt
  ].filter(Boolean);
}

/**
 * Resumes the current active continuous state audio:
 * 1. Tab Treason (if minimized) -> audioDashamoolam
 * 2. Head Tilt (if still tilted > 20°) -> audioTilt
 * 3. Yawn (if mouth open) -> audioGangeee
 * 4. Look Away (if looking away) -> audioEdaMone
 * 5. Baseline Compliance (if no violation) -> audioMosquito
 */
function resumeActiveStateAudio() {
  if (!audioCtx) return;

  if (tabPunishmentActive && audioDashamoolam) {
    playExclusiveAudio(audioDashamoolam, true);
    return;
  }
  if (headTiltActive && audioTilt) {
    playExclusiveAudio(audioTilt, false);
    return;
  }
  if (state.yawn.active && audioGangeee) {
    playExclusiveAudio(audioGangeee, false);
    return;
  }
  if (state.lookAway.active && audioEdaMone) {
    playExclusiveAudio(audioEdaMone, false);
    return;
  }

  // Baseline compliance — user is obeying
  if (!tabPunishmentActive && !headTiltActive && !state.yawn.active && !state.lookAway.active) {
    if (audioMosquito) {
      playExclusiveAudio(audioMosquito, true);
      document.body.classList.add('mosquito-mode');
    }
  }
}

/**
 * Exclusive Audio Manager (Strict Single-Audio + Last-In Priority):
 * - Exactly ONE audio is allowed to play at any given moment.
 * - When a new audio is triggered, it takes immediate priority (Last-In Priority),
 *   instantly pausing whatever was playing previously.
 * - When an audio finishes, resumeActiveStateAudio() checks if the trigger state
 *   is still active and loops on, or transitions back to baseline.
 */
function playExclusiveAudio(newAudio, loop = false) {
  if (!newAudio || !audioCtx) return;

  // STRICT PRIORITY LOCKS:
  // 1. GHOST HAS SUPREME PRIORITY:
  // Nothing (not Stay, mouse, keyboard, tilt) can cut off Ghost while active/playing.
  // (Only Tab Treason / switching away can interrupt it).
  if ((ghostLockActive || currentPlayingAudio === audioGhost) && newAudio !== audioGhost && newAudio !== audioDashamoolam) {
    return;
  }

  // 2. STAY HAS 2ND HIGHEST PRIORITY:
  // Stay cannot be interrupted by mouse, keyboard, tilt, or mosquito (only Ghost or Tab Treason).
  if (stayLockActive && newAudio !== audioStay && newAudio !== audioGhost && newAudio !== audioDashamoolam) {
    return;
  }

  // Mouse movements must NEVER interrupt or override stay.mp3 or ghost.mp3
  if ((currentPlayingAudio === audioStay || currentPlayingAudio === audioGhost) && newAudio === audioMouse) {
    return;
  }

  // 1. Pause and reset all other tracks
  getAllAudios().forEach((aud) => {
    if (aud !== newAudio && !aud.paused) {
      aud.pause();
      aud.currentTime = 0;
    }
  });

  // 2. Play the latest incoming audio (Last-In Priority)
  currentPlayingAudio = newAudio;
  newAudio.loop = loop;
  newAudio.currentTime = 0;
  newAudio.play().catch(() => {});

  // 3. When a sound finishes, automatically resume or loop active background state
  newAudio.onended = () => {
    if (currentPlayingAudio === newAudio) {
      currentPlayingAudio = null;
      resumeActiveStateAudio();
    }
  };
}

function safePlay(audio, loop = false) {
  playExclusiveAudio(audio, loop);
}

function safePause(audio) {
  if (!audio) return;
  audio.pause();
  audio.currentTime = 0;
  if (currentPlayingAudio === audio) {
    currentPlayingAudio = null;
  }
}

function activateTiltMode() {
  if (headTiltActive) return;
  headTiltActive = true;
  badgeTilt.setAttribute('data-active', 'true');
  if (audioTilt) {
    // If audioTilt is already playing its full run, let it complete
    if (currentPlayingAudio !== audioTilt || audioTilt.paused) {
      playExclusiveAudio(audioTilt, false);
    }
  } else {
    playSpillSound();
  }
}

function deactivateTiltMode() {
  if (!headTiltActive) return;
  headTiltActive = false;
  badgeTilt.setAttribute('data-active', 'false');
  // Allow audioTilt to complete its full playback run at least once!
  // It will not cut off abruptly; when it finishes, resumeActiveStateAudio()
  // will see headTiltActive is false and cleanly return to baseline compliance.
}

// ───────────────────────────────────────────────────────────────────────────
// Trigger: Yawn (Gangeee Mode)
// ───────────────────────────────────────────────────────────────────────────
function activateYawnMode() {
  if (state.yawn.active) return;
  state.yawn.active = true;
  totalViolations++;
  violationCount.textContent = totalViolations;

  document.body.classList.add("yawn-mode");
  badgeYawn.setAttribute("data-active", "true");

  // Loop gangeee while mouth is open
  audioGangeee.loop = true;
  safePlay(audioGangeee);

  // Comic popup slam
  showComicPopup('ഗംഗേേേ!', '#ff8800', 1800);

  showBanner("⚠", "GANGEEE MODE ACTIVATED", "Yawning detected — absolute disgrace");
  currentStateEl.textContent = "GANGEEE";
  currentStateEl.style.color = "var(--accent-amber)";

  log("TRIGGER: Yawn detected → Gangeee mode", "warn");

  // Count as violation time
  state.yawn._violationInterval = setInterval(() => { violationSeconds++; }, 1000);
}

function deactivateYawnMode() {
  if (!state.yawn.active) return;
  state.yawn.active = false;

  clearInterval(state.yawn._violationInterval);
  document.body.classList.remove("yawn-mode");
  badgeYawn.setAttribute("data-active", "false");
  safePause(audioGangeee);
  resumeActiveStateAudio();

  hideBannerIfNoTrigger();
  log("Yawn mode cleared", "info");
  updateStateLabel();
}

// ───────────────────────────────────────────────────────────────────────────
// Trigger: Look Away (Eda Mone Mode)
// ───────────────────────────────────────────────────────────────────────────
function activateLookAwayMode(direction) {
  if (state.lookAway.active) return;
  state.lookAway.active = true;
  totalViolations++;
  violationCount.textContent = totalViolations;

  document.body.classList.add("look-away-mode");
  badgeLook.setAttribute("data-active", "true");

  safePlay(audioEdaMone);

  // Comic popup for eda mone
  showComicPopup('ഏട മോനേ!', '#cc0022', 1600);

  showBanner("👀", "EDA MONE! FOCUS!", `Head turned ${direction} — looking for excuses?`);
  currentStateEl.textContent = "EDA MONE";
  currentStateEl.style.color = "var(--accent-red)";

  log(`TRIGGER: Looking ${direction} → Eda Mone mode`, "error");

  state.lookAway._violationInterval = setInterval(() => { violationSeconds++; }, 1000);
}

function deactivateLookAwayMode() {
  if (!state.lookAway.active) return;
  state.lookAway.active = false;

  clearInterval(state.lookAway._violationInterval);
  document.body.classList.remove("look-away-mode");
  badgeLook.setAttribute("data-active", "false");
  safePause(audioEdaMone);
  resumeActiveStateAudio();

  hideBannerIfNoTrigger();
  log("Look-away mode cleared", "info");
  updateStateLabel();
}

// ───────────────────────────────────────────────────────────────────────────
// Violation Banner
// ───────────────────────────────────────────────────────────────────────────
function showBanner(icon, title, sub) {
  bannerIcon.textContent  = icon;
  bannerTitle.textContent = title;
  bannerSub.textContent   = sub;
  banner.classList.add("visible");
}

function hideBannerIfNoTrigger() {
  if (!state.yawn.active && !state.lookAway.active) {
    banner.classList.remove("visible");
  }
}

function updateStateLabel() {
  if (!state.yawn.active && !state.lookAway.active) {
    currentStateEl.textContent = "MONITORING";
    currentStateEl.style.color = "var(--accent-green)";
  }
}

// ───────────────────────────────────────────────────────────────────────────
// Canvas Drawing — Face Mesh + Anime VFX
// ───────────────────────────────────────────────────────────────────────────

/**
 * Glowing red laser eyes drawn over both irises.
 * Uses ctx.shadowBlur for the glow effect.
 */
function drawLaserEyes(lms, w, h) {
  if (lms.length < 474) return; // need iris landmarks

  const irises = [LM_EYE.L_IRIS, LM_EYE.R_IRIS];
  for (const idx of irises) {
    const iris = lms[idx];
    const x = iris.x * w;
    const y = iris.y * h;

    // Outer hot anime glow
    ctx.save();
    ctx.shadowBlur  = 30;
    ctx.shadowColor = '#ff0044';
    ctx.fillStyle   = 'rgba(255,20,50,0.92)';
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fill();

    // Inner bright core
    ctx.shadowBlur  = 12;
    ctx.shadowColor = '#ffffff';
    ctx.fillStyle   = '#ffffff';
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Beam shooting to screen edge with jagged anime lightning
    ctx.save();
    ctx.globalAlpha  = 0.75;
    ctx.lineWidth    = 3.5;
    ctx.strokeStyle  = '#ff2255';
    ctx.shadowBlur   = 16;
    ctx.shadowColor  = '#ff0033';
    ctx.beginPath();
    ctx.moveTo(x, y);

    const targetX = idx === LM_EYE.L_IRIS ? 0 : w;
    const midX = (x + targetX) / 2;
    // Anime lightning jagged kink
    const kinkY = y + (Math.random() - 0.5) * 16;
    ctx.lineTo(midX, kinkY);
    ctx.lineTo(targetX, y);
    ctx.stroke();

    // Anime comic electric spark (⚡)
    if (Math.random() > 0.4) {
      ctx.font = '14px serif';
      ctx.fillText('⚡', midX, kinkY);
    }
    ctx.restore();
  }
}

/**
 * Draws manga speed lines radiating from the canvas centre.
 * Called inside drawMesh when look-away is active.
 */
function drawSpeedLines(w, h) {
  const cx = w / 2, cy = h / 2;
  const numLines = 38;
  const minR = Math.min(w, h) * 0.15;
  const maxR = Math.sqrt(w * w + h * h);

  ctx.save();
  for (let i = 0; i < numLines; i++) {
    const angle  = (i / numLines) * Math.PI * 2;
    const spread = (Math.PI * 2) / numLines * 0.35;
    const r0     = minR + Math.random() * 20;

    // White line
    ctx.globalAlpha  = 0.45 + Math.random() * 0.3;
    ctx.strokeStyle  = '#ffffff';
    ctx.lineWidth    = 1.5 + Math.random() * 4;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(angle) * r0,        cy + Math.sin(angle) * r0);
    ctx.lineTo(cx + Math.cos(angle) * maxR,      cy + Math.sin(angle) * maxR);
    ctx.stroke();

    // Shadow line for depth
    ctx.globalAlpha  = 0.25;
    ctx.strokeStyle  = '#000000';
    ctx.lineWidth    = 1;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(angle + spread) * r0,   cy + Math.sin(angle + spread) * r0);
    ctx.lineTo(cx + Math.cos(angle + spread) * maxR, cy + Math.sin(angle + spread) * maxR);
    ctx.stroke();
  }
  ctx.restore();
}

/**
 * Draws anime & cartoon VFX pinned to real face landmarks:
 *   - 💢 pulsing anger vein at temple (look-away mode)
 *   - 💧 sweat drop at forehead (yawn mode / head tilt)
 *   - 🌀 dizzy swirl eyes (head tilt > 20°)
 *   - ゴゴゴゴ floating menacing manga kanji on violations
 */
function drawAnimeVFX(lms, w, h) {
  const isLook = state.lookAway.active;
  const isYawn = state.yawn.active;
  const isTilt = headTiltActive;

  // 1. 💢 Anime Anger Vein (Look-Away Mode)
  if (isLook && lms.length > LM_EYE.R_TEMPLE) {
    const temple = lms[LM_EYE.R_TEMPLE];
    const emojiSize = Math.round(w * 0.08);
    ctx.save();
    ctx.font      = `${emojiSize}px serif`;
    ctx.textAlign = 'center';
    const pulse = 1 + 0.22 * Math.sin(Date.now() / 100);
    ctx.translate(temple.x * w + emojiSize * 0.4, temple.y * h - emojiSize * 0.3);
    ctx.scale(pulse, pulse);
    ctx.fillText('💢', 0, 0);
    ctx.restore();
  }

  // 2. 💧 Anime Sweat Drop (Yawn Mode)
  if (isYawn && lms.length > LM_EYE.FOREHEAD) {
    const forehead = lms[LM_EYE.FOREHEAD];
    const emojiSize = Math.round(w * 0.065);
    const slideY = ((Date.now() % 1000) / 1000) * emojiSize * 2.0;
    ctx.save();
    ctx.font      = `${emojiSize}px serif`;
    ctx.textAlign = 'center';
    ctx.fillText('💧', forehead.x * w + emojiSize, forehead.y * h + slideY);
    ctx.restore();
  }

  // 3. 🌀 Dizzy Spiral Eyes + Anime Sweat Drop on Head Tilt (>20°)
  if (isTilt && lms.length > 474) {
    const forehead = lms[LM_EYE.FOREHEAD];
    const emojiSize = Math.round(w * 0.075);
    const slideY = ((Date.now() % 800) / 800) * emojiSize * 2.2;
    ctx.save();
    ctx.font      = `${emojiSize}px serif`;
    ctx.textAlign = 'center';
    ctx.fillText('💧', forehead.x * w + (calTiltAngle > 0 ? emojiSize * 1.5 : -emojiSize * 1.5), forehead.y * h + slideY);
    ctx.restore();

    // Dizzy spiral eyes over both pupils
    const irises = [LM_EYE.L_IRIS, LM_EYE.R_IRIS];
    for (const idx of irises) {
      const iris = lms[idx];
      ctx.save();
      const spiralSize = Math.round(w * 0.05);
      ctx.font = `${spiralSize}px serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.translate(iris.x * w, iris.y * h);
      ctx.rotate((Date.now() / 140) % (Math.PI * 2));
      ctx.fillText('🌀', 0, 0);
      ctx.restore();
    }
  }

  // 4. Dramatic Anime Menacing Kanji ("ゴゴゴゴ") on any violation
  if (isLook || isYawn || isTilt) {
    const kanjiSize = Math.max(22, Math.round(w * 0.055));
    const now = Date.now();
    ctx.save();
    ctx.font = `900 ${kanjiSize}px "Arial Black", sans-serif`;
    ctx.fillStyle = '#ff2255';
    ctx.shadowColor = '#000000';
    ctx.shadowBlur = 6;
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#000000';

    const offsets = [
      { text: 'ゴ', x: w * 0.12, y: h * 0.35 + Math.sin(now / 200) * 8 },
      { text: 'ゴ', x: w * 0.16, y: h * 0.50 + Math.sin(now / 200 + 1) * 8 },
      { text: 'ゴ', x: w * 0.82, y: h * 0.32 + Math.sin(now / 200 + 2) * 8 },
      { text: 'ゴ', x: w * 0.86, y: h * 0.48 + Math.sin(now / 200 + 3) * 8 }
    ];

    for (const item of offsets) {
      ctx.strokeText(item.text, item.x, item.y);
      ctx.fillText(item.text, item.x, item.y);
    }
    ctx.restore();
  }
}

// MediaPipe face mesh connections (simplified subset for visual flair)
const FACE_CONNECTIONS = [
  // Jawline
  [10, 338],[338,297],[297,332],[332,284],[284,251],[251,389],[389,356],[356,454],
  [454,323],[323,361],[361,288],[288,397],[397,365],[365,379],[379,378],[378,377],
  [377,152],[152,148],[148,176],[176,149],[149,150],[150,136],[136,172],[172,58],
  [58,132],[132,93],[93,234],[234,127],[127,162],[162,21],[21,54],[54,103],[103,67],
  [67,109],[109,10],
  // Left eye
  [33,7],[7,163],[163,144],[144,145],[145,153],[153,154],[154,155],[155,133],
  [33,246],[246,161],[161,160],[160,159],[159,158],[158,157],[157,173],[173,133],
  // Right eye
  [362,382],[382,381],[381,380],[380,374],[374,373],[373,390],[390,249],[249,263],
  [362,398],[398,384],[384,385],[385,386],[386,387],[387,388],[388,466],[466,263],
  // Nose
  [168,5],[5,4],[4,1],[1,19],[19,94],[94,2],
  [98,97],[97,2],[2,326],[326,327],
  // Lips (outer)
  [61,185],[185,40],[40,39],[39,37],[37,0],[0,267],[267,269],[269,270],[270,409],
  [409,291],[291,375],[375,321],[321,405],[405,314],[314,17],[17,84],[84,181],
  [181,91],[91,146],[146,61],
  // Lips (inner)
  [78,191],[191,80],[80,81],[81,82],[82,13],[13,312],[312,311],[311,310],[310,415],
  [415,308],[308,324],[324,318],[318,402],[402,317],[317,14],[14,87],[87,178],
  [178,88],[88,95],[95,78],
];

function drawMesh(landmarks) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!landmarks || landmarks.length === 0) return;

  const lms = landmarks[0]; // first face
  const w = canvas.width;
  const h = canvas.height;

  const isYawn = state.yawn.active;
  const isLook = state.lookAway.active;

  // Choose mesh color
  let dotColor, lineColor;
  if (isYawn) {
    dotColor  = "rgba(255, 170, 0, 0.9)";
    lineColor = "rgba(255, 170, 0, 0.35)";
  } else if (isLook) {
    dotColor  = "rgba(255, 51, 85, 0.9)";
    lineColor = "rgba(255, 51, 85, 0.35)";
  } else {
    dotColor  = "rgba(0, 255, 136, 0.85)";
    lineColor = "rgba(0, 255, 136, 0.25)";
  }

  // Draw connections
  ctx.lineWidth = 1;
  ctx.strokeStyle = lineColor;
  for (const [a, b] of FACE_CONNECTIONS) {
    if (a >= lms.length || b >= lms.length) continue;
    ctx.beginPath();
    ctx.moveTo(lms[a].x * w, lms[a].y * h);
    ctx.lineTo(lms[b].x * w, lms[b].y * h);
    ctx.stroke();
  }

  // Draw landmark dots
  ctx.fillStyle = dotColor;
  for (const lm of lms) {
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, 1.5, 0, Math.PI * 2);
    ctx.fill();
  }

  // Highlight key points
  const keyPoints = [LM.UPPER_LIP, LM.LOWER_LIP, LM.NOSE_TIP, LM.LEFT_CHEEK, LM.RIGHT_CHEEK];
  ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
  ctx.strokeStyle = dotColor;
  ctx.lineWidth = 1.5;
  for (const idx of keyPoints) {
    if (idx >= lms.length) continue;
    const lm = lms[idx];
    ctx.beginPath();
    ctx.arc(lm.x * w, lm.y * h, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  // ── Anime VFX drawn on top of mesh ──
  if (isLook) drawSpeedLines(w, h); // radial manga lines on look-away
  drawLaserEyes(lms, w, h);         // always show laser eyes (eerie default look)
  drawAnimeVFX(lms, w, h);          // 💢 vein + 💧 sweat drop
} // end drawMesh

// ───────────────────────────────────────────────────────────────────────────
// Comic Popup System
// ───────────────────────────────────────────────────────────────────────────
let comicPopupTimeout = null;

/**
 * Slams a comic book word onto the screen.
 * @param {string} text  — e.g. "ഗംഗേേേ!"
 * @param {string} color — CSS colour for the starburst background
 * @param {number} duration — ms before it flies off
 */
function showComicPopup(text, color = '#ff2244', duration = 1400) {
  if (comicPopupTimeout) clearTimeout(comicPopupTimeout);

  comicPopupInner.textContent = text;
  comicPopup.style.setProperty('--comic-color', color);

  // Reset then trigger slam-in
  comicPopup.classList.remove('slam-in', 'slam-out');
  void comicPopup.offsetWidth; // force reflow
  comicPopup.classList.add('slam-in');

  comicPopupTimeout = setTimeout(() => {
    comicPopup.classList.replace('slam-in', 'slam-out');
  }, duration);
}

// ───────────────────────────────────────────────────────────────────────────
// Alarm Beep (Web Audio API — no file needed)
// ───────────────────────────────────────────────────────────────────────────
function playAlarmBeep() {
  if (!audioCtx || ghostAudioActive) return;
  const osc  = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.connect(gain);
  gain.connect(audioCtx.destination);
  osc.type      = 'square';
  osc.frequency.setValueAtTime(880, audioCtx.currentTime);
  osc.frequency.setValueAtTime(660, audioCtx.currentTime + 0.12);
  osc.frequency.setValueAtTime(880, audioCtx.currentTime + 0.24);
  gain.gain.setValueAtTime(0.4, audioCtx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.45);
  osc.start(audioCtx.currentTime);
  osc.stop(audioCtx.currentTime + 0.45);
}

// ───────────────────────────────────────────────────────────────────────────
// Trigger 3: Work Alert (Keystroke Trap + Amba Pop-Up)
// ───────────────────────────────────────────────────────────────────────────
const ambaPopup = document.getElementById('amba-popup');
let ambaPopupTimeout = null;

function showBigAmbaPopup() {
  const ambaEl = document.getElementById('amba-popup');
  if (!ambaEl) return;
  // If Ghost or Stay is active, do not overlap
  if (ghostLockActive || stayLockActive) return;

  ambaEl.classList.add('visible');
  clearTimeout(ambaPopupTimeout);
  ambaPopupTimeout = setTimeout(() => {
    ambaEl.classList.remove('visible');
  }, 2200);
}

function hideBigAmbaPopup() {
  const ambaEl = document.getElementById('amba-popup');
  if (!ambaEl) return;
  ambaEl.classList.remove('visible');
  clearTimeout(ambaPopupTimeout);
}

const WORK_MESSAGES = [
  'CRITICAL ERROR: PRODUCTIVE BEHAVIOUR DETECTED',
  'UNAUTHORIZED WORK IN PROGRESS — CEASE IMMEDIATELY',
  'ALERT: KEYBOARD ACTIVITY LOGGED',
  'WARNING: FOCUS DETECTED. UNACCEPTABLE.',
  'SYSTEM OVERRIDE: STOP DOING THINGS',
];

function activateWorkAlert() {
  if (!workAlertBanner) return;
  // Rotate messages
  const msg = WORK_MESSAGES[totalViolations % WORK_MESSAGES.length];
  document.getElementById('work-alert-title').textContent = msg;

  document.body.classList.add('work-alert-mode');
  workAlertBanner.classList.add('visible');
  if (badgeWork) badgeWork.setAttribute('data-active', 'true');

  showBigAmbaPopup();

  if (audioKeyboard && !ghostAudioActive && !ghostLockActive && !stayLockActive) {
    if (currentPlayingAudio !== audioKeyboard || audioKeyboard.paused) {
      playExclusiveAudio(audioKeyboard, false);
    }
  } else if (!ghostLockActive && !stayLockActive) {
    playAlarmBeep();
  }

  totalViolations++;
  violationCount.textContent = totalViolations;
  log('TRIGGER: Keystroke → Work alert mode + Amba popup', 'error');

  // Auto-dismiss after 3 seconds
  clearTimeout(state.work._timeout);
  state.work._timeout = setTimeout(deactivateWorkAlert, 3000);
}

function deactivateWorkAlert() {
  document.body.classList.remove('work-alert-mode');
  workAlertBanner.classList.remove('visible');
  if (badgeWork) badgeWork.setAttribute('data-active', 'false');
  hideBigAmbaPopup();
}

// Global keydown trap — any printable key fires work alert + Amba popup
window.addEventListener('keydown', (e) => {
  // Allow browser shortcuts & F-keys so user can still open DevTools
  if (e.key.length > 1 && !['Enter', 'Tab', 'Backspace', 'Delete', 'Space'].includes(e.key)) return;
  if (!audioCtx) return; // only active after Start Monitoring
  activateWorkAlert();
});

// Clickable trigger to test Amba easily
if (badgeWork) {
  badgeWork.style.cursor = 'pointer';
  badgeWork.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!audioCtx) unlockAudioContext();
    activateWorkAlert();
  });
}

const ruleWorkCard = document.querySelector('.rule-work');
if (ruleWorkCard) {
  ruleWorkCard.style.cursor = 'pointer';
  ruleWorkCard.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!audioCtx) unlockAudioContext();
    activateWorkAlert();
  });
}

// ───────────────────────────────────────────────────────────────────────────
// Trigger 4: Dizzy Mode (Rapid Mouse Jiggle)
// ───────────────────────────────────────────────────────────────────────────
const MOUSE_SPEED_THRESHOLD = 1.5; // px/ms
const DIZZY_LINGER_MS = 1500;      // stays dizzy this long after mouse stops

let lastMouseX = 0, lastMouseY = 0, lastMouseTime = 0;

function activateDizzyMode() {
  if (!state.dizzy.active) {
    state.dizzy.active = true;
    document.body.classList.add('dizzy-mode');
    badgeDizzy.setAttribute('data-active', 'true');
    log('TRIGGER: Mouse jiggle → Dizzy mode', 'warn');
  }
  // Reset linger timer on every fast movement
  clearTimeout(state.dizzy._timeout);
  state.dizzy._timeout = setTimeout(deactivateDizzyMode, DIZZY_LINGER_MS);
}

function deactivateDizzyMode() {
  state.dizzy.active = false;
  document.body.classList.remove('dizzy-mode');
  badgeDizzy.setAttribute('data-active', 'false');
}

function playMouseAudio() {
  if (!audioCtx || !audioMouse) return;

  // High priority lock: ghost and stay must NEVER be cut off by mouse movements
  if (ghostLockActive || stayLockActive || currentPlayingAudio === audioStay || currentPlayingAudio === audioGhost) {
    return;
  }

  // When cursor moves, the audio must complete in full at least once before re-triggering
  if (!audioMouse.paused && currentPlayingAudio === audioMouse) {
    return;
  }
  playExclusiveAudio(audioMouse, false);
}

window.addEventListener('mousemove', (e) => {
  if (!audioCtx) return; // only active after Start Monitoring

  // If user previously attempted to close window, mouse movement confirms they stayed!
  if (triedToClose) {
    handleUserStayed();
    return;
  }

  // If Ghost or Stay is playing, cursor movement must NOT play mouse sound or FaFa popup
  if (ghostLockActive || stayLockActive || currentPlayingAudio === audioGhost || currentPlayingAudio === audioStay) {
    return;
  }

  playMouseAudio();
  showBigFafaPopup();

  const now = performance.now();
  const dt  = now - lastMouseTime;
  if (dt > 0 && dt < 100) { // ignore stale readings
    const dx = e.clientX - lastMouseX;
    const dy = e.clientY - lastMouseY;
    const speed = Math.sqrt(dx * dx + dy * dy) / dt;

    if (speed > MOUSE_SPEED_THRESHOLD) activateDizzyMode();
  }
  lastMouseX = e.clientX;
  lastMouseY = e.clientY;
  lastMouseTime = now;
});

// ───────────────────────────────────────────────────────────────────────────
// Trigger 5: Big FaFa Pop-Up (Cursor Movement)
// ───────────────────────────────────────────────────────────────────────────
const fafaPopup = document.getElementById('fafa-popup');
let fafaPopupTimeout = null;

function showBigFafaPopup() {
  if (!fafaPopup) return;
  // Ghost and Stay have priority: never show FaFa over Fath ghost
  if (ghostLockActive || stayLockActive || currentPlayingAudio === audioGhost || currentPlayingAudio === audioStay) {
    return;
  }

  badgeCursor.setAttribute('data-active', 'true');
  fafaPopup.classList.add('visible');

  clearTimeout(fafaPopupTimeout);
  fafaPopupTimeout = setTimeout(() => {
    fafaPopup.classList.remove('visible');
    badgeCursor.setAttribute('data-active', 'false');
  }, 1200);
}

/**
 * Synthesised tyre-screech: bandpass-filtered white noise burst.
 * No audio file needed.
 */
function playTireScreech() {
  if (!audioCtx || ghostAudioActive) return;
  const sr         = audioCtx.sampleRate;
  const dur        = 0.22;
  const frameCount = Math.floor(sr * dur);
  const buf        = audioCtx.createBuffer(1, frameCount, sr);
  const data       = buf.getChannelData(0);

  for (let i = 0; i < frameCount; i++) {
    // White noise with exponential envelope
    data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / frameCount, 0.55);
  }

  const src  = audioCtx.createBufferSource();
  src.buffer = buf;

  // Band-pass around 1.8 kHz = rubbery tyre quality
  const bpf       = audioCtx.createBiquadFilter();
  bpf.type        = 'bandpass';
  bpf.frequency.value = 1800;
  bpf.Q.value         = 1.4;

  const gain = audioCtx.createGain();
  gain.gain.setValueAtTime(0.5, audioCtx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + dur);

  src.connect(bpf);
  bpf.connect(gain);
  gain.connect(audioCtx.destination);
  src.start(audioCtx.currentTime);
  src.stop(audioCtx.currentTime + dur);
}

// ───────────────────────────────────────────────────────────────────────────
// Compliance Monitor: 3D Spatial Mosquito Audio
// When zero face violations are active — you’re behaving — the mosquito orbits.
// The moment you twitch, gangeee or eda mone blows your eardrums.
// ───────────────────────────────────────────────────────────────────────────
let complianceAudioActive = false;

function checkComplianceAudio() {
  if (!audioCtx) return;
  const faceViolation = state.yawn.active || state.lookAway.active || headTiltActive;
  const anyViolation = faceViolation || tabPunishmentActive;

  if (!anyViolation) {
    // Only engage mosquito if no other violation audio is currently playing
    if (!currentPlayingAudio || currentPlayingAudio === audioMosquito) {
      if (!complianceAudioActive) {
        complianceAudioActive = true;
        playExclusiveAudio(audioMosquito, true);
        document.body.classList.add('mosquito-mode');
      }
    }
    // Orbit the real mosquito recording smoothly around ears
    mosquitoOrbit += 0.035;
    if (mosquitoPanner) {
      const panVal = Math.sin(mosquitoOrbit);
      mosquitoPanner.pan.setValueAtTime(panVal, audioCtx.currentTime);
    }
  } else {
    if (complianceAudioActive || currentPlayingAudio === audioMosquito) {
      complianceAudioActive = false;
      safePause(audioMosquito);
      document.body.classList.remove('mosquito-mode');
    }
  }
}
function analyseLandmarks(landmarkResults) {
  if (!landmarkResults || landmarkResults.length === 0) {
    faceCounter.textContent = "0";
    mouthApertureEl.innerHTML = `--<span class="metric-unit">px</span>`;
    headRotationEl.innerHTML = `0.00<span class="metric-unit">°</span>`;

    // No face → deactivate all (with debounce)
    scheduleDeactivate("yawn");
    scheduleDeactivate("lookAway");
    return;
  }

  faceCounter.textContent = landmarkResults.length;

  const lms = landmarkResults[0]; // use first face

  // ---- Mouth Open (Yawn) Detection ----
  const upperLip = lms[LM.UPPER_LIP];
  const lowerLip = lms[LM.LOWER_LIP];
  const lipDist  = Math.abs(lowerLip.y - upperLip.y); // normalised 0-1

  // Get face height estimate (from top-ish to chin)
  const foreheadPt = lms[10];
  const chinPt     = lms[152];
  const faceHeight = Math.abs(chinPt.y - foreheadPt.y) || 0.001;

  const mouthRatio = lipDist / faceHeight;

  // Pixel-ish value for display (multiply by canvas height)
  const mouthPx = Math.round(lipDist * canvas.height);
  mouthApertureEl.innerHTML = `${mouthPx}<span class="metric-unit">px</span>`;

  if (mouthRatio > THRESH.MOUTH_OPEN_RATIO) {
    scheduleActivate("yawn");
    cancelDeactivate("yawn");
  } else {
    cancelActivate("yawn");
    scheduleDeactivate("yawn");
  }

  // ---- Head Turn (Look Away) Detection ----
  const noseTip    = lms[LM.NOSE_TIP];
  const leftCheek  = lms[LM.LEFT_CHEEK];
  const rightCheek = lms[LM.RIGHT_CHEEK];

  const faceMidX  = (leftCheek.x + rightCheek.x) / 2;
  const faceWidth = Math.abs(rightCheek.x - leftCheek.x) || 0.001;

  // Offset of nose from face centre, normalised by face half-width
  const noseOffset = (noseTip.x - faceMidX) / (faceWidth / 2);
  const absOffset  = Math.abs(noseOffset);

  // Convert to rough degrees (≈ linear up to ~45°)
  const estimatedDegrees = absOffset * 45;
  headRotationEl.innerHTML = `${estimatedDegrees.toFixed(1)}<span class="metric-unit">°</span>`;

  // Note: video is mirrored, so left/right are swapped visually
  const direction = noseOffset > 0 ? "LEFT" : "RIGHT";

  if (absOffset > THRESH.HEAD_TURN_RATIO) {
    scheduleActivate("lookAway", direction);
    cancelDeactivate("lookAway");
  } else {
    cancelActivate("lookAway");
    scheduleDeactivate("lookAway");
  }

  // ── 1. Head Tilt (Roll) for Gravity Progress Bar ──
  const dxRoll = (lms[LM_EYE.R_OUTER].x - lms[LM_EYE.L_OUTER].x);
  const dyRoll = (lms[LM_EYE.R_OUTER].y - lms[LM_EYE.L_OUTER].y);
  calTiltAngle = Math.atan2(dyRoll, dxRoll) * (180 / Math.PI);


  // ── 3. Mosquito Spatial Audio Evade Check ──
  if (mosquitoPanner) {
    const currentPan = mosquitoPanner.pan.value;
    const now = performance.now();
    // If mosquito is on the right and user turns right to face it:
    if (noseOffset > 0.12 && currentPan > 0.15 && now - lastMosquitoEvadeTime > 750) {
      mosquitoOrbit = Math.PI - mosquitoOrbit; // flip behind left ear
      lastMosquitoEvadeTime = now;
      log('🦟 Mosquito evaded right head-turn → jumped behind left ear!', 'info');
    } else if (noseOffset < -0.12 && currentPan < -0.15 && now - lastMosquitoEvadeTime > 750) {
      mosquitoOrbit = -mosquitoOrbit; // flip behind right ear
      lastMosquitoEvadeTime = now;
      log('🦟 Mosquito evaded left head-turn → jumped behind right ear!', 'info');
    }
  }

  // Update compliance audio every frame based on current violation state
  checkComplianceAudio();
}

// ───────────────────────────────────────────────────────────────────────────
// Debounced Activate / Deactivate helpers
// ───────────────────────────────────────────────────────────────────────────
const activateTimers   = {};
const deactivateTimers = {};

function scheduleActivate(trigger, ...args) {
  if (activateTimers[trigger]) return; // already scheduled
  activateTimers[trigger] = setTimeout(() => {
    delete activateTimers[trigger];
    if (trigger === "yawn")     activateYawnMode();
    if (trigger === "lookAway") activateLookAwayMode(...args);
  }, DEBOUNCE_MS);
}

function cancelActivate(trigger) {
  if (activateTimers[trigger]) {
    clearTimeout(activateTimers[trigger]);
    delete activateTimers[trigger];
  }
}

function scheduleDeactivate(trigger) {
  if (deactivateTimers[trigger]) return;
  deactivateTimers[trigger] = setTimeout(() => {
    delete deactivateTimers[trigger];
    if (trigger === "yawn")     deactivateYawnMode();
    if (trigger === "lookAway") deactivateLookAwayMode();
  }, DEBOUNCE_MS);
}

function cancelDeactivate(trigger) {
  if (deactivateTimers[trigger]) {
    clearTimeout(deactivateTimers[trigger]);
    delete deactivateTimers[trigger];
  }
}

// ───────────────────────────────────────────────────────────────────────────
// Main Detection Loop
// ───────────────────────────────────────────────────────────────────────────
let lastVideoTime = -1;

function detectionLoop() {
  rafId = requestAnimationFrame(detectionLoop);

  if (video.readyState < 2) return; // not ready yet

  const nowMs = performance.now();

  // FPS calc
  frameCount++;
  const elapsed = nowMs - lastFpsTime;
  if (elapsed >= 1000) {
    fpsCounter.textContent = Math.round((frameCount / elapsed) * 1000);
    frameCount = 0;
    lastFpsTime = nowMs;
  }

  // Only process if video frame changed
  if (video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  let results;
  try {
    results = faceLandmarker.detectForVideo(video, nowMs);
  } catch (e) {
    return; // skip frame on error
  }

  // Draw mesh
  drawMesh(results.faceLandmarks);

  // Analyse landmarks and trigger effects
  analyseLandmarks(results.faceLandmarks);

  // ── 4. Physics Progress Bar Physics Engine ──
  updateCalibrationPhysics(nowMs);
}

/**
 * Updates neural calibration bar with head-tilt gravity physics.
 * Liquid drops rapidly when tilted >20.0°, and asymptotically caps at 99.48%.
 */
function updateCalibrationPhysics(nowMs) {
  if (!calWrap) return;
  const isTilted = Math.abs(calTiltAngle) > 20.0;

  if (isTilted) {
    calWrap.classList.add('spilling');
    calProgress = Math.max(18.2, calProgress - 1.2);
    calStatus.textContent = 'SPILLING!';
    calStatus.style.color = 'var(--accent-red)';
    activateTiltMode();
  } else {
    calWrap.classList.remove('spilling');
    calStatus.textContent = 'CALIBRATING';
    calStatus.style.color = 'var(--accent-green)';
    deactivateTiltMode();

    // Slow, cruel climb
    if (calProgress < 94.0) {
      calProgress += 0.038;
    } else if (calProgress < 98.8) {
      calProgress += 0.007;
    } else if (calProgress < 99.48) {
      calProgress += 0.001;
    } else {
      // Oscillates near 99.49% — NEVER 100%!
      calProgress = 99.48 + Math.sin(nowMs / 250) * 0.008;
    }
  }

  calPercentage.textContent = calProgress.toFixed(2) + '%';
  calFill.style.width = calProgress + '%';
  calFill.style.setProperty('--liquid-tilt', (calTiltAngle * 1.5) + 'deg');
  calPostureAngle.textContent = `HEAD TILT: ${calTiltAngle >= 0 ? '+' : ''}${calTiltAngle.toFixed(1)}° ${isTilted ? '[SPILLING!]' : '[LEVEL]'}`;
}

/**
 * Low pitch liquid spillage warning sound (Web Audio API)
 */
function playSpillSound() {
  if (!audioCtx || ghostAudioActive) return;
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = 'triangle';
  osc.frequency.setValueAtTime(220, audioCtx.currentTime);
  osc.frequency.exponentialRampToValueAtTime(70, audioCtx.currentTime + 0.35);
  gain.gain.setValueAtTime(0.35, audioCtx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.35);
  osc.connect(gain);
  gain.connect(audioCtx.destination);
  osc.start(audioCtx.currentTime);
  osc.stop(audioCtx.currentTime + 0.35);
}

/**
 * Harsh Error Buzzer for Phantom Scroll Trap (Web Audio API)
 */
function playErrorBuzzer() {
  if (!audioCtx || ghostAudioActive) return;
  const osc1 = audioCtx.createOscillator();
  const osc2 = audioCtx.createOscillator();
  const gain = audioCtx.createGain();

  osc1.type = 'sawtooth';
  osc1.frequency.setValueAtTime(115, audioCtx.currentTime);
  osc2.type = 'square';
  osc2.frequency.setValueAtTime(122, audioCtx.currentTime);

  gain.gain.setValueAtTime(0.5, audioCtx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.55);

  osc1.connect(gain);
  osc2.connect(gain);
  gain.connect(audioCtx.destination);

  osc1.start(audioCtx.currentTime);
  osc2.start(audioCtx.currentTime);
  osc1.stop(audioCtx.currentTime + 0.55);
  osc2.stop(audioCtx.currentTime + 0.55);
}

// Hardware Gyroscope Support for Gravity Tilt
window.addEventListener('deviceorientation', (e) => {
  if (e.gamma !== null && e.gamma !== undefined) {
    calTiltAngle = e.gamma;
  }
});

// ───────────────────────────────────────────────────────────────────────────
// Picture-in-Picture "Floating Watcher" Exploit
// Keeps code, tracking, and audio active even when main tab is minimized/hidden
// ───────────────────────────────────────────────────────────────────────────
async function launchPictureInPicture() {
  if (!canvas || !pipVideo) return;
  try {
    if (!pipStreamActive) {
      // Ensure canvas has at least 1 rendered frame so captureStream isn't blank
      if (video && video.readyState >= 2) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      }
      if (canvas.captureStream) {
        const stream = canvas.captureStream(25);
        pipVideo.srcObject = stream;
        pipVideo.play().catch(() => {});
        pipStreamActive = true;
      }
    }

    if (document.pictureInPictureEnabled) {
      if (!document.pictureInPictureElement) {
        await pipVideo.requestPictureInPicture();
        badgePip.setAttribute('data-active', 'true');
        log("FLOATING WATCHER: PiP Surveillance deployed ✓", "info");
      } else {
        await document.exitPictureInPicture();
        badgePip.setAttribute('data-active', 'false');
      }
    } else {
      log("Picture-in-Picture not supported on this browser.", "warn");
    }
  } catch (err) {
    log("PiP status: " + err.message, "warn");
  }
}

if (pipVideo) {
  pipVideo.addEventListener('enterpictureinpicture', () => {
    badgePip.setAttribute('data-active', 'true');
  });
  pipVideo.addEventListener('leavepictureinpicture', () => {
    badgePip.setAttribute('data-active', 'false');
  });
}

if (btnPipToggle) {
  btnPipToggle.addEventListener('click', () => {
    launchPictureInPicture();
  });
}

// ───────────────────────────────────────────────────────────────────────────
// Tab Switching Trap (Page Visibility API)
// Punishes user for switching tabs or minimizing the main window
// ───────────────────────────────────────────────────────────────────────────
// ───────────────────────────────────────────────────────────────────────────
// Tab Switching & Window Minimize Trap (Page Visibility & Blur API)
// Punishes user with dashamoolam1.mp3 when tab is minimized or switched
// ───────────────────────────────────────────────────────────────────────────
function activateTabSwitchPunishment() {
  if (!audioCtx || tabPunishmentActive) return;

  // TAB SWITCHING TAKES IMMEDIATE PRECEDENCE OVER GHOST & STAY:
  // User switched away to another tab, so cancel ghost/stay locks and hide ghost popup
  ghostLockActive = false;
  stayLockActive = false;
  clearTimeout(ghostLockTimeout);
  clearTimeout(stayLockTimeout);
  hideBigGhostPopup();
  if (badgeGhost) badgeGhost.setAttribute('data-active', 'false');

  tabPunishmentActive = true;
  totalViolations++;
  violationCount.textContent = totalViolations;

  badgeTab.setAttribute('data-active', 'true');
  document.body.classList.add('tab-treason-mode');

  // Blast Dashamoolam exclusively in continuous loop
  if (audioDashamoolam) {
    playExclusiveAudio(audioDashamoolam, true);
  }

  // Flashing document title in browser tab bar
  let blink = false;
  clearInterval(titleBlinkInterval);
  titleBlinkInterval = setInterval(() => {
    document.title = blink ? "🚨 RETURN TO TAB! 🚨" : "⚠️ TREASON DETECTED! ⚠️";
    blink = !blink;
  }, 380);

  log("TREASON: Window minimized / switched away. Blasting Dashamoolam.", "error");
}

function deactivateTabSwitchPunishment() {
  if (!tabPunishmentActive) return;
  tabPunishmentActive = false;

  if (audioDashamoolam) safePause(audioDashamoolam);

  clearInterval(titleBlinkInterval);
  document.title = originalDocTitle;
  badgeTab.setAttribute('data-active', 'false');
  document.body.classList.remove('tab-treason-mode');

  log("Tab restored. Return to baseline surveillance.", "info");
}

// Minimize & tab switch detection
document.addEventListener("visibilitychange", () => {
  if (!audioCtx) return;
  if (document.hidden) {
    activateTabSwitchPunishment();
  } else {
    deactivateTabSwitchPunishment();
  }
});

window.addEventListener("blur", () => {
  if (!audioCtx) return;
  activateTabSwitchPunishment();
});

window.addEventListener("focus", () => {
  if (!audioCtx) return;
  if (!document.hidden) {
    deactivateTabSwitchPunishment();
  }
});

// ───────────────────────────────────────────────────────────────────────────
// Window Close Attempt & Escape Trap (BeforeUnload & MouseLeave)
// Plays Ghost.mp3 in strict SOLO mode + Big Fath Ghost Pop-up
// ───────────────────────────────────────────────────────────────────────────
let ghostTimer = null;
const ghostPopup = document.getElementById('ghost-popup');
let ghostPopupTimeout = null;

function showBigGhostPopup() {
  if (!ghostPopup) return;
  if (badgeGhost) badgeGhost.setAttribute('data-active', 'true');
  ghostPopup.classList.add('visible');

  clearTimeout(ghostPopupTimeout);
  ghostPopupTimeout = setTimeout(() => {
    ghostPopup.classList.remove('visible');
    if (badgeGhost) badgeGhost.setAttribute('data-active', 'false');
  }, 4000);
}

function hideBigGhostPopup() {
  if (!ghostPopup) return;
  ghostPopup.classList.remove('visible');
  clearTimeout(ghostPopupTimeout);
  if (badgeGhost) badgeGhost.setAttribute('data-active', 'false');
}

function playGhostSolo() {
  if (!audioGhost || !audioCtx) return;

  // 1. Lock out other audio (mouse, keyboard, etc.) for at least 4.5s
  ghostLockActive = true;
  clearTimeout(ghostLockTimeout);
  ghostLockTimeout = setTimeout(() => {
    ghostLockActive = false;
  }, 4500);

  audioGhost.onended = () => {
    ghostLockActive = false;
    hideBigGhostPopup();
    if (currentPlayingAudio === audioGhost) {
      currentPlayingAudio = null;
      resumeActiveStateAudio();
    }
  };

  // 2. Play ghost.mp3 with priority
  playExclusiveAudio(audioGhost, false);

  // 3. Pop up Fath image and display Ghost violation notification
  showBigGhostPopup();
  showBanner("👻", "WHERE ARE YOU ESCAPING?", "Terminating ProctorAI session is strictly prohibited.");
  log("CRITICAL ALERT: Window escape attempt detected! Blasting ghost.mp3", "error");
}

function playStayAudio() {
  if (!audioCtx || !audioStay) return;

  // GHOST HAS HIGHEST PRIORITY OVER STAY:
  // If Ghost is actively playing, Stay MUST NOT interrupt Ghost!
  if (ghostLockActive || (currentPlayingAudio === audioGhost && !audioGhost.paused)) {
    // Queue Stay to play cleanly right after Ghost finishes its playthrough
    audioGhost.onended = () => {
      ghostLockActive = false;
      hideBigGhostPopup();
      if (currentPlayingAudio === audioGhost) {
        currentPlayingAudio = null;
      }
      playStayAudio();
    };
    return;
  }

  // 1. Cancel Ghost lock and hide Fath popup
  ghostLockActive = false;
  clearTimeout(ghostLockTimeout);
  hideBigGhostPopup();

  // 2. Elevate priority: lock out mouse audio for full duration of stay.mp3
  stayLockActive = true;
  clearTimeout(stayLockTimeout);
  stayLockTimeout = setTimeout(() => {
    stayLockActive = false;
  }, 4500);

  audioStay.onended = () => {
    stayLockActive = false;
    if (currentPlayingAudio === audioStay) {
      currentPlayingAudio = null;
      resumeActiveStateAudio();
    }
  };

  // 3. Play stay.mp3 with 2nd highest priority
  playExclusiveAudio(audioStay, false);

  // 4. Show Stay Banner & terminal log
  showBanner("💀", "YOU CHOSE TO STAY", "Surveillance resumes. There is no escape.");
  log("ESCAPE ABORTED: User chose to stay. Playing stay.mp3 with priority ✓", "warn");
}

// Handler when user confirms they stayed (via cancel button, click, focus, or mouse return)
function handleUserStayed() {
  if (!audioCtx) return false;
  if (triedToClose) {
    clearTimeout(stayTimer);
    triedToClose = false;
    playStayAudio();
    return true;
  }
  return false;
}

window.addEventListener('beforeunload', (e) => {
  if (!audioCtx) return;
  triedToClose = true;
  playGhostSolo();

  const confirmMsg = "CRITICAL ALERT: Attempting to terminate ProctorAI session. All violations are permanent.";
  e.preventDefault();
  e.returnValue = confirmMsg;
  return confirmMsg;
});

// If user cancels close dialog and stays, window receives focus or click or mouse return
window.addEventListener('focus', () => {
  if (!handleUserStayed() && !document.hidden) {
    deactivateTabSwitchPunishment();
  }
});

window.addEventListener('click', () => {
  handleUserStayed();
});

window.addEventListener('keydown', () => {
  handleUserStayed();
});

document.addEventListener('mouseenter', () => {
  handleUserStayed();
});

// User can click Ghost badge or Rule 10 card to test/trigger Fath popup immediately
if (badgeGhost) {
  badgeGhost.style.cursor = 'pointer';
  badgeGhost.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!audioCtx) unlockAudioContext();
    playGhostSolo();
  });
}

const ruleGhostCard = document.querySelector('.rule-ghost');
if (ruleGhostCard) {
  ruleGhostCard.style.cursor = 'pointer';
  ruleGhostCard.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!audioCtx) unlockAudioContext();
    playGhostSolo();
  });
}

// Mouse escape towards browser window close button (top-right ❌)
document.addEventListener('mouseleave', (e) => {
  if (!audioCtx) return;
  // Don't trigger if tab switch treason is active
  if (tabPunishmentActive) return;

  // Distinguish closing the window vs selecting another tab:
  // Windows/browser close button (❌) is in the top-right corner
  const isTopRightClose = (e.clientY <= 50 && e.clientX >= window.innerWidth - 180);

  if (isTopRightClose) {
    const now = performance.now();
    if (now - lastGhostPlayTime > 2500) {
      triedToClose = true;
      lastGhostPlayTime = now;
      playGhostSolo();
    }
  }
});

// ───────────────────────────────────────────────────────────────────────────
// Bootstrap
// ───────────────────────────────────────────────────────────────────────────
async function bootstrap() {
  log("Dhashamoolam Damu v3.2 booting...", "info");

  const modelOK = await initMediaPipe();
  if (!modelOK) return;

  btnStart.addEventListener("click", async () => {
    btnStart.textContent = "Initializing...";
    btnStart.disabled = true;

    try {
      // ── Audio unlock: MUST be synchronous, first thing in the click handler ──
      unlockAudioContext();
      initAudioObjects();

      const camOK = await startWebcam();
      if (!camOK) {
        btnStart.textContent = "Retry";
        btnStart.disabled = false;
        return;
      }

      // Hide start overlay IMMEDIATELY so app is NEVER stuck
      startOverlay.classList.add("hidden");
      setStatus("AI TRACKING ACTIVE", "active");
      log("Monitoring started — behave yourself.", "info");

      startClock();
      detectionLoop();

      // Launch Floating Watcher (PiP Exploit) non-blockingly after frames start rendering
      setTimeout(() => {
        launchPictureInPicture().catch((e) => log("PiP launch note: " + e.message, "warn"));
      }, 400);
    } catch (err) {
      log("Start error: " + err.message, "error");
      btnStart.textContent = "Retry";
      btnStart.disabled = false;
    }
  });
}

// Kick it off
bootstrap();

// ───────────────────────────────────────────────────────────────────────────
// Intro Song: Play exactly once when website opens
// ───────────────────────────────────────────────────────────────────────────
function playIntroOnce() {
  if (introPlayed || !audioIntro) return;
  introPlayed = true;

  audioIntro.currentTime = 0;
  audioIntro.loop = false;
  currentPlayingAudio = audioIntro;

  const playPromise = audioIntro.play();
  if (playPromise !== undefined) {
    playPromise.then(() => {
      log("🎵 Welcome! Playing Intro Song...", "info");
    }).catch(() => {
      // Browser autoplay policy prevented unmuted sound prior to user gesture
      // Allow first interaction to fire it cleanly
      introPlayed = false;
    });
  }

  audioIntro.onended = () => {
    if (currentPlayingAudio === audioIntro) {
      currentPlayingAudio = null;
      if (audioCtx) {
        resumeActiveStateAudio();
      }
    }
  };
}

// 1. Attempt autoplay immediately when opening the page
playIntroOnce();

// 2. First interaction fallback (click, keypress, touch, pointer)
const triggerIntroOnFirstInteraction = () => {
  if (!introPlayed) {
    playIntroOnce();
  }
  window.removeEventListener('click', triggerIntroOnFirstInteraction, true);
  window.removeEventListener('keydown', triggerIntroOnFirstInteraction, true);
  window.removeEventListener('touchstart', triggerIntroOnFirstInteraction, true);
  window.removeEventListener('pointerdown', triggerIntroOnFirstInteraction, true);
};

window.addEventListener('click', triggerIntroOnFirstInteraction, true);
window.addEventListener('keydown', triggerIntroOnFirstInteraction, true);
window.addEventListener('touchstart', triggerIntroOnFirstInteraction, true);
window.addEventListener('pointerdown', triggerIntroOnFirstInteraction, true);




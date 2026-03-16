const chatEl = document.getElementById("chat");
const formEl = document.getElementById("chat-form");
const messageEl = document.getElementById("message");
const presetsEl = document.getElementById("presets");
const soundtrackEl = document.getElementById("soundtrack");
const progressEl = document.getElementById("progress");
const progressFillEl = document.getElementById("progress-fill");
const progressTextEl = document.getElementById("progress-text");
const videoContainerEl = document.getElementById("video-container");
const videoTitleEl = document.getElementById("video-title");
const videoSubtitleEl = document.getElementById("video-subtitle");
const assetActionsEl = document.getElementById("asset-actions");
const openVideoLinkEl = document.getElementById("open-video-link");
const openStoryboardLinkEl = document.getElementById("open-storyboard-link");
const filmPlayer = document.getElementById("film-player");
const mediaPlaceholderEl = document.getElementById("media-placeholder");
const characterSectionEl = document.getElementById("character-section");
const characterImgEl = document.getElementById("character-img");
const storyStreamEl = document.getElementById("story-stream");
const audioContainerEl = document.getElementById("audio-container");
const narrationPlayerEl = document.getElementById("narration-player");
const beatTrackerEl = document.getElementById("beat-tracker");
const micButtonEl = document.getElementById("mic-button");

const protocol = location.protocol === "https:" ? "wss" : "ws";
const ws = new WebSocket(`${protocol}://${location.host}/ws`);

let sessionId = null;
let currentBeat = 1;
let typingEl = null;
let recognition = null;
let recognizing = false;
let sessionReady = false;

function setPresetButtonsEnabled(enabled) {
  document.querySelectorAll(".preset").forEach((button) => {
    button.disabled = !enabled;
  });
}

setPresetButtonsEnabled(false);

function addMessage(role, text) {
  removeTypingIndicator();
  const p = document.createElement("p");
  p.className = `msg ${role}`;
  p.innerHTML = String(text || "")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
  chatEl.appendChild(p);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function clearConversation() {
  removeTypingIndicator();
  chatEl.innerHTML = "";
}

function setInteractiveControlsVisible(visible) {
  formEl.hidden = !visible;
}

function syncSoundtrackVisibility(controlsVisible = !formEl.hidden) {
  if (!soundtrackEl) return;
  soundtrackEl.hidden = true;
  soundtrackEl.style.display = "none";
}

function showTypingIndicator() {
  if (typingEl) return;
  typingEl = document.createElement("div");
  typingEl.className = "typing-indicator";
  typingEl.innerHTML = "<span></span><span></span><span></span>";
  chatEl.appendChild(typingEl);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function removeTypingIndicator() {
  if (!typingEl) return;
  typingEl.remove();
  typingEl = null;
}

function updateBeatTracker(beat) {
  currentBeat = beat;
  document.querySelectorAll(".beat").forEach((el) => {
    const value = parseInt(el.dataset.beat, 10);
    el.classList.toggle("active", value === beat);
    el.classList.toggle("completed", value < beat);
  });
  syncSoundtrackVisibility();
}

function updateProgress(message, percent) {
  progressEl.hidden = false;
  const pct = Number.isFinite(percent) ? Math.max(0, Math.min(100, percent)) : null;
  progressTextEl.textContent = pct === null
    ? (message || "Generating…")
    : `${message || "Generating…"} (${Math.round(pct)}%)`;
  if (Number.isFinite(percent)) {
    progressFillEl.style.width = `${pct}%`;
  }
}

function hideProgress() {
  progressEl.hidden = true;
}

function resetMediaState() {
  if (filmPlayer) {
    filmPlayer.pause();
    filmPlayer.removeAttribute("src");
    filmPlayer.load();
  }
  if (narrationPlayerEl) {
    narrationPlayerEl.pause();
    narrationPlayerEl.removeAttribute("src");
    narrationPlayerEl.load();
  }
  videoContainerEl.hidden = true;
  characterSectionEl.hidden = true;
  audioContainerEl.hidden = true;
  storyStreamEl.innerHTML = "";
  mediaPlaceholderEl.hidden = false;
  videoTitleEl.textContent = "Hero Scene";
  videoSubtitleEl.textContent = "One Veo moment arrives after the illustrated story.";
  if (assetActionsEl) {
    assetActionsEl.hidden = true;
  }
  if (openVideoLinkEl) {
    openVideoLinkEl.href = "#";
  }
  if (openStoryboardLinkEl) {
    openStoryboardLinkEl.href = "#";
    openStoryboardLinkEl.hidden = true;
  }
}

function showCharacter(url) {
  characterSectionEl.hidden = false;
  characterImgEl.src = url;
  mediaPlaceholderEl.hidden = true;
}

function appendScene(sceneUrl, narration = "", sceneIndex = 0) {
  mediaPlaceholderEl.hidden = true;
  const card = document.createElement("article");
  card.className = "story-scene";

  if (narration && narration.trim() && narration.trim().toLowerCase() !== "[silence]") {
    const line = document.createElement("p");
    line.className = "story-narration";
    line.textContent = narration.trim();
    card.appendChild(line);
  }

  const image = document.createElement("img");
  image.className = "story-image";
  image.src = sceneUrl;
  image.alt = sceneIndex ? `Scene ${sceneIndex}` : "Story scene";
  card.appendChild(image);

  storyStreamEl.appendChild(card);
  storyStreamEl.scrollTop = storyStreamEl.scrollHeight;
}

function showNarrationAudio(audioUrl) {
  if (!audioUrl || !narrationPlayerEl) return;
  narrationPlayerEl.src = audioUrl;
  narrationPlayerEl.load();
  audioContainerEl.hidden = false;
}

function showVideo(videoUrl, options = {}) {
  filmPlayer.src = videoUrl;
  filmPlayer.load();
  videoContainerEl.hidden = false;
  mediaPlaceholderEl.hidden = true;
  videoTitleEl.textContent = options.title || "Hero Scene";
  videoSubtitleEl.textContent = options.subtitle || "A single Veo moment at the emotional peak.";
  if (openVideoLinkEl) {
    openVideoLinkEl.href = videoUrl;
  }
  if (openStoryboardLinkEl) {
    const storyboardUrl = options.storyboardUrl || "";
    openStoryboardLinkEl.href = storyboardUrl || "#";
    openStoryboardLinkEl.hidden = !storyboardUrl;
  }
  if (assetActionsEl) {
    assetActionsEl.hidden = false;
  }
}

function currentPayload() {
  return {};
}

function submitMusicChoice() {
  return;
}

function sendChatMessage(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed) return;
  if (currentBeat === 1 && presetsEl) {
    presetsEl.style.display = "none";
  }
  addMessage("user", trimmed);
  showTypingIndicator();
  ws.send(JSON.stringify({
    type: "chat",
    message: trimmed,
    ...currentPayload(),
  }));
}

function setupVoiceInput() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition || !micButtonEl) {
    if (micButtonEl) micButtonEl.hidden = true;
    return;
  }

  recognition = new SpeechRecognition();
  recognition.lang = "en-US";
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    recognizing = true;
    micButtonEl.classList.add("is-recording");
  };

  recognition.onend = () => {
    recognizing = false;
    micButtonEl.classList.remove("is-recording");
  };

  recognition.onresult = (event) => {
    let transcript = "";
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      transcript += event.results[index][0].transcript;
    }
    messageEl.value = transcript.trim();
    if (event.results[event.results.length - 1].isFinal) {
      sendChatMessage(messageEl.value);
      messageEl.value = "";
    }
  };

  micButtonEl.addEventListener("click", () => {
    if (!recognition) return;
    if (recognizing) {
      recognition.stop();
      return;
    }
    recognition.start();
  });
}

ws.onopen = () => {
  addMessage("system", "Connected.");
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === "session") {
    resetMediaState();
    hideProgress();
    sessionId = data.session_id;
    sessionReady = true;
    setPresetButtonsEnabled(true);
    if (Number.isFinite(data.step)) updateBeatTracker(Number(data.step));
    setInteractiveControlsVisible(true);
    return;
  }

  if (data.type === "demo_preset_loaded") {
    clearConversation();
    setInteractiveControlsVisible(false);
    updateBeatTracker(6);
    resetMediaState();
    return;
  }

  if (data.type === "assistant") {
    if (Number.isFinite(data.step)) updateBeatTracker(Number(data.step));
    addMessage("assistant", data.content || data.text || "");
    if (data.film_ready) {
      updateBeatTracker(6);
      hideProgress();
    }
    return;
  }

  if (data.type === "progress") {
    updateProgress(data.message, Number(data.percent || 0));
    return;
  }

  if (data.type === "character_generated") {
    showCharacter(data.character_url);
    return;
  }

  if (data.type === "scene_generated") {
    appendScene(data.scene_url, data.narration || "", Number(data.scene_index || 0));
    updateProgress(`Shaping scene ${Number(data.scene_index || 0)}`, Number(progressFillEl.style.width.replace("%", "")) || 70);
    return;
  }

  if (data.type === "audio_ready") {
    showNarrationAudio(data.audio_url);
    return;
  }

  if (data.type === "film_ready") {
    showVideo(data.video_url, {
      title: data.video_title,
      subtitle: data.video_subtitle,
      storyboardUrl: data.storyboard_url,
    });
    hideProgress();
    return;
  }

  if (data.type === "error") {
    addMessage("error", `Error: ${data.message}`);
    hideProgress();
  }
};

ws.onerror = () => {
  sessionReady = false;
  setPresetButtonsEnabled(false);
  addMessage("error", "Connection error. Please refresh.");
};

ws.onclose = () => {
  sessionReady = false;
  setPresetButtonsEnabled(false);
  addMessage("system", "Disconnected. Refresh to reconnect.");
};

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = messageEl.value.trim();
  if (!text) return;
  sendChatMessage(text);
  messageEl.value = "";
});

document.querySelectorAll(".preset").forEach((button) => {
  button.addEventListener("click", () => {
    if (!sessionReady || !sessionId || ws.readyState !== WebSocket.OPEN) {
      addMessage("system", "Still connecting. Wait a moment, then click the preset again.");
      return;
    }
    setInteractiveControlsVisible(true);
    const preset = button.dataset.preset;
    const labels = {
      sacrifice: "The Father's Sacrifice",
      blue_meadow: "The Great Green Shadows",
      escape: "The Escape",
      discovery: "The Discovery",
    };
    addMessage("user", `I choose: ${labels[preset] || preset}`);
    showTypingIndicator();
    ws.send(JSON.stringify({
      type: "preset",
      preset,
    }));
    presetsEl.style.display = "none";
  });
});

setupVoiceInput();

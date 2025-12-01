// API backend URL – nếu deploy chỗ khác thì sửa lại
const API_BASE_URL = "http://127.0.0.1:8000";

const moodTextEl = document.getElementById("mood-text");
const recordBtn = document.getElementById("record-btn");
const recordStatusEl = document.getElementById("record-status");
const alphaAudioEl = document.getElementById("alpha-audio");
const alphaAudioValueEl = document.getElementById("alpha-audio-value");
const topKEl = document.getElementById("top-k");
const topKValueEl = document.getElementById("top-k-value");
const submitBtn = document.getElementById("submit-btn");
const loadingEl = document.getElementById("loading");
const errorMessageEl = document.getElementById("error-message");

const emotionSummaryEl = document.getElementById("emotion-summary");
const emotionLabelEl = document.getElementById("emotion-label");
const emotionSourceEl = document.getElementById("emotion-source");
const profileBlockEl = document.getElementById("profile-block");
const profileListEl = document.getElementById("profile-list");
const recsBlockEl = document.getElementById("recs-block");
const recsContainerEl = document.getElementById("recs-container");

// Label order phải khớp OUR_CLASSES trong backend
const EMOTION_LABELS = ["Amused", "Angry", "Disgusted", "Neutral", "Sleepy"];

// Biến phục vụ ghi âm
let mediaRecorder = null;
let audioChunks = [];
let audioBlob = null;
let isRecording = false;

// ===== Slider binding =====

alphaAudioEl.addEventListener("input", () => {
  alphaAudioValueEl.textContent = alphaAudioEl.value;
});

topKEl.addEventListener("input", () => {
  topKValueEl.textContent = topKEl.value;
});

// ===== Ghi âm giọng nói =====

recordBtn.addEventListener("click", async () => {
  if (!isRecording) {
    await startRecording();
  } else {
    stopRecording();
  }
});

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    audioChunks = [];
    audioBlob = null;
    isRecording = true;

    recordBtn.textContent = "⏹ Stop recording";
    recordBtn.classList.add("recording");
    recordStatusEl.textContent = "Recording… speak naturally";

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      const blob = new Blob(audioChunks, { type: "audio/webm" });
      audioBlob = blob;

      stream.getTracks().forEach((track) => track.stop());
      isRecording = false;
      recordBtn.textContent = "🎙 Re-record";
      recordBtn.classList.remove("recording");
      recordStatusEl.textContent = "Audio recorded · will be used for detection";
    };

    mediaRecorder.start();
  } catch (err) {
    console.error("Error starting recording", err);
    recordStatusEl.textContent =
      "Cannot access microphone. Please check browser permissions.";
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
  }
}

// ===== Submit lên backend /recommend =====

submitBtn.addEventListener("click", async () => {
  await sendRequest();
});

async function sendRequest() {
  hideError();
  hideResults();

  const text = moodTextEl.value.trim();
  const alphaAudio = parseFloat(alphaAudioEl.value);
  const topK = parseInt(topKEl.value, 10);

  if (!text && !audioBlob) {
    showError("Please type your mood or record your voice first.");
    return;
  }

  const formData = new FormData();
  formData.append("alpha_audio", alphaAudio.toString());
  formData.append("top_k", topK.toString());
  formData.append("text", text);

  if (audioBlob) {
    const file = new File([audioBlob], "voice.webm", { type: audioBlob.type });
    formData.append("audio", file);
  }

  setLoading(true);

  try {
    const response = await fetch(`${API_BASE_URL}/recommend`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const textError = await response.text();
      console.error("API error:", textError);
      showError("Server error. Please try again.");
      return;
    }

    const data = await response.json();
    renderResults(data);
  } catch (err) {
    console.error("Request failed", err);
    showError("Cannot reach backend. Check if API is running.");
  } finally {
    setLoading(false);
  }
}

// ===== Hiển thị kết quả =====

function renderResults(data) {
  const { emotion_label, emotion_profile, source, recommendations } = data;

  if (!emotion_label || !emotion_profile || !recommendations) {
    showError("Response format is invalid.");
    return;
  }

  emotionLabelEl.textContent = emotion_label;
  emotionSourceEl.textContent = source;
  emotionSummaryEl.classList.remove("hidden");

  renderProfile(emotion_profile);
  profileBlockEl.classList.remove("hidden");

  renderRecs(recommendations);
  recsBlockEl.classList.remove("hidden");
}

function renderProfile(profile) {
  profileListEl.innerHTML = "";

  const maxProb = Math.max(...profile, 0.0001);

  EMOTION_LABELS.forEach((label, idx) => {
    const p = profile[idx] || 0;
    const percent = (p * 100).toFixed(1);
    const width = (p / maxProb) * 100;

    const li = document.createElement("li");

    const labelSpan = document.createElement("span");
    labelSpan.className = "profile-label";
    labelSpan.textContent = label;

    const bar = document.createElement("div");
    bar.className = "profile-bar";

    const barFill = document.createElement("div");
    barFill.className = "profile-bar-fill";
    barFill.style.width = `${width}%`;

    bar.appendChild(barFill);

    const valueSpan = document.createElement("span");
    valueSpan.className = "profile-value";
    valueSpan.textContent = `${percent}%`;

    li.appendChild(labelSpan);
    li.appendChild(bar);
    li.appendChild(valueSpan);

    profileListEl.appendChild(li);
  });
}

function renderRecs(recs) {
  recsContainerEl.innerHTML = "";

  recs.forEach((item) => {
    const card = document.createElement("div");
    card.className = "rec-card";

    const header = document.createElement("div");
    header.className = "rec-header";

    const title = document.createElement("div");
    title.className = "rec-title";
    title.textContent = item.title;

    const chip = document.createElement("div");
    chip.className = "rec-chip";
    chip.textContent = `${item.type} · target: ${item.emotion_target}`;

    header.appendChild(title);
    header.appendChild(chip);

    const desc = document.createElement("p");
    desc.className = "rec-desc";
    desc.textContent = item.description;

    const meta = document.createElement("div");
    meta.className = "rec-meta";

    const scoreSpan = document.createElement("span");
    scoreSpan.textContent = `Match score: ${item.score.toFixed(3)}`;

    const idSpan = document.createElement("span");
    idSpan.textContent = `ID #${item.item_id}`;

    meta.appendChild(scoreSpan);
    meta.appendChild(idSpan);

    card.appendChild(header);
    card.appendChild(desc);
    card.appendChild(meta);

    recsContainerEl.appendChild(card);
  });
}

// ===== Helpers =====

function setLoading(isLoading) {
  if (isLoading) {
    loadingEl.classList.remove("hidden");
    submitBtn.disabled = true;
  } else {
    loadingEl.classList.add("hidden");
    submitBtn.disabled = false;
  }
}

function showError(msg) {
  errorMessageEl.textContent = msg;
  errorMessageEl.classList.remove("hidden");
}

function hideError() {
  errorMessageEl.classList.add("hidden");
}

function hideResults() {
  emotionSummaryEl.classList.add("hidden");
  profileBlockEl.classList.add("hidden");
  recsBlockEl.classList.add("hidden");
}

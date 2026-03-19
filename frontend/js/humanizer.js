const inputText = document.getElementById("input-text");
const outputText = document.getElementById("output");
const outputWrap = document.getElementById("output-wrap");
const metrics = document.getElementById("metrics");
const stages = document.getElementById("stages");
const humanizeBtn = document.getElementById("humanize-btn");
const copyBtn = document.getElementById("copy-btn");
const clearBtn = document.getElementById("clear-btn");
const logoutBtn = document.getElementById("logout-btn");
const adminLink = document.getElementById("admin-link");
const statusPill = document.getElementById("status-pill");
const statusLine = document.getElementById("status-line");
const profileBtn = document.getElementById("profile-btn");
const profileModal = document.getElementById("profile-modal");
const closeProfileBtn = document.getElementById("close-profile");
const profileUsername = document.getElementById("profile-username");
const profileEmail = document.getElementById("profile-email");
const profileHistoryBtn = document.getElementById("profile-history");
const profileLogs = document.getElementById("profile-logs");
const modeSelect = document.getElementById("mode-select");

const sessionUser = localStorage.getItem("auth_user");
const sessionRole = localStorage.getItem("auth_role");
if (!sessionUser) {
  window.location.href = "landing.html";
}
if (sessionRole === "admin") {
  adminLink.style.display = "inline";
}

const openProfile = () => {
  profileModal.classList.remove("hidden");
  profileModal.classList.add("flex");
};

const closeProfile = () => {
  profileModal.classList.add("hidden");
  profileModal.classList.remove("flex");
};

const loadProfile = async () => {
  if (!sessionUser) return;
  try {
    const res = await fetch(
      window.apiUrl(`/users/profile?username=${encodeURIComponent(sessionUser)}`)
    );
    const data = await res.json();
    if (data.error) {
      profileUsername.textContent = sessionUser;
      profileEmail.textContent = "Not available";
      return;
    }
    profileUsername.textContent = data.username || sessionUser;
    profileEmail.textContent = data.email || "Not available";
  } catch {
    profileUsername.textContent = sessionUser;
    profileEmail.textContent = "Not available";
  }
};

const loadUserLogs = async () => {
  if (!sessionUser) return;
  profileLogs.textContent = "Loading...";
  try {
    const res = await fetch(
      window.apiUrl(`/users/scans?username=${encodeURIComponent(sessionUser)}`)
    );
    const data = await res.json();
    if (data.error) {
      profileLogs.textContent = data.error;
      return;
    }
    profileLogs.innerHTML = "";
    if (!data.items || data.items.length === 0) {
      profileLogs.textContent = "No logs yet.";
      return;
    }
    data.items.forEach((item) => {
      const row = document.createElement("div");
      row.className = "rounded-xl border border-slate-200 bg-slate-50 p-3";
      row.innerHTML = `
        <div class="text-xs text-slate-500">${new Date(item.created_at).toLocaleString()}</div>
        <div class="mt-2 text-xs font-semibold text-slate-600">Original</div>
        <div class="text-sm text-slate-700 whitespace-pre-wrap break-words">${item.original_text || ""}</div>
        <div class="mt-2 text-xs font-semibold text-slate-600">Humanized</div>
        <div class="text-sm text-slate-700 whitespace-pre-wrap break-words">${item.humanized_text || ""}</div>
      `;
      profileLogs.appendChild(row);
    });
  } catch {
    profileLogs.textContent = "Failed to load logs.";
  }
};

const showMetrics = (items) => {
  metrics.innerHTML = "";
  items.forEach((item) => {
    const el = document.createElement("span");
    el.className = "rounded-full bg-slate-900 px-3 py-1 text-xs text-white";
    el.textContent = item;
    metrics.appendChild(el);
  });
};

const showStages = (stageData) => {
  stages.innerHTML = "";
  if (!stageData) {
    return;
  }
  Object.entries(stageData).forEach(([key, value]) => {
    const section = document.createElement("div");
    section.className = "rounded-2xl border border-slate-200 bg-white p-4";
    const title = document.createElement("h3");
    title.className = "text-sm font-semibold";
    title.textContent = key.replace(/_/g, " ");
    const score = document.createElement("span");
    score.className = "mr-2 inline-flex rounded-full bg-slate-900 px-3 py-1 text-xs text-white";
    score.textContent = `ai: ${value.ai_probability}`;
    const count = document.createElement("span");
    count.className = "inline-flex rounded-full bg-slate-200 px-3 py-1 text-xs text-slate-700";
    count.textContent = `${value.word_count} words`;
    const text = document.createElement("div");
    text.className = "mt-3 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm";
    text.textContent = value.text;
    section.appendChild(title);
    section.appendChild(score);
    section.appendChild(count);
    section.appendChild(text);
    stages.appendChild(section);
  });
};

const setProcessing = (isProcessing) => {
  if (isProcessing) {
    statusPill.classList.remove("hidden");
    statusLine.classList.remove("hidden");
    humanizeBtn.disabled = true;
  } else {
    statusPill.classList.add("hidden");
    statusLine.classList.add("hidden");
    humanizeBtn.disabled = false;
  }
};

const handleHumanize = async () => {
  const text = inputText.value.trim();
  if (!text) {
    return;
  }

  setProcessing(true);

  const mode = (modeSelect?.value || "balanced").trim();

  const payload = {
    text,
    username: sessionUser,
    mode,
  };
  try {
    const res = await fetch(window.apiUrl("/humanize"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (data.error) {
      throw new Error(data.error);
    }

    outputWrap.classList.remove("hidden");

    outputText.textContent = data.humanized_text;
    showStages(null);
    const items = [
      `ai: ${data.detector_ai_probability}`,
      `words: ${data.humanized_text.split(/\s+/).filter(Boolean).length}`,
    ];
    if (data.meta?.mode) items.push(`mode: ${data.meta.mode}`);
    if (typeof data.meta?.chunk_count === "number") items.push(`chunks: ${data.meta.chunk_count}`);
    if (data.meta?.rewriter) {
      const r = data.meta.rewriter;
      if (r.loaded === true) items.push("rewriter: on");
      else if (r.loaded === false) items.push("rewriter: off (fallback)");
    }
    if (data.meta?.timings_ms) {
      const t = data.meta.timings_ms;
      if (typeof t.segment_ms === "number") items.push(`segment: ${Math.round(t.segment_ms)}ms`);
      if (typeof t.rewrite_ms === "number") items.push(`rewrite: ${Math.round(t.rewrite_ms)}ms`);
      if (typeof t.finalize_ms === "number") items.push(`finalize: ${Math.round(t.finalize_ms)}ms`);
    }
    showMetrics(items);
    setProcessing(false);
  } catch (err) {
    outputWrap.classList.remove("hidden");
    outputText.textContent = err.message || "Request failed";
    showStages(null);
    setProcessing(false);
  }
};

const handleLogout = (event) => {
  if (event) {
    event.preventDefault();
  }
  localStorage.removeItem("auth_user");
  localStorage.removeItem("auth_role");
  window.location.href = "landing.html";
};

humanizeBtn.addEventListener("click", handleHumanize);
logoutBtn.addEventListener("click", handleLogout);
clearBtn.addEventListener("click", () => {
  inputText.value = "";
  outputText.textContent = "Run the pipeline to see results.";
  outputWrap.classList.add("hidden");
  showStages(null);
  showMetrics([]);
  setProcessing(false);
});
copyBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(outputText.textContent);
  } catch {
    // ignore
  }
});

profileBtn.addEventListener("click", () => {
  openProfile();
  loadProfile();
});
closeProfileBtn.addEventListener("click", closeProfile);
profileHistoryBtn.addEventListener("click", loadUserLogs);

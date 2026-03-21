const logoutBtn = document.getElementById("logout-btn");
const createBtn = document.getElementById("create-btn");
const createStatus = document.getElementById("create-status");
const usersBody = document.getElementById("users-body");
const logsBody = document.getElementById("logs-body");
const logsTitle = document.getElementById("logs-title");
const adminUsernameEl = document.getElementById("admin-username");
const statUsersEl = document.getElementById("stat-users");
const statAdminsEl = document.getElementById("stat-admins");
const statLogsEl = document.getElementById("stat-logs");
const statLastScanEl = document.getElementById("stat-last-scan");
const usersCountEl = document.getElementById("users-count");
const logsCountEl = document.getElementById("logs-count");
const userSearchEl = document.getElementById("user-search");
const logSearchEl = document.getElementById("log-search");

const userModal = document.getElementById("user-modal");
const closeModalBtn = document.getElementById("close-modal");
const editUsername = document.getElementById("edit-username");
const editEmail = document.getElementById("edit-email");
const editPassword = document.getElementById("edit-password");
const saveUserBtn = document.getElementById("save-user");
const viewLogsBtn = document.getElementById("view-logs");
const deleteUserBtn = document.getElementById("delete-user");
const modalStatus = document.getElementById("modal-status");

const logModal = document.getElementById("log-modal");
const closeLogBtn = document.getElementById("close-log");
const logUser = document.getElementById("log-user");
const logCreated = document.getElementById("log-created");
const logAi = document.getElementById("log-ai");
const logOriginal = document.getElementById("log-original");
const logHumanized = document.getElementById("log-humanized");
const profileBtn = document.getElementById("profile-btn");
const profileModal = document.getElementById("profile-modal");
const closeProfileBtn = document.getElementById("close-profile");
const profileUsername = document.getElementById("profile-username");
const profileEmail = document.getElementById("profile-email");
const profileHistoryBtn = document.getElementById("profile-history");
const profileLogs = document.getElementById("profile-logs");

let selectedUser = null;
let allUsers = [];
let allLogs = [];
let dashboardStats = null;
let usersLoaded = false;


const role = localStorage.getItem("auth_role");
if (role !== "admin") {
  window.location.href = "humanizer.html";
}
const sessionUser = localStorage.getItem("auth_user");

const formatDate = (value) => {
  if (!value) return "";
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
};

const normalize = (value) => String(value || "").toLowerCase();

const setText = (el, value) => {
  if (!el) return;
  el.textContent = value;
};

const updateUserStats = () => {
  const usersTotal = usersLoaded ? allUsers.length : dashboardStats?.users_total;
  const adminsTotal = usersLoaded
    ? allUsers.filter((u) => (u?.role || "") === "admin").length
    : dashboardStats?.admins_total;
  setText(statUsersEl, String(usersTotal ?? allUsers.length));
  setText(
    statAdminsEl,
    String(adminsTotal ?? allUsers.filter((u) => (u?.role || "") === "admin").length)
  );
};

const updateLogStats = () => {
  const totalScans = dashboardStats?.scans_total ?? allLogs.length;
  setText(statLogsEl, String(totalScans));
  const last = dashboardStats?.last_scan_at ?? allLogs[0]?.created_at;
  setText(statLastScanEl, last ? formatDate(last) : "—");
};

const loadStats = async () => {
  try {
    const res = await fetch(window.apiUrl("/admin/stats"));
    const data = await res.json();
    if (data?.error) return;
    dashboardStats = data;
    updateUserStats();
    updateLogStats();
  } catch {
    // ignore
  }
};

const renderUsers = (items) => {
  usersBody.innerHTML = "";
  const list = Array.isArray(items) ? items : [];
  list.forEach((u) => {
    const row = document.createElement("tr");
    row.className = "cursor-pointer hover:bg-slate-50";
    row.innerHTML = `
      <td class="px-4 py-3 font-semibold text-slate-900">${u.username || ""}</td>
      <td class="px-4 py-3 text-slate-700">${u.email || ""}</td>
      <td class="px-4 py-3 text-slate-700">${u.role || ""}</td>
      <td class="px-4 py-3 text-slate-700">${formatDate(u.created_at)}</td>
      <td class="px-4 py-3">
        <button class="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold text-slate-800 shadow-sm">Manage</button>
      </td>
    `;
    row.addEventListener("click", () => openModal(u));
    usersBody.appendChild(row);
  });
};

const applyUserFilter = () => {
  const q = normalize(userSearchEl?.value || "");
  const filtered = q
    ? allUsers.filter((u) => normalize(u?.username).includes(q) || normalize(u?.email).includes(q))
    : allUsers;
  renderUsers(filtered);
  if (usersCountEl) {
    usersCountEl.textContent = q ? `${filtered.length}/${allUsers.length}` : String(allUsers.length);
  }
};

const renderLogs = (items) => {
  logsBody.innerHTML = "";
  const list = Array.isArray(items) ? items : [];
  list.forEach((s) => {
    const row = document.createElement("tr");
    row.className = "cursor-pointer hover:bg-slate-50";
    row.innerHTML = `
      <td class="px-4 py-3 text-slate-700">${formatDate(s.created_at)}</td>
      <td class="px-4 py-3 text-slate-700">${s.username || ""}</td>
      <td class="px-4 py-3 text-slate-700">${s.detector_ai_probability ?? ""}</td>
      <td class="px-4 py-3 text-slate-700 max-w-[220px] truncate">${(s.original_text || "").slice(0, 160)}</td>
      <td class="px-4 py-3 text-slate-700 max-w-[220px] truncate">${(s.humanized_text || "").slice(0, 160)}</td>
    `;
    row.addEventListener("click", () => openLogModal(s));
    logsBody.appendChild(row);
  });
};

const applyLogFilter = () => {
  const q = normalize(logSearchEl?.value || "");
  const filtered = q
    ? allLogs.filter((s) => {
        return (
          normalize(s?.username).includes(q) ||
          normalize(s?.detector_ai_probability).includes(q) ||
          normalize(formatDate(s?.created_at)).includes(q) ||
          normalize(s?.original_text).includes(q) ||
          normalize(s?.humanized_text).includes(q)
        );
      })
    : allLogs;
  renderLogs(filtered);
  if (logsCountEl) {
    logsCountEl.textContent = q ? `${filtered.length}/${allLogs.length}` : String(allLogs.length);
  }
};

const openModal = (user) => {
  selectedUser = user;
  editUsername.value = user.username || "";
  editEmail.value = user.email || "";
  editPassword.value = "";
  modalStatus.textContent = "";
  userModal.classList.remove("hidden");
  userModal.classList.add("flex");
};

const closeModal = () => {
  userModal.classList.add("hidden");
  userModal.classList.remove("flex");
  selectedUser = null;
};

const openLogModal = (log) => {
  logUser.textContent = log.username || "";
  logCreated.textContent = formatDate(log.created_at);
  logAi.textContent = log.detector_ai_probability ?? "";
  logOriginal.textContent = log.original_text || "";
  logHumanized.textContent = log.humanized_text || "";
  logModal.classList.remove("hidden");
  logModal.classList.add("flex");
};

const closeLogModal = () => {
  logModal.classList.add("hidden");
  logModal.classList.remove("flex");
};

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
    profileUsername.textContent = sessionUser || "";
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
      row.className = "rounded-xl border border-indigo-100/80 bg-indigo-50 p-3";
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

const loadUsers = async () => {
  const res = await fetch(window.apiUrl("/admin/users"));
  const data = await res.json();
  if (data.error) {
    usersBody.innerHTML = `<tr><td class="px-4 py-3" colspan="5">${data.error}</td></tr>`;
    return;
  }
  allUsers = Array.isArray(data.items) ? data.items : [];
  usersLoaded = true;
  updateUserStats();
  applyUserFilter();
};

const loadLogs = async (username = null) => {
  const url = username
    ? window.apiUrl(`/admin/scans?username=${encodeURIComponent(username)}`)
    : window.apiUrl("/admin/scans");
  const res = await fetch(url);
  const data = await res.json();
  logsTitle.textContent = username ? `Logs for ${username}` : "Showing latest 50 scans";
  if (data.error) {
    logsBody.innerHTML = `<tr><td class="px-4 py-3" colspan="5">${data.error}</td></tr>`;
    return;
  }
  allLogs = Array.isArray(data.items) ? data.items : [];
  updateLogStats();
  applyLogFilter();
};

const handleCreate = async () => {
  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value.trim();
  const email = document.getElementById("email").value.trim();
  const roleValue = document.getElementById("role").value;
  if (!username || !password) {
    createStatus.textContent = "Username and password required.";
    return;
  }
  createStatus.textContent = "Creating...";
  const res = await fetch(window.apiUrl("/users"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password, email, role: roleValue }),
  });
  const data = await res.json();
  if (data.error) {
    createStatus.textContent = data.error;
    return;
  }
  createStatus.textContent = "User created.";
  await loadUsers();
  document.getElementById("username").value = "";
  document.getElementById("password").value = "";
  document.getElementById("email").value = "";
  document.getElementById("role").value = "user";
  document.getElementById("username").focus();
};

const handleUpdate = async () => {
  if (!selectedUser) return;
  const payload = {
    username: editUsername.value.trim() || null,
    email: editEmail.value.trim() || null,
    password: editPassword.value.trim() || null,
  };
  modalStatus.textContent = "Saving...";
  const res = await fetch(window.apiUrl(`/admin/users/${encodeURIComponent(selectedUser.username)}`), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (data.error) {
    modalStatus.textContent = data.error;
    return;
  }
  modalStatus.textContent = "Saved.";
  await loadUsers();
};

const handleDelete = async () => {
  if (!selectedUser) return;
  if (!confirm(`Delete ${selectedUser.username}?`)) return;
  modalStatus.textContent = "Deleting...";
  const res = await fetch(window.apiUrl(`/admin/users/${encodeURIComponent(selectedUser.username)}`), {
    method: "DELETE",
  });
  const data = await res.json();
  if (data.error) {
    modalStatus.textContent = data.error;
    return;
  }
  modalStatus.textContent = "Deleted.";
  await loadUsers();
  closeModal();
};

const handleViewLogs = async () => {
  if (!selectedUser) return;
  await loadLogs(selectedUser.username);
  closeModal();
  document.getElementById("logs").scrollIntoView({ behavior: "smooth" });
};

const handleLogout = (event) => {
  event.preventDefault();
  localStorage.removeItem("auth_user");
  localStorage.removeItem("auth_role");
  window.location.href = "landing.html";
};

createBtn.addEventListener("click", handleCreate);
logoutBtn.addEventListener("click", handleLogout);
closeModalBtn.addEventListener("click", closeModal);
saveUserBtn.addEventListener("click", handleUpdate);
deleteUserBtn.addEventListener("click", handleDelete);
viewLogsBtn.addEventListener("click", handleViewLogs);
closeLogBtn.addEventListener("click", closeLogModal);
profileBtn.addEventListener("click", () => {
  openProfile();
  loadProfile();
});
closeProfileBtn.addEventListener("click", closeProfile);
profileHistoryBtn.addEventListener("click", loadUserLogs);

if (userSearchEl) {
  userSearchEl.addEventListener("input", applyUserFilter);
}
if (logSearchEl) {
  logSearchEl.addEventListener("input", applyLogFilter);
}

if (adminUsernameEl) {
  adminUsernameEl.textContent = sessionUser || "";
}

(async () => {
  await loadStats();
  await loadUsers();
  await loadLogs();
})();


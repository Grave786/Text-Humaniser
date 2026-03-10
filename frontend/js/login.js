const usernameInput = document.getElementById("username");
const passwordInput = document.getElementById("password");
const errorEl = document.getElementById("error");
const loginBtn = document.getElementById("login");

const login = async () => {
  const username = usernameInput.value.trim();
  const password = passwordInput.value.trim();
  if (!username || !password) {
    errorEl.textContent = "Enter username and password.";
    return;
  }

  loginBtn.disabled = true;
  errorEl.textContent = "";
  try {
    const res = await fetch(window.apiUrl("/auth/login"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    const data = await res.json();
    if (data.error) {
      throw new Error(data.error);
    }
    localStorage.setItem("auth_user", data.username);
    localStorage.setItem("auth_role", data.role || "user");
    const target = data.role === "admin" ? "admin.html" : "humanizer.html";

    if (window.parent && window.parent !== window) {
      window.parent.postMessage(
        { type: "login-success", username: data.username, role: data.role || "user" },
        "*"
      );
      try {
        window.parent.location.href = target;
      } catch {
        // Parent navigation blocked; fallback to message only.
      }
    } else {
      window.location.href = target;
    }
  } catch (err) {
    errorEl.textContent = err.message || "Login failed.";
  } finally {
    loginBtn.disabled = false;
  }
};

loginBtn.addEventListener("click", login);

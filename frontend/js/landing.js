const loginModal = document.getElementById("login-modal");
const openLoginBtn = document.getElementById("open-login");
const openLoginCta = document.getElementById("open-login-cta");
const openLoginCta2 = document.getElementById("open-login-cta-2");
const closeLoginBtn = document.getElementById("close-login");
const openHumanizerBtn = document.getElementById("open-humanizer");

const openLogin = (event) => {
  if (event) {
    event.preventDefault();
  }
  loginModal.classList.remove("hidden");
  loginModal.classList.add("flex");
};

const closeLogin = () => {
  loginModal.classList.add("hidden");
  loginModal.classList.remove("flex");
};

const openHumanizer = () => {
  const session = localStorage.getItem("auth_user");
  const role = localStorage.getItem("auth_role");
  if (!session) {
    openLogin();
    return;
  }
  if (role === "admin") {
    window.location.href = "admin.html";
  } else {
    window.location.href = "humanizer.html";
  }
};

window.addEventListener("message", (event) => {
  if (!event.data || event.data.type !== "login-success") {
    return;
  }
  localStorage.setItem("auth_user", event.data.username);
  localStorage.setItem("auth_role", event.data.role || "user");
  closeLogin();
  if (event.data.role === "admin") {
    window.location.href = "admin.html";
  } else {
    window.location.href = "humanizer.html";
  }
});

openLoginBtn.addEventListener("click", openLogin);
openLoginCta.addEventListener("click", openLogin);
if (openLoginCta2) {
  openLoginCta2.addEventListener("click", openLogin);
}
closeLoginBtn.addEventListener("click", closeLogin);
openHumanizerBtn.addEventListener("click", openHumanizer);

window.FRONTEND_CONFIG = {
  API_BASE_DEFAULT: "http://127.0.0.1:8000",
};

window.apiUrl = (path) => {
  const base = localStorage.getItem("api_base") || window.FRONTEND_CONFIG.API_BASE_DEFAULT;
  return `${base.replace(/\/$/, "")}${path}`;
};

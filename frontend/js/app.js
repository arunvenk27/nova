// Entry point: menu, routing between pages, dark-mode toggle.
import { disposeCharts } from "./ui.js";
import * as overview from "./pages/overview.js";
import * as explorer from "./pages/explorer.js";
import * as monitor from "./pages/monitor.js";
import * as eis from "./pages/eis.js";
import * as models from "./pages/models.js";
import * as about from "./pages/about.js";
import * as upload from "./pages/upload.js";

const PAGES = { overview, explorer, monitor, eis, models, upload, about };
const root = document.getElementById("page");
const sidebar = document.getElementById("sidebar");

function currentPage() {
  const name = (location.hash.replace(/^#\//, "").split("?")[0]) || "overview";
  return PAGES[name] ? name : "overview";
}

async function render() {
  const name = currentPage();
  document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === name));
  sidebar.classList.remove("open");
  document.querySelector(".sim-banner").hidden = name === "upload";   // that page shows the user's own data
  document.getElementById("menuToggle").setAttribute("aria-expanded", "false");
  disposeCharts();
  root.style.animation = "none";
  void root.offsetWidth;                 // restart the fade-in animation
  root.style.animation = "";
  try {
    await PAGES[name].render(root);
  } catch (err) {
    console.error(err);
    root.innerHTML = `<div class="error"><b>Something went wrong on this page.</b><br>${err.message}</div>`;
  }
}

// ---------------------------------------------------------------- theme
const themeBtn = document.getElementById("themeToggle");
function syncThemeLabel() {
  const dark = getComputedStyle(document.documentElement).getPropertyValue("color-scheme").trim() === "dark";
  themeBtn.querySelector("span").textContent = dark ? "Light mode" : "Dark mode";
}
themeBtn.addEventListener("click", () => {
  const dark = getComputedStyle(document.documentElement).getPropertyValue("color-scheme").trim() === "dark";
  const next = dark ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("ps-theme", next); } catch (e) { /* private mode */ }
  syncThemeLabel();
  render();                               // redraw charts with the new colours
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { syncThemeLabel(); render(); });

// ---------------------------------------------------------------- mobile menu
document.getElementById("menuToggle").addEventListener("click", (e) => {
  const open = sidebar.classList.toggle("open");
  e.currentTarget.setAttribute("aria-expanded", String(open));
});

window.addEventListener("hashchange", render);
syncThemeLabel();
render();

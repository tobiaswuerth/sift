// sift ui: keyword search as you type, keys to move through results, and the reader scrolled to the hit.
const box = () => document.getElementById("q");
let pending;

document.addEventListener("input", (e) => {
  if (e.target.id !== "q") return;
  const mode = document.querySelector("[name=mode]:checked");
  if (mode && mode.value !== "search") return; // meaning and hybrid search take longer: they run on Enter
  clearTimeout(pending);
  pending = setTimeout(() => htmx.trigger("#search", "search"), 200);
});

document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return; // the browser's own shortcuts, such as Ctrl+K
  const typing = e.target.matches("input, select, textarea");
  if (e.key === "/" && !typing) {
    e.preventDefault();
    box().focus();
    box().select();
    return;
  }
  if (e.key === "Escape" && typing) return e.target.blur();
  if (typing && !(e.target === box() && e.key === "ArrowDown")) return; // from the box down into the results
  const hits = [...document.querySelectorAll("a.hit")];
  const at = hits.indexOf(document.activeElement);
  const step = { ArrowDown: 1, j: 1, ArrowUp: -1, k: -1 }[e.key];
  if (!hits.length || !step) return;
  e.preventDefault();
  const next = hits[Math.max(0, Math.min(hits.length - 1, at + step))];
  next.focus();
  next.scrollIntoView({ block: "nearest" });
});

document.addEventListener("htmx:configRequest", (e) => {
  // Empty fields stay out of the address, which the server reads with the same defaults.
  const form = e.detail.formData;
  if (form) [...form.entries()].filter(([, v]) => v === "").forEach(([k]) => form.delete(k));
});

document.addEventListener("click", (e) => {
  const hit = e.target.closest("a.hit");
  if (!hit) return;
  document.querySelectorAll("a.hit.on").forEach((a) => a.classList.remove("on"));
  hit.classList.add("on");
});

function showHit() {
  const reader = document.getElementById("reader");
  const line = reader?.querySelector(".l.hit");
  if (line && line.offsetTop > reader.clientHeight / 2) line.scrollIntoView({ block: "center" });
  else reader?.scrollTo(0, 0); // a hit near the top stays below the title
}
document.addEventListener("DOMContentLoaded", showHit);
document.addEventListener("htmx:afterSettle", (e) => {
  if (e.target.id === "reader") showHit();
});

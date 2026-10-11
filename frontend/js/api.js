// Frontend -> backend communication. Every request goes through here, so the
// rest of the frontend never needs to know URLs or handle HTTP details.
const cache = new Map();

async function get(path) {
  if (cache.has(path)) return cache.get(path);
  const p = fetch(path).then(async (r) => {
    if (!r.ok) {
      let msg = `${r.status} ${r.statusText}`;
      try { msg = (await r.json()).detail || msg; } catch (e) { /* not JSON */ }
      throw new Error(msg);
    }
    return r.json();
  });
  cache.set(path, p);
  p.catch(() => cache.delete(path));   // do not cache failures
  return p;
}

const enc = encodeURIComponent;

// POST (never cached): used by the "Your Data" page
async function post(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch (e) { /* not JSON */ }
    throw new Error(msg);
  }
  return r.json();
}

export const api = {
  overview: () => get("/api/overview"),
  species: () => get("/api/species"),
  plants: (crop) => get(`/api/species/${enc(crop)}/plants`),
  timeline: (crop, plant) => get(`/api/species/${enc(crop)}/plants/${enc(plant)}/timeline`),
  eis: (crop, plant, day) =>
    get(`/api/species/${enc(crop)}/plants/${enc(plant)}/eis${day == null ? "" : `?day=${day}`}`),
  correlations: (crop, feature) => get(`/api/species/${enc(crop)}/correlations?feature=${enc(feature)}`),
  models: () => get("/api/models"),
  status: () => get("/api/status"),
  analyzeUpload: (crop, csv) => post("/api/upload/analyze", { crop, csv }),
};

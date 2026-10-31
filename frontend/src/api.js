const BASE = import.meta.env.VITE_API_URL || "";

async function handle(r) {
  if (!r.ok) {
    let m = r.statusText;
    try { m = (await r.json()).detail || m; } catch {}
    throw new Error(m);
  }
  return r.json();
}

export const postForm = (path, fd) => fetch(BASE + path, { method: "POST", body: fd }).then(handle);
export const getJSON = (path) => fetch(BASE + path).then(handle);
export const asset = (p) => BASE + p;

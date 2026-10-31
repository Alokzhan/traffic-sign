const BASE = import.meta.env.VITE_API_URL || (import.meta.env.DEV ? "http://127.0.0.1:8000" : "https://traffic-sign-nvpc.onrender.com");

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

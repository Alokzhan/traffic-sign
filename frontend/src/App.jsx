import { useState, useEffect } from "react";
import { postForm, getJSON, asset } from "./api";

const DISCLAIMER =
  "Educational demo only. Model is NOT suitable for real driving or any safety-critical use.";

/* ---------- small shared pieces ---------- */
function DropZone({ onFiles, multiple = false, label }) {
  const [over, setOver] = useState(false);
  return (
    <label
      className={"drop" + (over ? " over" : "")}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault(); setOver(false);
        onFiles([...e.dataTransfer.files].filter((f) => f.type.startsWith("image/")));
      }}
    >
      <input
        type="file" accept="image/*" multiple={multiple} hidden
        onChange={(e) => { onFiles([...e.target.files]); e.target.value = ""; }}
      />
      <span>📁 {label}</span>
      <small>drag &amp; drop ya click karein (mobile par camera bhi chalega)</small>
    </label>
  );
}

function Bars({ classes, probs, top }) {
  const [showAll, setShowAll] = useState(false);
  if (!classes || !probs) return null;
  
  const indexed = probs.map((p, i) => ({ c: classes[i], p, i })).sort((a, b) => b.p - a.p);
  const visible = showAll || classes.length <= 10 ? indexed : indexed.slice(0, 8);

  return (
    <div className="bars-container">
      <div className="bars">
        {visible.map(({ c, p, i }) => (
          <div className="bar-row" key={c}>
            <span className="bar-label" title={c}>{c}</span>
            <div className="bar-track">
              <div
                className={"bar-fill" + (i === top ? (c.startsWith("Other") ? " other" : " top") : "")}
                style={{
                  width: `${(p * 100).toFixed(1)}%`,
                  backgroundColor: i === top && c.startsWith("Other") ? "#f59e0b" : undefined
                }}
              />
            </div>
            <span className="bar-val">{(p * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
      {classes.length > 10 && (
        <button
          type="button"
          onClick={() => setShowAll(!showAll)}
          style={{ marginTop: "10px", fontSize: "0.8rem", padding: "4px 12px" }}
        >
          {showAll ? "Show Top 8 Only" : `Show All ${classes.length} Classes`}
        </button>
      )}
    </div>
  );
}

const isUncertain = (conf, ent, s) => conf < s.thr || ent > s.ent;

function Verdict({ name, conf, ent, s, isOther }) {
  const isOtherClass = isOther || name.startsWith("Other");
  const bad = isUncertain(conf, ent, s);

  if (isOtherClass) {
    return (
      <div className="verdict" style={{ background: "rgba(245, 158, 11, 0.12)", border: "1px solid #f59e0b", color: "#d97706" }}>
        🛑 <b>Not one of the supported traffic signs</b> — classified as <i>{name}</i> with {(conf * 100).toFixed(1)}% confidence (uncertainty: {ent.toFixed(2)})
      </div>
    );
  }

  return bad ? (
    <div className="verdict warn">
      ⚠️ <b>UNCERTAIN</b> — best guess: {name} ({(conf * 100).toFixed(1)}%, uncertainty {ent.toFixed(2)}). Ho sakta hai
      ye sign model ke supported classes mein hai hi nahi.
    </div>
  ) : (
    <div className="verdict ok">
      ✅ <b>{name}</b> — confidence {(conf * 100).toFixed(1)}%, uncertainty {ent.toFixed(2)}
    </div>
  );
}

function Busy({ on }) { return on ? <div className="busy">Processing…</div> : null; }
function Err({ msg }) { return msg ? <div className="verdict warn">❌ {msg}</div> : null; }

function NlpAdvisoryCard({ advisory, narrative, hasMl, deepConf, mlConf }) {
  if (!advisory) return null;
  const sevColors = {
    CRITICAL: { bg: "rgba(239, 68, 68, 0.15)", border: "#ef4444", text: "#f87171" },
    HIGH: { bg: "rgba(245, 158, 11, 0.15)", border: "#f59e0b", text: "#fbbf24" },
    MODERATE: { bg: "rgba(59, 130, 246, 0.15)", border: "#3b82f6", text: "#60a5fa" },
    INFORMATIONAL: { bg: "rgba(16, 185, 129, 0.15)", border: "#10b981", text: "#34d399" }
  };
  const sc = sevColors[advisory.severity] || sevColors.MODERATE;

  return (
    <div style={{
      background: "var(--glass-b)",
      border: "1px solid var(--border)",
      borderRadius: "var(--radius)",
      padding: "16px",
      marginTop: "16px",
      boxShadow: "var(--shadow)"
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px", flexWrap: "wrap", gap: "8px" }}>
        <h4 style={{ margin: 0, fontSize: "0.95rem", display: "flex", alignItems: "center", gap: "8px" }}>
          💬 <b>NLP Autonomous Driving Advisory</b>
        </h4>
        <div style={{ display: "flex", gap: "8px" }}>
          <span style={{
            fontSize: "0.75rem", padding: "3px 10px", borderRadius: "99px",
            background: sc.bg, border: `1px solid ${sc.border}`, color: sc.text, fontWeight: 700
          }}>
            {advisory.severity}
          </span>
          <span style={{
            fontSize: "0.75rem", padding: "3px 10px", borderRadius: "99px",
            background: "rgba(255,255,255,0.06)", border: "1px solid var(--border)", color: "var(--text2)"
          }}>
            {advisory.category}
          </span>
        </div>
      </div>

      <div style={{ fontSize: "0.88rem", color: "var(--text)", lineHeight: 1.5, marginBottom: "8px" }}>
        <b>Action (EN):</b> {advisory.action_en}
      </div>

      <div style={{ fontSize: "0.85rem", color: "var(--text2)", lineHeight: 1.5, marginBottom: "10px", background: "rgba(255,255,255,0.02)", padding: "8px 12px", borderRadius: "8px" }}>
        🇮🇳 <b>Action (Hindi/Hinglish):</b> {advisory.action_hi}
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.78rem", color: "var(--text3)", flexWrap: "wrap", gap: "6px" }}>
        <span>⚖️ Legal Code: <b>{advisory.legal_rule}</b></span>
        {hasMl && mlConf !== undefined && mlConf !== null && (
          <span style={{ color: "var(--cyan)", fontWeight: 600 }}>
            🤖 Hybrid Ensemble: Deep {(deepConf * 100).toFixed(0)}% + Calibrated ML {(mlConf * 100).toFixed(0)}%
          </span>
        )}
      </div>
    </div>
  );
}

/* ---------- Predict ---------- */
function Predict({ s }) {
  const [preview, setPreview] = useState(null);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function onFiles([f]) {
    if (!f) return;
    setPreview(URL.createObjectURL(f)); setRes(null); setErr(""); setBusy(true);
    try {
      const fd = new FormData(); fd.append("file", f);
      setRes(await postForm("/api/predict", fd));
    } catch (e) {
      if (e.message && e.message.includes("Failed to fetch")) {
        setErr("Failed to fetch: Backend server (http://127.0.0.1:8000) se connect nahi ho pa raha hai. Kripya check karein ki FastAPI server terminal me running hai.");
      } else {
        setErr(e.message || "Prediction failed");
      }
    } finally { setBusy(false); }
  }

  return (
    <section>
      <DropZone onFiles={onFiles} label="Traffic sign ya image chunein" />
      <Busy on={busy} /><Err msg={err} />
      {res && (
        <>
          <Verdict name={res.name} conf={res.conf} ent={res.entropy} s={s} isOther={res.is_other} />
          <NlpAdvisoryCard
            advisory={res.advisory}
            narrative={res.narrative}
            hasMl={res.has_ml_ensemble}
            deepConf={res.deep_conf}
            mlConf={res.ml_conf}
          />
          <div className="grid3" style={{ marginTop: "18px" }}>
            <figure><img src={preview} alt="original" /><figcaption>Original</figcaption></figure>
            <figure><img src={res.model_input} alt="input" /><figcaption>Model input (48×48 + CLAHE)</figcaption></figure>
            {res.gradcam && (
              <figure><img src={res.gradcam} alt="gradcam" /><figcaption>Grad-CAM: model kahan dekh raha hai</figcaption></figure>
            )}
          </div>
          <h3>Class probabilities</h3>
          <Bars classes={res.classes} probs={res.probs} top={res.top} />
        </>
      )}
    </section>
  );
}

/* ---------- Robustness ---------- */
function LineChart({ items }) {
  const W = 560, H = 190, P = 28;
  const x = (i) => P + (i * (W - 2 * P)) / (items.length - 1);
  const y = (v) => H - P - v * (H - 2 * P);
  const line = (key) => items.map((d, i) => `${x(i)},${y(d[key])}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart">
      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <line x1={P} x2={W - P} y1={y(v)} y2={y(v)} className="grid" />
          <text x={4} y={y(v) + 4} className="axis">{Math.round(v * 100)}%</text>
        </g>
      ))}
      <polyline points={line("conf")} className="l-top" fill="none" />
      <polyline points={line("base_conf")} className="l-base" fill="none" />
      {items.map((d, i) => (
        <text key={i} x={x(i)} y={H - 8} textAnchor="middle" className="axis">{d.level}</text>
      ))}
    </svg>
  );
}

function Robustness() {
  const [kinds, setKinds] = useState([]);
  const [kind, setKind] = useState("brightness");
  const [file, setFile] = useState(null);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => { getJSON("/api/perturbations").then(setKinds).catch((e) => setErr(e.message)); }, []);

  async function run(f, k) {
    if (!f) return;
    setBusy(true); setErr("");
    try {
      const fd = new FormData(); fd.append("file", f); fd.append("kind", k);
      setRes(await postForm("/api/robustness", fd));
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  }

  const flips = res ? res.items.filter((d) => d.pred !== res.base).length : 0;

  return (
    <section>
      <p className="muted">Image ko jaan-bujhkar kharab karke dekhein ki model kab fail hota hai.</p>
      <div className="row">
        <select value={kind} onChange={(e) => { setKind(e.target.value); run(file, e.target.value); }}>
          {kinds.map((k) => <option key={k.key} value={k.key}>{k.label}</option>)}
        </select>
      </div>
      <DropZone onFiles={([f]) => { if (f) { setFile(f); run(f, kind); } }} label="Stress-test ke liye image chunein" />
      <Busy on={busy} /><Err msg={err} />
      {res && (
        <>
          <div className="verdict ok">
            Original prediction: <b>{res.base}</b> — {flips}/{res.items.length} levels par prediction badal gayi.
          </div>
          <div className="strip">
            {res.items.map((d) => (
              <figure key={d.level} className={d.pred !== res.base ? "flip" : ""}>
                <img src={d.image} alt={String(d.level)} />
                <figcaption><b>{d.level}</b><br />{d.pred}<br />{(d.conf * 100).toFixed(0)}%</figcaption>
              </figure>
            ))}
          </div>
          <LineChart items={res.items} />
          <div className="legend"><span className="dot top" /> top confidence <span className="dot base" /> confidence in original class</div>
        </>
      )}
    </section>
  );
}

/* ---------- Batch ---------- */
function Batch({ s }) {
  const [rows, setRows] = useState([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function onFiles(files) {
    if (!files.length) return;
    setBusy(true); setErr("");
    try {
      const fd = new FormData(); files.forEach((f) => fd.append("files", f));
      setRows((await postForm("/api/batch", fd)).rows);
    } catch (e) { setErr(e.message); } finally { setBusy(false); }
  }

  function download() {
    const head = "file,prediction,confidence,uncertainty,status\n";
    const body = rows.map((r) =>
      [r.file, r.prediction, r.confidence.toFixed(4), r.entropy.toFixed(3),
       r.is_other ? "NOT_A_SIGN" : (isUncertain(r.confidence, r.entropy, s) ? "UNCERTAIN" : "OK")].join(",")).join("\n");
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([head + body], { type: "text/csv" }));
    a.download = "predictions.csv"; a.click();
  }

  return (
    <section>
      <DropZone multiple onFiles={onFiles} label="Kai images ek saath chunein" />
      <Busy on={busy} /><Err msg={err} />
      {rows.length > 0 && (
        <>
          <table>
            <thead><tr><th>File</th><th>Prediction</th><th>Confidence</th><th>Uncertainty</th><th>Status</th></tr></thead>
            <tbody>
              {rows.map((r, i) => {
                const isOther = r.is_other || r.prediction.startsWith("Other");
                const bad = isUncertain(r.confidence, r.entropy, s);
                const statusText = isOther ? "NOT A SIGN" : (bad ? "UNCERTAIN" : "OK");
                const statusClass = isOther ? "warn" : (bad ? "bad" : "good");
                return (
                  <tr key={i}>
                    <td>{r.file}</td><td>{r.prediction}</td>
                    <td>{(r.confidence * 100).toFixed(1)}%</td><td>{r.entropy.toFixed(2)}</td>
                    <td className={statusClass}><b>{statusText}</b></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <button onClick={download}>⬇️ CSV download</button>
        </>
      )}
    </section>
  );
}

/* ---------- Report ---------- */
function Report() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => { getJSON("/api/report").then(setData).catch((e) => setErr(e.message)); }, []);
  if (err) return <Err msg={err} />;
  if (!data) return <Busy on />;
  const { metrics: m, classes, images } = data;
  return (
    <section>
      <div className="cards">
        <div className="card"><span>Held-out test accuracy</span><b>{(m.test_accuracy * 100).toFixed(2)}%</b></div>
        <div className="card"><span>Macro F1</span><b>{m.report["macro avg"]["f1-score"].toFixed(3)}</b></div>
        <div className="card"><span>Total Classes</span><b>{classes.length}</b></div>
      </div>
      <table>
        <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
        <tbody>
          {classes.map((c) => (
            <tr key={c} style={c.startsWith("Other") ? { fontWeight: 600, background: "rgba(245, 158, 11, 0.08)" } : {}}>
              <td>{c}</td>
              <td>{m.report[c] ? m.report[c].precision.toFixed(3) : "—"}</td>
              <td>{m.report[c] ? m.report[c].recall.toFixed(3) : "—"}</td>
              <td>{m.report[c] ? m.report[c]["f1-score"].toFixed(3) : "—"}</td>
              <td>{m.report[c] ? m.report[c].support : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted">8 GTSRB sign classes + 1 'Other / not a sign' open-set negative class par evaluate hua hai.</p>
      <div className="gallery">
        {images.map((n) => (
          <figure key={n}>
            <img className="plot" src={asset("/outputs/" + n)} alt={n} />
            <figcaption>{n.replace(".png", "").replace("_", " ")}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}

/* ---------- About ---------- */
function About() {
  const flow = ["User image", "Preprocess (resize + CLAHE)", "CNN 9-class (final.keras)", "Softmax probabilities",
    "Open-set Reject / Other verdict", "Grad-CAM explanation", "React UI"];
  return (
    <section>
      <h3>System architecture (inference path)</h3>
      <div className="flow">
        {flow.map((f, i) => (<span key={f} className="node">{f}{i < flow.length - 1 && <i>→</i>}</span>))}
      </div>
      <p>React (Vite) UI → FastAPI (<code>api.py</code>) → Keras 9-class model (8 traffic signs + 1 "Other / not a sign").</p>
      <div className="verdict warn">{DISCLAIMER}</div>
    </section>
  );
}

/* ---------- NLP Assistant ---------- */
function NlpAssistant() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [busy, setBusy] = useState(false);

  const chips = [
    "School children",
    "Speed limit 50",
    "Red traffic light",
    "Give way / Yield",
    "Roundabout",
    "No overtaking",
    "Slippery snow",
    "Turn right"
  ];

  async function search(q) {
    const qStr = q || query;
    if (!qStr.trim()) return;
    setQuery(qStr);
    setBusy(true);
    try {
      const res = await getJSON(`/api/nlp/search?q=${encodeURIComponent(qStr)}`);
      setResults(res.results || []);
    } catch (e) {
      console.error(e);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <h3>💬 NLP Natural Language Sign Assistant</h3>
      <p style={{ color: "var(--text2)", fontSize: "0.88rem", marginBottom: "16px" }}>
        Type any natural language scenario, driver observation, or traffic situation. Our NLP semantic engine will match the official sign, autonomous driving actions, and legal codes.
      </p>

      <div style={{ display: "flex", gap: "10px", marginBottom: "12px" }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && search(query)}
          placeholder="e.g. school zone, turn right, red light, icy road, highway speed..."
          style={{
            flex: 1, padding: "12px 16px", borderRadius: "var(--radius)",
            background: "var(--bg3)", border: "1px solid var(--border)", color: "var(--text)", fontSize: "0.9rem"
          }}
        />
        <button type="button" onClick={() => search(query)} style={{ padding: "12px 20px" }}>
          🔍 Search
        </button>
      </div>

      <div style={{ display: "flex", gap: "6px", flexWrap: "wrap", marginBottom: "20px" }}>
        <span style={{ fontSize: "0.78rem", color: "var(--text3)", alignSelf: "center" }}>Quick queries:</span>
        {chips.map((c) => (
          <button
            key={c}
            type="button"
            onClick={() => search(c)}
            style={{ fontSize: "0.78rem", padding: "4px 10px", borderRadius: "99px" }}
          >
            {c}
          </button>
        ))}
      </div>

      {busy && <div className="busy">Analyzing natural language semantics…</div>}

      <div style={{ display: "grid", gap: "12px" }}>
        {results.map((r) => (
          <div
            key={r.name}
            style={{
              background: "var(--glass-b)", border: "1px solid var(--border)",
              borderRadius: "var(--radius)", padding: "16px 20px", boxShadow: "var(--shadow)"
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px", flexWrap: "wrap", gap: "8px" }}>
              <b style={{ fontSize: "1.05rem", color: "var(--text)" }}>{r.name}</b>
              <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                <span style={{
                  fontSize: "0.75rem", padding: "2px 8px", borderRadius: "99px",
                  background: "rgba(99,102,241,0.15)", border: "1px solid #6366f1", color: "#a5b4fc"
                }}>
                  {r.category}
                </span>
                <span style={{ fontSize: "0.78rem", color: "var(--cyan)", fontWeight: 600 }}>
                  {Math.round((r.relevance_score || 0) * 100)}% NLP match
                </span>
              </div>
            </div>
            <p style={{ fontSize: "0.88rem", color: "var(--text)", marginBottom: "6px" }}>
              <b>Autonomous Action:</b> {r.action_en}
            </p>
            <p style={{ fontSize: "0.85rem", color: "var(--text2)", marginBottom: "8px", background: "rgba(255,255,255,0.02)", padding: "6px 10px", borderRadius: "6px" }}>
              🇮🇳 <b>Hindi Advisory:</b> {r.action_hi}
            </p>
            <div style={{ fontSize: "0.75rem", color: "var(--text3)" }}>
              ⚖️ Rule: {r.legal_rule} · Severity: <b>{r.severity}</b>
            </div>
          </div>
        ))}
        {!busy && query && results.length === 0 && (
          <p style={{ color: "var(--text3)", fontStyle: "italic" }}>No matching traffic signs found. Try different keywords.</p>
        )}
      </div>
    </section>
  );
}

/* ---------- App ---------- */
const TABS = [
  ["predict", "🔍 Predict + Explain"],
  ["nlp", "💬 NLP Driver Assistant"],
  ["robust", "🧪 Robustness Lab"],
  ["batch", "📦 Batch"],
  ["report", "📊 Model Report"],
  ["about", "🏗 Architecture"],
];

export default function App() {
  const [tab, setTab] = useState("predict");
  const [s, setS] = useState({ thr: 0.7, ent: 0.5 });
  const [online, setOnline] = useState(null);
  const [classes, setClasses] = useState([]);

  useEffect(() => {
    getJSON("/api/health")
      .then((d) => {
        setOnline(true);
        if (d.classes) setClasses(d.classes);
      })
      .catch(() => setOnline(false));
  }, []);

  return (
    <div className="app">
      <header>
        <div>
          <h1>🚦 Traffic Sign AI Lab</h1>
          <p style={{color:"var(--text3)",fontSize:".82rem",marginTop:"2px",fontWeight:500}}>
            Deep Residual CNN + Calibrated ML + NLP Semantic Advisory — 44 Classes (43 GTSRB Signs + 1 Open-Set Other)
          </p>
        </div>
        <span className={"pill " + (online ? "on" : "off")}>
          {online === null ? "⏳ connecting…" : online ? "API online" : "⚠️ API offline — start uvicorn first"}
        </span>
      </header>
      <div className="layout">
        <aside>
          <h4>Reject option</h4>
          <label>Min confidence: <b>{s.thr.toFixed(2)}</b>
            <input type="range" min="0.3" max="0.99" step="0.01" value={s.thr} onChange={(e) => setS({ ...s, thr: +e.target.value })} />
          </label>
          <label>Max uncertainty: <b>{s.ent.toFixed(2)}</b>
            <input type="range" min="0.1" max="1" step="0.05" value={s.ent} onChange={(e) => setS({ ...s, ent: +e.target.value })} />
          </label>
          <p className="muted small">Confidence kam ya entropy zyada ho to prediction UNCERTAIN maani jati hai.</p>
          <div className="verdict warn small">{DISCLAIMER}</div>
        </aside>
        <main>
          <nav>{TABS.map(([k, l]) => (<button key={k} className={tab === k ? "active" : ""} onClick={() => setTab(k)}>{l}</button>))}</nav>
          {tab === "predict" && <Predict s={s} />}
          {tab === "nlp" && <NlpAssistant />}
          {tab === "robust" && <Robustness />}
          {tab === "batch" && <Batch s={s} />}
          {tab === "report" && <Report />}
          {tab === "about" && <About />}
        </main>
      </div>
    </div>
  );
}

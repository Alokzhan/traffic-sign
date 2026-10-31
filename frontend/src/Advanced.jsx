import { useState, useEffect, useRef } from "react";
import { postForm, getJSON, asset } from "./api";

const isUnc = (conf, ent, s) => conf < s.thr || ent > s.ent;
const entropyOf = (p) => -p.reduce((a, v) => a + v * Math.log(v + 1e-9), 0) / Math.log(p.length);

function MiniBars({ classes, probs, top }) {
  return (
    <div className="bars">
      {classes.map((c, i) => (
        <div className="bar-row" key={c}>
          <span className="bar-label">{c}</span>
          <div className="bar-track"><div className={"bar-fill" + (i === top ? " top" : "")} style={{ width: `${(probs[i] * 100).toFixed(1)}%` }} /></div>
          <span className="bar-val">{(probs[i] * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

/* generic SVG line chart: series = [{pts:[[x,y],...], cls, dots}] , x in [xmin,xmax], y in [0,1] */
function Chart({ series, xmin = 0, xmax = 1, marker, xlabel }) {
  const W = 560, H = 210, P = 32;
  const X = (v) => P + ((v - xmin) / (xmax - xmin)) * (W - 2 * P);
  const Y = (v) => H - P - v * (H - 2 * P);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="chart">
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}><line x1={P} x2={W - P} y1={Y(v)} y2={Y(v)} className="grid" />
          <text x={2} y={Y(v) + 4} className="axis">{Math.round(v * 100)}%</text></g>
      ))}
      {[0, 0.5, 1].map((t) => { const v = xmin + t * (xmax - xmin);
        return <text key={t} x={X(v)} y={H - 12} textAnchor="middle" className="axis">{v.toFixed(2)}</text>; })}
      {marker != null && <line x1={X(marker)} x2={X(marker)} y1={P - 8} y2={H - P} className="marker" />}
      {series.map((s, i) => (
        <g key={i}>
          <polyline points={s.pts.map(([x, y]) => `${X(x)},${Y(y)}`).join(" ")} className={"line " + s.cls} fill="none" />
          {s.dots && s.pts.map(([x, y], j) => <circle key={j} cx={X(x)} cy={Y(y)} r="4" className={"dotc " + s.cls} />)}
        </g>
      ))}
      {xlabel && <text x={W / 2} y={H - 1} textAnchor="middle" className="axis">{xlabel}</text>}
    </svg>
  );
}

/* ---------- Live camera (temporal smoothing) ---------- */
export function Live({ s }) {
  const videoRef = useRef(null), streamRef = useRef(null), busy = useRef(false), hist = useRef([]);
  const [on, setOn] = useState(false);
  const [err, setErr] = useState("");
  const [cur, setCur] = useState(null);

  async function start() {
    setErr("");
    try {
      const st = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
      streamRef.current = st; videoRef.current.srcObject = st; await videoRef.current.play();
      hist.current = []; setOn(true);
    } catch (e) { setErr("Camera nahi khula: " + e.message); }
  }
  function stop() { streamRef.current?.getTracks().forEach((t) => t.stop()); setOn(false); setCur(null); }
  useEffect(() => () => streamRef.current?.getTracks().forEach((t) => t.stop()), []);

  useEffect(() => {
    if (!on) return;
    const canvas = document.createElement("canvas"); canvas.width = canvas.height = 224;
    const id = setInterval(async () => {
      const v = videoRef.current;
      if (busy.current || !v || !v.videoWidth) return;
      busy.current = true;
      try {
        const side = Math.min(v.videoWidth, v.videoHeight) * 0.6;   // centre box ka crop
        canvas.getContext("2d").drawImage(v, (v.videoWidth - side) / 2, (v.videoHeight - side) / 2, side, side, 0, 0, 224, 224);
        const blob = await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.9));
        const fd = new FormData(); fd.append("file", blob, "frame.jpg");
        const r = await postForm("/api/predict?cam=0", fd);
        hist.current = [...hist.current.slice(-4), r.probs];               // last 5 frames
        const n = hist.current.length;
        const avg = r.probs.map((_, i) => hist.current.reduce((a, p) => a + p[i], 0) / n);
        const top = avg.indexOf(Math.max(...avg));
        setCur({ classes: r.classes, probs: avg, top, conf: avg[top], ent: entropyOf(avg), n, input: r.model_input });
      } catch (e) { setErr(e.message); } finally { busy.current = false; }
    }, 700);
    return () => clearInterval(id);
  }, [on]);

  const bad = cur && isUnc(cur.conf, cur.ent, s);
  return (
    <section>
      <p className="muted">Sign ko camera ke beech wale box mein rakhein. Prediction last 5 frames ka average hai (temporal smoothing), isliye flicker kam hota hai.</p>
      <div className="row">
        {!on ? <button className="active" onClick={start}>▶ Camera start</button> : <button onClick={stop}>⏹ Stop</button>}
      </div>
      {err && <div className="verdict warn">❌ {err}</div>}
      <div className="live">
        <div className="vwrap">
          <video ref={videoRef} playsInline muted />
          {on && <div className="cropbox" />}
        </div>
        {cur && (
          <div className="live-side">
            <img src={cur.input} alt="model input" className="live-in" />
            <div className={"verdict " + (bad ? "warn" : "ok")}>
              {bad ? "⚠️ UNCERTAIN / no known sign" : "✅ " + cur.classes[cur.top]}
              <br /><small>{cur.classes[cur.top]} · {(cur.conf * 100).toFixed(1)}% · uncertainty {cur.ent.toFixed(2)} · {cur.n} frames</small>
            </div>
          </div>
        )}
      </div>
      {cur && <MiniBars classes={cur.classes} probs={cur.probs} top={cur.top} />}
    </section>
  );
}

function useAdvanced() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => { getJSON("/api/advanced").then(setD).catch((e) => setErr(e.message)); }, []);
  return [d, err];
}

const NeedEval = ({ msg }) => (
  <div className="verdict warn">
    ❌ {msg}<br />
    Terminal mein chalayein: <code>python evaluate_advanced.py --data "&lt;gtsrb folder ka path&gt;" --out outputs</code>, phir uvicorn restart karein.
  </div>
);

/* ---------- Safety Lab: calibration + unseen sign rejection ---------- */
export function Safety({ s }) {
  const [d, err] = useAdvanced();
  if (err) return <NeedEval msg={err} />;
  if (!d) return <div className="busy">Loading…</div>;
  const near = d.sweep.reduce((b, r) => (Math.abs(r.thr - s.thr) < Math.abs(b.thr - s.thr) ? r : b), d.sweep[0]);
  const safe = d.sweep.find((r) => r.ood_accept <= 0.05);
  return (
    <section>
      <p className="muted">Model 8 specific signs aur 1 open-set ('Other') class jaanta hai. Yahan dekhein ki unseen traffic signs aur non-sign images kitna reject hote hain
        ({d.n_ood} unseen-class test images vs {d.n_id} known-class images).</p>
      <div className="cards">
        <div className="card"><span>Known-class accuracy</span><b>{(d.id_accuracy * 100).toFixed(2)}%</b></div>
        <div className="card"><span>Calibration error (ECE)</span><b>{d.ece.toFixed(4)}</b></div>
        <div className="card"><span>Unseen-sign AUROC (confidence)</span><b>{d.auroc_conf.toFixed(3)}</b></div>
        <div className="card"><span>Unseen-sign AUROC (entropy)</span><b>{d.auroc_entropy.toFixed(3)}</b></div>
      </div>

      <h3>Threshold {s.thr.toFixed(2)} par kya hota hai</h3>
      <div className="cards">
        <div className="card"><span>Known signs accepted (coverage)</span><b>{(near.id_accept * 100).toFixed(1)}%</b></div>
        <div className="card"><span>Accuracy on accepted</span><b>{near.selective_acc == null ? "—" : (near.selective_acc * 100).toFixed(2) + "%"}</b></div>
        <div className="card"><span className="bad">Unseen signs wrongly accepted</span><b className="bad">{(near.ood_accept * 100).toFixed(1)}%</b></div>
      </div>
      <Chart
        marker={s.thr} xmin={0.3} xmax={1} xlabel="confidence threshold"
        series={[
          { pts: d.sweep.map((r) => [r.thr, r.id_accept]), cls: "l-top" },
          { pts: d.sweep.map((r) => [r.thr, r.ood_accept]), cls: "l-red" },
          { pts: d.sweep.filter((r) => r.selective_acc != null).map((r) => [r.thr, r.selective_acc]), cls: "l-green" },
        ]}
      />
      <div className="legend">
        <span className="dot top" /> known signs accepted <span className="dot red" /> unseen signs wrongly accepted <span className="dot green" /> accuracy on accepted
      </div>
      {safe && <p>💡 Sidebar ko <b>{safe.thr.toFixed(2)}</b> par rakhne se unseen signs ka accept hona ≤5% ho jata hai, aur known signs ka {(safe.id_accept * 100).toFixed(1)}% phir bhi accept hota hai.</p>}
      {!safe && <p className="muted">Is model ke saath unseen-sign accept rate kisi bhi threshold par 5% se neeche nahi gaya — ye model ki asli limitation hai.</p>}

      <h3>Reliability diagram (calibration)</h3>
      <p className="muted small">Dotted diagonal = perfectly calibrated. Diagonal se door = model jitna sure bolta hai utna sahi nahi hota (ya ulta).</p>
      <Chart xlabel="predicted confidence → actual accuracy" series={[
        { pts: [[0, 0], [1, 1]], cls: "l-diag" },
        { pts: d.reliability.map((b) => [b.conf, b.acc]), cls: "l-top", dots: true },
      ]} />

      <h3>Kaun si unseen signs model ko sabse zyada bewakoof banati hain (thr 0.70)</h3>
      <table>
        <thead><tr><th>Unseen class</th><th>Images</th><th>Wrongly accepted</th></tr></thead>
        <tbody>{d.worst_unseen.map((w) => (
          <tr key={w.cls}>
            <td>{w.name ? w.name : ("class " + w.cls)}</td>
            <td>{w.n}</td>
            <td className={w.accepted > 0.3 ? "bad" : ""}>{(w.accepted * 100).toFixed(0)}%</td>
          </tr>
        ))}</tbody>
      </table>
      <p className="muted small">Calibration aur unseen-sign numbers held-out Test.csv par nikle hain. Ye numbers is dataset ke liye hain, real road ke liye nahi.</p>
    </section>
  );
}

/* ---------- Test Explorer ---------- */
const FILTERS = [
  ["all", "Sab"], ["id", "Known signs"], ["wrong", "Known par galat"],
  ["ood", "Unseen signs"], ["ood_acc", "Unseen jo accept ho gaye ⚠️"],
];

export function Explorer({ s }) {
  const [d, err] = useAdvanced();
  const [f, setF] = useState("ood_acc");
  const [page, setPage] = useState(0);
  if (err) return <NeedEval msg={err} />;
  if (!d) return <div className="busy">Loading…</div>;
  const PER = 24;
  const list = d.samples.filter((x) => {
    const acc = !isUnc(x.conf, x.entropy, s);
    return f === "all" ? true : f === "id" ? !x.ood : f === "wrong" ? !x.ood && !x.correct
      : f === "ood" ? x.ood : x.ood && acc;
  });
  const pages = Math.max(1, Math.ceil(list.length / PER));
  const cur = list.slice(page * PER, page * PER + PER);
  return (
    <section>
      <div className="row">
        <select value={f} onChange={(e) => { setF(e.target.value); setPage(0); }}>
          {FILTERS.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
        </select>
        <span className="muted"> {list.length} images · sidebar thresholds ke hisaab se accept/reject badge</span>
      </div>
      <div className="xgrid">
        {cur.map((x) => {
          const acc = !isUnc(x.conf, x.entropy, s);
          const wrong = x.ood ? acc : !x.correct;
          return (
            <figure key={x.file} className={wrong ? "flip" : ""}>
              <img src={asset("/testimg/" + x.file)} alt={x.file} />
              <figcaption>
                <b>{x.true}</b><br />→ {x.pred} {(x.conf * 100).toFixed(0)}%<br />
                <span className={"badge " + (acc ? "acc" : "rej")}>{acc ? "accepted" : "rejected"}</span>
              </figcaption>
            </figure>
          );
        })}
      </div>
      {!cur.length && <p className="muted">Is filter mein koi image nahi (ya sab reject ho gayi — achhi baat hai).</p>}
      <div className="row">
        <button disabled={page === 0} onClick={() => setPage(page - 1)}>← Prev</button>
        <span className="muted"> {page + 1} / {pages} </span>
        <button disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>Next →</button>
      </div>
    </section>
  );
}

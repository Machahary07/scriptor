"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { CaptureCanvas, type CaptureHandle } from "@/components/CaptureCanvas";
import { api, type Job, type Writer } from "@/lib/api";

const PROMPTS = [
  "The quick brown fox jumps over the lazy dog.",
  "Pack my box with five dozen liquor jugs.",
  "A graph of current against voltage is a straight line.",
  "Seven students measured the length of the pendulum.",
  "Water boils at a lower temperature on high mountains.",
  "Every reading was checked twice before writing it down.",
  "Light travels faster than sound through the air.",
  "Jim quickly realized the beautiful gowns were expensive.",
  "Heat the solution gently and stir it until it dissolves.",
  "The experiment was repeated three times for accuracy.",
  "Zinc reacts with dilute acid to give hydrogen gas.",
  "My notebook has neat margins and numbered pages.",
  "Sixty zippers were quickly picked from the woven jute bag.",
  "The final answer is correct to two decimal places.",
  "Record the temperature every minute for ten minutes.",
];

export default function WritersPage() {
  const [writers, setWriters] = useState<Writer[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [consentText, setConsentText] = useState("");
  const [name, setName] = useState("");
  const [consent, setConsent] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [prompt, setPrompt] = useState(0);
  const [strokes, setStrokes] = useState(0);
  const [busy, setBusy] = useState(false);
  const [lora, setLora] = useState<Job | null>(null);
  const cap = useRef<CaptureHandle>(null);

  const refresh = useCallback(async () => {
    try {
      const ws = await api.writers();
      setWriters(ws);
      return ws;
    } catch (e) {
      setError((e as Error).message);
      return [];
    }
  }, []);

  useEffect(() => {
    refresh().then((ws) => ws[0] && setSel((s) => s ?? ws[0].id));
    api.consent().then((c) => setConsentText(c.text)).catch(() => {});
  }, [refresh]);

  // Poll a running LoRA job.
  useEffect(() => {
    if (!lora || lora.status !== "running") return;
    const id = setInterval(async () => {
      const j = await api.job(lora.id).catch(() => null);
      if (j) {
        setLora(j);
        if (j.status !== "running") refresh();
      }
    }, 3000);
    return () => clearInterval(id);
  }, [lora, refresh]);

  const w = writers.find((x) => x.id === sel) ?? null;

  const act = async (fn: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    setMsg(null);
    try {
      await fn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const saveLine = () =>
    act(async () => {
      if (!w || !cap.current?.count()) return;
      const sample = { text: PROMPTS[prompt], strokes: cap.current.strokes(), device: "browser" };
      await api.uploadSamples(w.id, [new Blob([JSON.stringify(sample)], { type: "application/json" })], [`line${Date.now()}.json`]);
      cap.current.clear();
      setPrompt((p) => (p + 1) % PROMPTS.length);
      await refresh();
      setMsg("Line saved.");
    });

  return (
    <div className="page">
      <div className="split">
        <div className="stack">
          <div className="card stack">
            <h2>New writer</h2>
            <label className="field">
              Name
              <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Neelashree" />
            </label>
            <label className="check">
              <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
              <span className="note" style={{ color: "var(--text)" }}>{consentText || "I consent to my handwriting samples being stored to build my style."}</span>
            </label>
            <button
              className="primary"
              disabled={!name.trim() || !consent || busy}
              onClick={() =>
                act(async () => {
                  const nw = await api.createWriter(name.trim(), consent);
                  setName("");
                  setConsent(false);
                  await refresh();
                  setSel(nw.id);
                })
              }
            >
              Register writer
            </button>
          </div>
          <div className="card stack" style={{ gap: 4 }}>
            <h3 style={{ marginBottom: 6 }}>Writers</h3>
            {!writers.length && <p className="note">No writers yet.</p>}
            {writers.map((x) => (
              <div key={x.id} className={`writer-item ${x.id === sel ? "on" : ""}`} onClick={() => setSel(x.id)}>
                <span className="avatar">{x.name.slice(0, 2).toUpperCase()}</span>
                <div className="grow">
                  <div>{x.name}</div>
                  <div className="note">{x.samples} lines{x.photos ? `, ${x.photos} photos` : ""}</div>
                </div>
                {x.adapter ? <span className="badge ok">LoRA</span> : x.style ? <span className="badge ok">style</span> : <span className="badge">no style</span>}
              </div>
            ))}
          </div>
        </div>

        <div className="stack">
          {error && <div className="error">{error}</div>}
          {msg && <div className="okbox">{msg}</div>}
          {!w ? (
            <div className="card empty">Register a writer to start recording samples.</div>
          ) : (
            <>
              <div className="card stack">
                <div className="card-head" style={{ marginBottom: 0 }}>
                  <div>
                    <h2>Record handwriting · {w.name}</h2>
                    <p className="note">Write the sentence on the lines with a stylus, finger or mouse. 10 to 20 lines give the best style.</p>
                  </div>
                  <span className="badge">{w.samples} / 20 lines</span>
                </div>
                <div className="prompt">{PROMPTS[prompt]}</div>
                <CaptureCanvas ref={cap} onChange={setStrokes} />
                <div className="row">
                  <button className="primary" onClick={saveLine} disabled={!strokes || busy}>Save line</button>
                  <button onClick={() => cap.current?.clear()} disabled={!strokes}>Clear</button>
                  <button onClick={() => { cap.current?.clear(); setPrompt((p) => (p + 1) % PROMPTS.length); }}>Skip sentence</button>
                  <span className="note">{strokes} strokes</span>
                </div>
              </div>

              <div className="card stack">
                <h2>Upload samples</h2>
                <p className="note">
                  Tablet files (<code>.json</code> in Scriptor format, or IAM-OnDB <code>.xml</code>) or photos of notebook pages (<code>.jpg</code>, <code>.png</code>).
                  Photos are converted to strokes by InkSight when it is configured on the engine.
                </p>
                <label className="btn" style={{ alignSelf: "flex-start", cursor: "pointer" }}>
                  Choose files
                  <input
                    type="file"
                    multiple
                    hidden
                    accept=".json,.xml,.jpg,.jpeg,.png"
                    onChange={(e) => {
                      const files = Array.from(e.target.files ?? []);
                      e.target.value = "";
                      if (files.length)
                        act(async () => {
                          const r = await api.uploadSamples(w.id, files);
                          await refresh();
                          setMsg(`Added ${r.added} sample${r.added === 1 ? "" : "s"}.`);
                        });
                    }}
                  />
                </label>
              </div>

              <div className="card stack">
                <h2>Style</h2>
                {w.style ? (
                  <div className="stats">
                    <div className="stat"><div className="k">Slant</div><div className="v">{(Math.abs(w.style.slant) < 1e-3 ? 0 : (w.style.slant * 180) / Math.PI).toFixed(1)}°</div></div>
                    <div className="stat"><div className="k">Letter width</div><div className="v">{w.style.width.toFixed(2)}×</div></div>
                    <div className="stat"><div className="k">Samples used</div><div className="v">{w.style.samples}</div></div>
                    <div className="stat"><div className="k">LoRA adapter</div><div className="v">{w.adapter ? "yes" : "no"}</div></div>
                  </div>
                ) : (
                  <p className="note">No style yet. Record or upload a few lines, then build it.</p>
                )}
                <div className="row">
                  <button
                    className="primary"
                    disabled={busy || !w.samples}
                    onClick={() => act(async () => { await api.buildStyle(w.id, "measure"); await refresh(); setMsg("Style measured from the samples."); })}
                  >
                    Measure style
                  </button>
                  <button
                    disabled={busy || w.samples < 3}
                    title="Needs the trained base model on the engine"
                    onClick={() =>
                      act(async () => {
                        const r = await api.buildStyle(w.id, "lora");
                        if (r.job) setLora(await api.job(r.job));
                      })
                    }
                  >
                    Train LoRA adapter
                  </button>
                </div>
                <p className="note">
                  Measuring takes seconds and drives the font model. A LoRA adapter personalises the neural model; it needs the pre-trained base weights and takes 1 to 2 hours on a GPU.
                </p>
                {lora && (
                  <div className="stack" style={{ gap: 6 }}>
                    <div className="row">
                      <span className={`badge ${lora.status === "done" ? "ok" : lora.status === "failed" ? "warn" : ""}`}>LoRA job {lora.status}</span>
                      {lora.error && <span className="note">{lora.error}</span>}
                    </div>
                    {lora.log_tail && <pre className="gline" style={{ whiteSpace: "pre-wrap", maxHeight: 160, overflow: "auto" }}>{lora.log_tail}</pre>}
                  </div>
                )}
              </div>

              <div className="card row" style={{ justifyContent: "space-between" }}>
                <p className="note">Deleting removes the writer, every sample, the style and any adapter from this machine.</p>
                <button
                  className="danger"
                  disabled={busy}
                  onClick={() => {
                    if (confirm(`Delete ${w.name} and all their samples?`))
                      act(async () => { await api.deleteWriter(w.id); setSel(null); const ws = await refresh(); setSel(ws[0]?.id ?? null); });
                  }}
                >
                  Delete writer
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { PagePreview, type PagePreviewHandle } from "@/components/PagePreview";
import { TimeBar } from "@/components/TimeBar";
import { api, type ModelInfo, type Writer } from "@/lib/api";
import { download, fmtMm, fmtTime, useJob } from "@/lib/job-store";

const SAMPLE = `Experiment 4: Verification of Ohm's Law

Aim: To verify Ohm's law by measuring the current through a resistor for different applied voltages.

Apparatus: DC power supply (0-12 V), ammeter, voltmeter, carbon resistor of 47 ohm, rheostat and connecting wires.

[box 40 Fig. 1: Circuit diagram]

Result: The graph of V against I is a straight line through the origin, so Ohm's law is verified.`;

export default function Studio() {
  const { job, setJob } = useJob();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [writers, setWriters] = useState<Writer[]>([]);
  const [text, setText] = useState(SAMPLE);
  const [writer, setWriter] = useState<string>("");
  const [model, setModel] = useState("font");
  const [bias, setBias] = useState(1);
  const [seed, setSeed] = useState(7);
  const [xHeight, setXHeight] = useState(2.6);
  const [spacing, setSpacing] = useState(8);
  const [join, setJoin] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [inkStyle, setInkStyle] = useState<"ink" | "constant">("ink");
  const [showTravel, setShowTravel] = useState(false);
  const preview = useRef<PagePreviewHandle>(null);

  useEffect(() => {
    api.models().then(setModels).catch(() => {});
    api.writers().then((ws) => setWriters(ws.filter((w) => w.style))).catch(() => {});
  }, []);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      const res = await api.generate({ text, writer_id: writer || null, model, bias, seed, x_height: xHeight, line_spacing: spacing });
      const pages = [];
      for (let i = 0; i < res.pages.length; i++) {
        const g = await api.gcode(res.pages[i].paths, { job: res.job, page: i + 1, join_mm: join ? 0.3 : 0 });
        pages.push(g);
      }
      const w = writers.find((x) => x.id === writer);
      setJob({ id: res.job, title: text.split("\n")[0].slice(0, 60), model, writer: w?.name ?? null, pages, createdAt: Date.now() });
      setPage(0);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const cur = job?.pages[Math.min(page, (job?.pages.length ?? 1) - 1)];
  const s = cur?.stats;

  return (
    <div className="page">
      <div className="split">
        <div className="stack">
          <div className="card stack">
            <div className="card-head" style={{ marginBottom: 0 }}>
              <h2>Write a page</h2>
              <span className="note">{text.length} characters</span>
            </div>
            <textarea value={text} onChange={(e) => setText(e.target.value)} spellCheck />
            <p className="note">
              A line like <code>[box 40 Fig. 1: caption]</code> reserves a 40 mm box for a diagram.
            </p>
            <div className="grid2">
              <label className="field">
                Writer
                <select value={writer} onChange={(e) => setWriter(e.target.value)}>
                  <option value="">Default style</option>
                  {writers.map((w) => (
                    <option key={w.id} value={w.id}>{w.name}</option>
                  ))}
                </select>
              </label>
              <label className="field">
                Model
                <select value={model} onChange={(e) => setModel(e.target.value)}>
                  {models.map((m) => (
                    <option key={m.name} value={m.name} disabled={!m.available}>
                      {m.label}{m.available ? "" : " (unavailable)"}
                    </option>
                  ))}
                  {!models.length && <option value="font">Varied single-stroke font</option>}
                </select>
              </label>
            </div>
            <label className="field">
              Neatness (bias {bias.toFixed(1)}): {bias < 0.6 ? "loose" : bias < 2 ? "natural" : "very neat"}
              <input type="range" min={0} max={4} step={0.1} value={bias} onChange={(e) => setBias(+e.target.value)} />
            </label>
            <div className="grid2">
              <label className="field">
                Letter height (mm)
                <input type="number" min={1.5} max={6} step={0.1} value={xHeight} onChange={(e) => setXHeight(+e.target.value)} />
              </label>
              <label className="field">
                Line spacing (mm)
                <input type="number" min={5} max={16} step={0.5} value={spacing} onChange={(e) => setSpacing(+e.target.value)} />
              </label>
            </div>
            <div className="row">
              <label className="field grow">
                Variation seed
                <div className="row" style={{ flexWrap: "nowrap" }}>
                  <input type="number" value={seed} onChange={(e) => setSeed(+e.target.value)} />
                  <button className="small" onClick={() => setSeed(Math.floor(Math.random() * 10000))} title="New random seed">Shuffle</button>
                </div>
              </label>
            </div>
            <label className="check">
              <input type="checkbox" checked={join} onChange={(e) => setJoin(e.target.checked)} />
              <span>Join touching strokes so the pen stays down within letters</span>
            </label>
            {error && <div className="error">{error}</div>}
            <button className="primary" onClick={run} disabled={busy || !text.trim()}>
              {busy ? "Writing…" : "Generate handwriting"}
            </button>
          </div>
          {models.length > 0 && (
            <div className="card stack" style={{ gap: 8 }}>
              <h3>Models</h3>
              {models.map((m) => (
                <div key={m.name} className="row" style={{ justifyContent: "space-between", flexWrap: "nowrap" }}>
                  <div>
                    <div>{m.label}</div>
                    <div className="note">{m.available ? m.description : m.status}</div>
                  </div>
                  <span className={`badge ${m.available ? "ok" : "warn"}`}>{m.available ? "ready" : "not ready"}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="stack">
          {!job || !cur || !s ? (
            <div className="card empty">
              <h2 style={{ marginBottom: 6 }}>No page yet</h2>
              <p>Type some text and press Generate. The page appears here with its toolpath statistics.</p>
            </div>
          ) : (
            <>
              <div className="card">
                <div className="card-head">
                  <div>
                    <h2>{job.title || "Untitled"}</h2>
                    <p className="note">
                      job {job.id} · {job.model}
                      {job.writer ? ` · ${job.writer}` : ""}
                    </p>
                  </div>
                  <div className="row">
                    {job.pages.length > 1 && (
                      <div className="seg">
                        {job.pages.map((_, i) => (
                          <button key={i} className={i === page ? "on" : ""} onClick={() => setPage(i)}>Page {i + 1}</button>
                        ))}
                      </div>
                    )}
                    <Link className="btn primary" href={`/simulator?page=${page}`} style={{ background: "var(--accent)", color: "#1a0d04", borderColor: "var(--accent)" }}>
                      Open in simulator
                    </Link>
                  </div>
                </div>
                <div className="stats">
                  <div className="stat"><div className="k">Plot time</div><div className="v">{fmtTime(s.total_s)}</div><div className="s">simulated, constant speed</div></div>
                  <div className="stat"><div className="k">Pen paths</div><div className="v">{s.paths}</div><div className="s">{s.paths_before_join !== s.paths ? `${s.paths_before_join} before joining` : "no joining"}</div></div>
                  <div className="stat"><div className="k">Ink drawn</div><div className="v">{fmtMm(s.draw_mm)}</div><div className="s">{s.points.toLocaleString()} points</div></div>
                  <div className="stat">
                    <div className="k">Pen-up travel</div>
                    <div className="v">{fmtMm(s.travel_mm)}</div>
                    <div className="s">{Math.round(100 * (1 - s.travel_mm / s.travel_mm_written_order))}% less than written order</div>
                  </div>
                  <div className="stat"><div className="k">G-code</div><div className="v">{s.gcode_lines.toLocaleString()}</div><div className="s">lines</div></div>
                </div>
                <div style={{ marginTop: 14 }}>
                  <TimeBar draw={s.draw_s} travel={s.travel_s} pen={s.pen_s} />
                </div>
              </div>
              <div className="card">
                <div className="card-head">
                  <div className="row">
                    <div className="seg">
                      <button className={inkStyle === "ink" ? "on" : ""} onClick={() => setInkStyle("ink")}>Ink model</button>
                      <button className={inkStyle === "constant" ? "on" : ""} onClick={() => setInkStyle("constant")}>Constant width</button>
                    </div>
                    <label className="check">
                      <input type="checkbox" checked={showTravel} onChange={(e) => setShowTravel(e.target.checked)} />
                      <span>Show pen-up travel</span>
                    </label>
                  </div>
                  <div className="row">
                    <button className="small" onClick={() => download(`scriptor-${job.id}-p${page + 1}.gcode`, cur.gcode)}>Download G-code</button>
                    <button
                      className="small"
                      onClick={async () => {
                        const b = await preview.current?.toPng();
                        if (b) download(`scriptor-${job.id}-p${page + 1}.png`, b);
                      }}
                    >
                      Download preview
                    </button>
                  </div>
                </div>
                <div className="paper-wrap">
                  <PagePreview ref={preview} gcode={cur.gcode} inkStyle={inkStyle} showTravel={showTravel} />
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

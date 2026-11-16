"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { TimeBar } from "@/components/TimeBar";
import { download, fmtMm, fmtTime } from "@/lib/job-store";
import { DEFAULT_MACHINE, simulate, stateAt, stepsPerMm } from "@/simulator/CoreXY";
import { parseGcode } from "@/simulator/GcodeParser";
import { DEFAULT_INK, clearPaper, drawInk, inkCountAt, prepareInk, type InkStyle } from "@/simulator/InkRenderer";
import { BUILD_STEPS, PlotterScene, type ViewMode } from "@/simulator/Scene3D";

const PX_PER_MM = 4;
const SPEEDS = [1, 4, 16, 64];

export function SimulatorView({ gcode, name }: { gcode: string; name: string }) {
  const host = useRef<HTMLDivElement>(null);
  const scene = useRef<PlotterScene | null>(null);
  const paper = useRef<HTMLCanvasElement | null>(null);
  const drawn = useRef(0);
  const inkStyleRef = useRef<InkStyle>("ink");

  const [mode, setMode] = useState<ViewMode>("plot");
  const [step, setStep] = useState(BUILD_STEPS.length);
  const [autoBuild, setAutoBuild] = useState(false);
  const [explode, setExplode] = useState(0.6);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(16);
  const [inkStyle, setInkStyle] = useState<InkStyle>("ink");

  const lines = useMemo(() => gcode.split(/\r?\n/), [gcode]);
  const sim = useMemo(() => simulate(parseGcode(gcode), DEFAULT_MACHINE), [gcode]);
  const ink = useMemo(() => prepareInk(sim), [sim]);
  const total = sim.totals.totalS;
  const state = stateAt(sim, t);

  // ---- three.js scene lifetime ----
  useEffect(() => {
    if (!host.current) return;
    const s = new PlotterScene(host.current);
    const c = document.createElement("canvas");
    c.width = DEFAULT_MACHINE.sheetW * PX_PER_MM;
    c.height = DEFAULT_MACHINE.sheetH * PX_PER_MM;
    clearPaper(c.getContext("2d")!, c.width, c.height);
    s.setPaperCanvas(c);
    scene.current = s;
    paper.current = c;
    return () => {
      s.dispose();
      scene.current = null;
    };
  }, []);

  const paintPaper = useCallback(
    (time: number, force = false) => {
      const c = paper.current;
      if (!c) return;
      const ctx = c.getContext("2d")!;
      const n = inkCountAt(ink, time);
      const opts = { ...DEFAULT_INK, pxPerMm: PX_PER_MM, style: inkStyleRef.current };
      if (force || n < drawn.current) {
        clearPaper(ctx, c.width, c.height);
        drawInk(ctx, ink, 0, n, opts);
      } else if (n > drawn.current) {
        drawInk(ctx, ink, drawn.current, n, opts);
      }
      drawn.current = n;
      scene.current?.markPaperDirty();
    },
    [ink],
  );

  // New file: rewind.
  useEffect(() => {
    setPlaying(false);
    setT(mode === "top" ? total : 0);
    drawn.current = 0;
    paintPaper(mode === "top" ? total : 0, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sim]);

  useEffect(() => {
    inkStyleRef.current = inkStyle;
    paintPaper(t, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inkStyle]);

  // Machine follows the playhead.
  useEffect(() => {
    scene.current?.setMachine(state.x, state.y, state.pen);
    paintPaper(t);
  }, [t, state.x, state.y, state.pen, paintPaper]);

  // Playback loop.
  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      setT((prev) => {
        const next = Math.min(total, prev + dt * speed);
        if (next >= total) setPlaying(false);
        return next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, total]);

  // Mode changes.
  useEffect(() => {
    const s = scene.current;
    if (!s) return;
    s.setMode(mode);
    if (mode === "build") {
      setStep(1);
      setAutoBuild(true);
    }
    if (mode === "exploded") s.setExplode(explode);
    if (mode === "top") {
      setPlaying(false);
      setT(total);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  useEffect(() => {
    if (mode === "build") scene.current?.setBuildStep(step);
  }, [step, mode]);

  useEffect(() => {
    if (mode === "exploded") scene.current?.setExplode(explode);
  }, [explode, mode]);

  useEffect(() => {
    if (!autoBuild || mode !== "build") return;
    if (step >= BUILD_STEPS.length) {
      setAutoBuild(false);
      return;
    }
    const id = setTimeout(() => setStep((n) => n + 1), 1500);
    return () => clearTimeout(id);
  }, [autoBuild, step, mode]);

  const tracked = sim.comments.find((c) => c.text.startsWith("scriptor"))?.text;
  const curStep = BUILD_STEPS[Math.max(0, step - 1)];

  return (
    <div className="split-r">
      <div className="viewport">
        <div ref={host} style={{ position: "absolute", inset: 0 }} />
        <div className="viewport-overlay">
          <div className="seg">
            {(["build", "exploded", "plot", "top"] as ViewMode[]).map((m) => (
              <button key={m} className={mode === m ? "on" : ""} onClick={() => setMode(m)}>
                {m === "build" ? "Build" : m === "exploded" ? "Exploded" : m === "plot" ? "Plot" : "Top view"}
              </button>
            ))}
          </div>
        </div>
        {(mode === "plot" || mode === "top") && (
          <div className="hud">
            <button className="small primary" onClick={() => { if (t >= total) setT(0); setPlaying((p) => !p); }} style={{ minWidth: 64 }}>
              {playing ? "Pause" : t >= total ? "Replay" : "Play"}
            </button>
            <input type="range" min={0} max={total || 1} step={0.01} value={t} onChange={(e) => { setPlaying(false); setT(+e.target.value); }} aria-label="Playhead" />
            <span className="mono" style={{ minWidth: 120, textAlign: "right" }}>{fmtTime(t)} / {fmtTime(total)}</span>
            <div className="seg">
              {SPEEDS.map((v) => (
                <button key={v} className={speed === v ? "on" : ""} onClick={() => setSpeed(v)}>{v}×</button>
              ))}
            </div>
            <button className="small" onClick={() => { setPlaying(false); setT(total); }}>End</button>
          </div>
        )}
        {mode === "exploded" && (
          <div className="hud">
            <span className="note">Spread</span>
            <input type="range" min={0} max={1.5} step={0.01} value={explode} onChange={(e) => setExplode(+e.target.value)} aria-label="Explode" />
            <span className="note">{BUILD_STEPS.length} part groups</span>
          </div>
        )}
      </div>

      <div className="stack">
        <div className="card tight stack" style={{ gap: 8 }}>
          <div className="card-head" style={{ marginBottom: 0 }}>
            <h2 style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{name}</h2>
            <span className="badge">{sim.lineCount.toLocaleString()} lines</span>
          </div>
          {tracked && <p className="note mono">; {tracked}</p>}
        </div>

        {mode === "build" ? (
          <div className="card stack">
            <h3>Assembly</h3>
            <div className="step-card stack" style={{ gap: 6 }}>
              <span className="n">Step {Math.max(1, step)} of {BUILD_STEPS.length}</span>
              <h2>{curStep.title}</h2>
              <p>{curStep.text}</p>
              <p className="note">Parts: {curStep.parts}</p>
            </div>
            <div className="row">
              <button className="small" onClick={() => { setAutoBuild(false); setStep((n) => Math.max(1, n - 1)); }} disabled={step <= 1}>Back</button>
              <button className="small" onClick={() => { setAutoBuild(false); setStep((n) => Math.min(BUILD_STEPS.length, n + 1)); }} disabled={step >= BUILD_STEPS.length}>Next</button>
              <button className="small" onClick={() => { if (step >= BUILD_STEPS.length) setStep(1); setAutoBuild((a) => !a); }}>{autoBuild ? "Pause" : "Auto"}</button>
            </div>
            <p className="note">Planned hardware: about ₹4,950. The same G-code drives it unchanged.</p>
          </div>
        ) : (
          <div className="card stack">
            <h3>Machine state</h3>
            <dl className="kv">
              <dt>Position</dt><dd>X {state.x.toFixed(2)} · Y {state.y.toFixed(2)} mm</dd>
              <dt>Motor A steps</dt><dd>{state.a.toLocaleString()}</dd>
              <dt>Motor B steps</dt><dd>{state.b.toLocaleString()}</dd>
              <dt>Pen</dt><dd className={state.pen ? "pen-down" : "pen-up"}>{state.pen ? "down (drawing)" : "up"}</dd>
              <dt>Resolution</dt><dd>{stepsPerMm(sim.config)} steps/mm</dd>
            </dl>
            <div className="gline" title={lines[state.line - 1]}>
              <span style={{ color: "var(--faint)" }}>{String(state.line).padStart(5, " ")} </span>
              {lines[state.line - 1] ?? ""}
            </div>
          </div>
        )}

        <div className="card stack">
          <h3>Simulated job</h3>
          <div className="grid2">
            <div className="stat"><div className="k">Plot time</div><div className="v">{fmtTime(total)}</div></div>
            <div className="stat"><div className="k">Pen lifts</div><div className="v">{sim.totals.penLifts}</div></div>
            <div className="stat"><div className="k">Ink</div><div className="v">{fmtMm(sim.totals.drawMm)}</div></div>
            <div className="stat"><div className="k">Travel</div><div className="v">{fmtMm(sim.totals.travelMm)}</div></div>
          </div>
          <TimeBar draw={sim.totals.drawS} travel={sim.totals.travelS} pen={sim.totals.penS} />
          <p className="note">Constant-speed model; acceleration is ignored, so a real plot takes somewhat longer.</p>
        </div>

        <div className="card stack">
          <div className="card-head" style={{ marginBottom: 0 }}>
            <h3>Validation</h3>
            <span className={`badge ${sim.warnings.length ? "warn" : "ok"}`}>
              {sim.warnings.length ? `${sim.warnings.length} warning${sim.warnings.length > 1 ? "s" : ""}` : "all checks passed"}
            </span>
          </div>
          {sim.warnings.length ? (
            <div className="warnings">
              {sim.warnings.slice(0, 200).map((w, i) => (
                <div key={i}><span className="ln">L{w.line}</span><span>{w.message}</span></div>
              ))}
            </div>
          ) : (
            <p className="note">Every move stays on the sheet, G1 only with the pen down, G0 only with it up, and both motors return to 0 steps at home.</p>
          )}
        </div>

        <div className="card stack">
          <h3>Paper</h3>
          <div className="row">
            <div className="seg">
              <button className={inkStyle === "ink" ? "on" : ""} onClick={() => setInkStyle("ink")}>Ink model</button>
              <button className={inkStyle === "constant" ? "on" : ""} onClick={() => setInkStyle("constant")}>Constant</button>
            </div>
            <button className="small" onClick={() => paper.current?.toBlob((b) => b && download(`${name.replace(/\W+/g, "-")}-paper.png`, b))}>Save paper PNG</button>
            <button className="small" onClick={() => { const url = scene.current?.snapshot(); if (url) { const a = document.createElement("a"); a.href = url; a.download = "scriptor-3d.png"; a.click(); } }}>Save 3D view</button>
          </div>
        </div>
      </div>
    </div>
  );
}

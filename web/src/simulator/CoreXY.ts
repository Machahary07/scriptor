// CoreXY kinematics, plot-time estimate and validation checks.
//
// In a CoreXY machine both motors move for every axis move. Belt travel of the
// two motors is
//     ΔA = Δx + Δy        ΔB = Δx − Δy
// and the step count is that distance times the steps per millimetre
//     steps/mm = (motor steps × microsteps) / (pulley teeth × belt pitch)
//              = (200 × 16) / (20 × 2 mm) = 80.
// Time uses the same constant-speed model as the engine: drawn length at the
// draw feed, travel at the travel feed, plus the G4 pen dwells.

import type { Cmd, Parsed } from "./GcodeParser";

export type MachineConfig = {
  stepsPerRev: number;
  microsteps: number;
  pulleyTeeth: number;
  beltPitch: number; // mm
  sheetW: number; // mm
  sheetH: number;
  travelFeed: number; // mm/min, used for G0 when no F is given
  drawFeed: number; // mm/min, used for G1 when no F is given
  penMoveS: number; // servo travel time if the file has no G4 after M3/M5
};

export const DEFAULT_MACHINE: MachineConfig = {
  stepsPerRev: 200,
  microsteps: 16,
  pulleyTeeth: 20,
  beltPitch: 2,
  sheetW: 210,
  sheetH: 297,
  travelFeed: 5000,
  drawFeed: 3000,
  penMoveS: 0,
};

export const stepsPerMm = (m: MachineConfig) => (m.stepsPerRev * m.microsteps) / (m.pulleyTeeth * m.beltPitch);

export type Segment = {
  kind: "draw" | "travel" | "dwell" | "pen";
  x0: number; y0: number; x1: number; y1: number; // machine mm, y up
  t0: number; t1: number; // seconds from start
  pen: boolean; // pen state during the segment
  line: number;
  len: number;
};

export type Warning = { line: number; message: string };

export type SimResult = {
  segments: Segment[];
  totals: { drawMm: number; travelMm: number; drawS: number; travelS: number; penS: number; totalS: number; penLifts: number };
  steps: { a: number; b: number };
  warnings: Warning[];
  unknown: Parsed["unknown"];
  comments: Parsed["comments"];
  lineCount: number;
  config: MachineConfig;
};

export type MachineState = {
  x: number; y: number; pen: boolean; line: number; a: number; b: number; t: number; segment: number;
};

export function simulate(parsed: Parsed, cfg: MachineConfig = DEFAULT_MACHINE): SimResult {
  const spm = stepsPerMm(cfg);
  const segments: Segment[] = [];
  const warnings: Warning[] = [];
  let x = 0, y = 0, pen = false, t = 0;
  let feedG0 = cfg.travelFeed, feedG1 = cfg.drawFeed;
  let drawMm = 0, travelMm = 0, drawS = 0, travelS = 0, penS = 0, penLifts = 0;
  let mm = true, absolute = true;
  const outside = (px: number, py: number) => px < -1e-6 || py < -1e-6 || px > cfg.sheetW + 1e-6 || py > cfg.sheetH + 1e-6;

  for (const c of parsed.cmds as Cmd[]) {
    switch (c.code) {
      case "G21": mm = true; break;
      case "G90": absolute = true; break;
      case "G0":
      case "G1": {
        if (!mm || !absolute) warnings.push({ line: c.line, message: "only mm (G21) and absolute (G90) are supported" });
        const nx = c.x ?? x, ny = c.y ?? y;
        if (c.f !== undefined) {
          if (c.code === "G0") feedG0 = c.f; else feedG1 = c.f;
        }
        const draw = c.code === "G1";
        if (draw && !pen) warnings.push({ line: c.line, message: "G1 drawing move with the pen up" });
        if (!draw && pen) warnings.push({ line: c.line, message: "G0 travel move with the pen down (would draw a line)" });
        if (outside(nx, ny)) warnings.push({ line: c.line, message: `move to (${nx.toFixed(1)}, ${ny.toFixed(1)}) is outside the ${cfg.sheetW}×${cfg.sheetH} mm sheet` });
        const len = Math.hypot(nx - x, ny - y);
        const dt = (len / (draw ? feedG1 : feedG0)) * 60;
        if (len > 0) {
          segments.push({ kind: pen ? "draw" : "travel", x0: x, y0: y, x1: nx, y1: ny, t0: t, t1: t + dt, pen, line: c.line, len });
          if (pen) { drawMm += len; drawS += dt; } else { travelMm += len; travelS += dt; }
          t += dt;
        }
        x = nx; y = ny;
        break;
      }
      case "M3":
      case "M5": {
        const next = c.code === "M3";
        if (next !== pen) {
          if (!next) penLifts++;
          pen = next;
          segments.push({ kind: "pen", x0: x, y0: y, x1: x, y1: y, t0: t, t1: t + cfg.penMoveS, pen, line: c.line, len: 0 });
          t += cfg.penMoveS;
          penS += cfg.penMoveS;
        }
        break;
      }
      case "G4": {
        const dt = c.p ?? 0;
        segments.push({ kind: "dwell", x0: x, y0: y, x1: x, y1: y, t0: t, t1: t + dt, pen, line: c.line, len: 0 });
        t += dt;
        penS += dt;
        break;
      }
      case "M2":
        break;
    }
  }
  const a = Math.round((x + y) * spm);
  const b = Math.round((x - y) * spm);
  if (a !== 0 || b !== 0) warnings.push({ line: parsed.lineCount, message: `job ends away from home: motor A ${a}, motor B ${b} steps` });
  if (pen) warnings.push({ line: parsed.lineCount, message: "job ends with the pen down" });
  for (const u of parsed.unknown) warnings.push({ line: u.line, message: `unknown command: ${u.raw.trim()}` });
  return {
    segments,
    totals: { drawMm, travelMm, drawS, travelS, penS, totalS: t, penLifts },
    steps: { a, b },
    warnings,
    unknown: parsed.unknown,
    comments: parsed.comments,
    lineCount: parsed.lineCount,
    config: cfg,
  };
}

/** Index of the segment active at time t (binary search). */
export function segmentAt(segs: Segment[], t: number): number {
  let lo = 0, hi = segs.length - 1;
  if (hi < 0) return -1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (segs[mid].t1 < t) lo = mid + 1; else hi = mid;
  }
  return lo;
}

export function stateAt(sim: SimResult, t: number): MachineState {
  const spm = stepsPerMm(sim.config);
  const segs = sim.segments;
  if (!segs.length) return { x: 0, y: 0, pen: false, line: 0, a: 0, b: 0, t: 0, segment: -1 };
  const i = segmentAt(segs, t);
  const s = segs[i];
  const k = s.t1 > s.t0 ? Math.min(1, Math.max(0, (t - s.t0) / (s.t1 - s.t0))) : t >= s.t1 ? 1 : 0;
  const x = s.x0 + (s.x1 - s.x0) * k;
  const y = s.y0 + (s.y1 - s.y0) * k;
  return { x, y, pen: s.pen, line: s.line, a: Math.round((x + y) * spm), b: Math.round((x - y) * spm), t, segment: i };
}

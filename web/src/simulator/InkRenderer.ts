// Ink model for the print preview and the paper texture in the 3D view.
//
// A constant-width line looks printed. A real pen slows down at sharp corners
// and at the start and end of a stroke, and the line gets wider and darker
// where it is slow. For each point of a stroke we estimate the relative speed
//     v = (1 − 0.6 · θ/π) · min(1, √(s_start / r), √(s_end / r))
// where θ is the turning angle at the point, s_start and s_end are the distances
// along the stroke to its two ends and r = 1.5 mm is the acceleration ramp.
// The line width is then
//     w = w0 · (1 + 0.6 · (1 − v))
// with w0 the nominal pen width. A small blob marks each pen touch-down.

import type { SimResult } from "./CoreXY";

export type InkStyle = "ink" | "constant";

export type InkSegment = {
  x0: number; y0: number; x1: number; y1: number; // page mm, y down
  w0: number; w1: number; // width factor at each end (1 = nominal)
  t1: number; // time the pen reaches the end, for progressive drawing
  touchDown: boolean; // first segment of a stroke
};

const RAMP = 1.5;

export function prepareInk(sim: SimResult): InkSegment[] {
  const H = sim.config.sheetH;
  const out: InkSegment[] = [];
  let stroke: { x: number; y: number; t: number }[] = [];
  const flush = () => {
    if (stroke.length < 2) { stroke = []; return; }
    const n = stroke.length;
    const s = new Array<number>(n).fill(0);
    for (let i = 1; i < n; i++) s[i] = s[i - 1] + Math.hypot(stroke[i].x - stroke[i - 1].x, stroke[i].y - stroke[i - 1].y);
    const total = s[n - 1];
    const v = stroke.map((p, i) => {
      let theta = 0;
      if (i > 0 && i < n - 1) {
        const ax = p.x - stroke[i - 1].x, ay = p.y - stroke[i - 1].y;
        const bx = stroke[i + 1].x - p.x, by = stroke[i + 1].y - p.y;
        const la = Math.hypot(ax, ay), lb = Math.hypot(bx, by);
        if (la > 0 && lb > 0) theta = Math.acos(Math.max(-1, Math.min(1, (ax * bx + ay * by) / (la * lb))));
      }
      const ramp = Math.min(1, Math.sqrt(s[i] / RAMP), Math.sqrt((total - s[i]) / RAMP));
      return (1 - (0.6 * theta) / Math.PI) * ramp;
    });
    const w = v.map((vi) => 1 + 0.6 * (1 - vi));
    for (let i = 1; i < n; i++) {
      out.push({
        x0: stroke[i - 1].x, y0: stroke[i - 1].y, x1: stroke[i].x, y1: stroke[i].y,
        w0: w[i - 1], w1: w[i], t1: stroke[i].t, touchDown: i === 1,
      });
    }
    stroke = [];
  };
  for (const seg of sim.segments) {
    if (seg.kind === "draw") {
      if (!stroke.length) stroke.push({ x: seg.x0, y: H - seg.y0, t: seg.t0 });
      stroke.push({ x: seg.x1, y: H - seg.y1, t: seg.t1 });
    } else if (seg.kind === "pen" && !seg.pen) {
      flush();
    }
  }
  flush();
  return out;
}

export type InkOptions = {
  pxPerMm: number;
  penWidthMm: number;
  color: string;
  style: InkStyle;
};

export const DEFAULT_INK: InkOptions = { pxPerMm: 4, penWidthMm: 0.4, color: "#1b2f8f", style: "ink" };

/** Draw segments [from, to) onto a canvas context in page millimetres. */
export function drawInk(ctx: CanvasRenderingContext2D, ink: InkSegment[], from: number, to: number, o: InkOptions) {
  const k = o.pxPerMm;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = o.color;
  ctx.fillStyle = o.color;
  for (let i = from; i < Math.min(to, ink.length); i++) {
    const s = ink[i];
    const wf = o.style === "ink" ? (s.w0 + s.w1) / 2 : 1;
    ctx.globalAlpha = o.style === "ink" ? Math.min(1, 0.72 + 0.25 * (wf - 1) / 0.6) : 1;
    ctx.lineWidth = o.penWidthMm * wf * k;
    ctx.beginPath();
    ctx.moveTo(s.x0 * k, s.y0 * k);
    ctx.lineTo(s.x1 * k, s.y1 * k);
    ctx.stroke();
    if (o.style === "ink" && s.touchDown) {
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      ctx.arc(s.x0 * k, s.y0 * k, o.penWidthMm * 0.85 * k, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
}

/** Number of ink segments finished by time t (ink is sorted by time). */
export function inkCountAt(ink: InkSegment[], t: number): number {
  let lo = 0, hi = ink.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (ink[mid].t1 <= t) lo = mid + 1; else hi = mid;
  }
  return lo;
}

export function clearPaper(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#fbfaf6";
  ctx.fillRect(0, 0, w, h);
}

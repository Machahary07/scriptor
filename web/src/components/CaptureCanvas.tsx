"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";

export type Point = [number, number, number]; // x, y (px), t (ms from first touch)
export type CaptureHandle = { strokes: () => Point[][]; clear: () => void; count: () => number };

/**
 * Records handwriting as online strokes with a pen, finger or mouse.
 * Each pointer-down to pointer-up is one stroke of (x, y, t) points.
 */
export const CaptureCanvas = forwardRef<CaptureHandle, { onChange?: (n: number) => void }>(function CaptureCanvas({ onChange }, ref) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const strokes = useRef<Point[][]>([]);
  const t0 = useRef<number | null>(null);
  const drawing = useRef(false);

  const redraw = () => {
    const c = canvas.current;
    if (!c) return;
    const ctx = c.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = c.width / dpr, h = c.height / dpr;
    ctx.fillStyle = "#fbfaf6";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "#cfd8e6";
    ctx.lineWidth = 1;
    for (const y of [h * 0.42, h * 0.68]) {
      ctx.beginPath();
      ctx.moveTo(12, y);
      ctx.lineTo(w - 12, y);
      ctx.stroke();
    }
    ctx.strokeStyle = "#1b2f8f";
    ctx.lineWidth = 2.2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const s of strokes.current) {
      ctx.beginPath();
      s.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      if (s.length === 1) ctx.lineTo(s[0][0] + 0.1, s[0][1]);
      ctx.stroke();
    }
  };

  useEffect(() => {
    const c = canvas.current!;
    const fit = () => {
      const dpr = window.devicePixelRatio || 1;
      c.width = c.clientWidth * dpr;
      c.height = c.clientHeight * dpr;
      redraw();
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(c);
    return () => ro.disconnect();
  }, []);

  useImperativeHandle(ref, () => ({
    strokes: () => strokes.current,
    count: () => strokes.current.length,
    clear: () => {
      strokes.current = [];
      t0.current = null;
      redraw();
      onChange?.(0);
    },
  }));

  const pos = (e: React.PointerEvent): Point => {
    const r = canvas.current!.getBoundingClientRect();
    if (t0.current === null) t0.current = e.timeStamp;
    return [e.clientX - r.left, e.clientY - r.top, Math.round(e.timeStamp - t0.current)];
  };

  return (
    <canvas
      ref={canvas}
      className="capture"
      onPointerDown={(e) => {
        canvas.current!.setPointerCapture(e.pointerId);
        drawing.current = true;
        strokes.current.push([pos(e)]);
        redraw();
      }}
      onPointerMove={(e) => {
        if (!drawing.current) return;
        const events = e.nativeEvent.getCoalescedEvents?.() ?? [e.nativeEvent];
        const s = strokes.current[strokes.current.length - 1];
        const r = canvas.current!.getBoundingClientRect();
        for (const ev of events) s.push([ev.clientX - r.left, ev.clientY - r.top, Math.round(ev.timeStamp - (t0.current ?? ev.timeStamp))]);
        redraw();
      }}
      onPointerUp={() => {
        drawing.current = false;
        onChange?.(strokes.current.length);
      }}
      onPointerCancel={() => (drawing.current = false)}
    />
  );
});

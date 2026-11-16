"use client";

import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef } from "react";
import { DEFAULT_MACHINE, simulate } from "@/simulator/CoreXY";
import { parseGcode } from "@/simulator/GcodeParser";
import { DEFAULT_INK, clearPaper, drawInk, prepareInk, type InkStyle } from "@/simulator/InkRenderer";

type Props = { gcode: string; inkStyle?: InkStyle; showTravel?: boolean; pxPerMm?: number };
export type PagePreviewHandle = { toPng: () => Promise<Blob | null> };

/** Full-page print preview of a G-code file, drawn with the simulator's ink model. */
export const PagePreview = forwardRef<PagePreviewHandle, Props>(function PagePreview(
  { gcode, inkStyle = "ink", showTravel = false, pxPerMm = 4 },
  ref,
) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const sim = useMemo(() => simulate(parseGcode(gcode), DEFAULT_MACHINE), [gcode]);
  const ink = useMemo(() => prepareInk(sim), [sim]);

  useImperativeHandle(ref, () => ({
    toPng: () => new Promise((res) => (canvas.current ? canvas.current.toBlob(res, "image/png") : res(null))),
  }));

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const { sheetW, sheetH } = sim.config;
    c.width = sheetW * pxPerMm;
    c.height = sheetH * pxPerMm;
    const ctx = c.getContext("2d")!;
    clearPaper(ctx, c.width, c.height);
    drawInk(ctx, ink, 0, ink.length, { ...DEFAULT_INK, pxPerMm, style: inkStyle });
    if (showTravel) {
      ctx.save();
      ctx.strokeStyle = "rgba(220, 50, 50, 0.65)";
      ctx.lineWidth = 0.6;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      for (const s of sim.segments) {
        if (s.kind !== "travel") continue;
        ctx.moveTo(s.x0 * pxPerMm, (sheetH - s.y0) * pxPerMm);
        ctx.lineTo(s.x1 * pxPerMm, (sheetH - s.y1) * pxPerMm);
      }
      ctx.stroke();
      ctx.restore();
    }
  }, [sim, ink, inkStyle, showTravel, pxPerMm]);

  return <canvas ref={canvas} />;
});

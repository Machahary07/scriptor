import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { DEFAULT_MACHINE, simulate, stateAt, stepsPerMm } from "./CoreXY";
import { parseGcode } from "./GcodeParser";
import { prepareInk } from "./InkRenderer";

const JOB = `; scriptor job=test
G21 ; mm
G90 ; absolute
M5 ; pen up
G0 X0 Y0
G0 X10.00 Y20.00 F5000
M3 S90 ; pen down
G4 P0.15
G1 X30.00 Y20.00 F3000
G1 X30.00 Y40.00 F3000
M5 ; pen up
G4 P0.10
G0 X0 Y0 F5000
M2 ; end
`;

describe("parser", () => {
  it("reads commands, words and comments", () => {
    const p = parseGcode(JOB);
    expect(p.unknown).toHaveLength(0);
    expect(p.comments[0]).toEqual({ line: 1, text: "scriptor job=test" });
    const g1 = p.cmds.find((c) => c.code === "G1")!;
    expect(g1).toMatchObject({ x: 30, y: 20, f: 3000, line: 9 });
    expect(p.cmds.find((c) => c.code === "G4")!.p).toBe(0.15);
  });

  it("lists unknown commands instead of ignoring them", () => {
    const p = parseGcode("G28\nG1 X1 Y1\nT2");
    expect(p.unknown.map((u) => u.line)).toEqual([1, 3]);
  });
});

describe("CoreXY simulation", () => {
  it("uses 80 steps per mm for 200-step motors, 1/16 microstepping, 20T GT2", () => {
    expect(stepsPerMm(DEFAULT_MACHINE)).toBe(80);
  });

  it("computes lengths and constant-speed time", () => {
    const s = simulate(parseGcode(JOB));
    expect(s.totals.drawMm).toBeCloseTo(40);
    const travel = Math.hypot(10, 20) + Math.hypot(30, 40);
    expect(s.totals.travelMm).toBeCloseTo(travel);
    expect(s.totals.penS).toBeCloseTo(0.25);
    expect(s.totals.totalS).toBeCloseTo((40 / 3000) * 60 + (travel / 5000) * 60 + 0.25);
    expect(s.totals.penLifts).toBe(1);
  });

  it("returns both motors to zero steps at home and passes all checks", () => {
    const s = simulate(parseGcode(JOB));
    expect(s.steps).toEqual({ a: 0, b: 0 });
    expect(s.warnings).toEqual([]);
  });

  it("maps XY moves to CoreXY motor steps: A = x + y, B = x - y", () => {
    const s = simulate(parseGcode(JOB));
    const mid = s.segments.find((x) => x.kind === "draw")!;
    const st = stateAt(s, mid.t1);
    expect(st.x).toBeCloseTo(30);
    expect(st.y).toBeCloseTo(20);
    expect(st.a).toBe((30 + 20) * 80);
    expect(st.b).toBe((30 - 20) * 80);
    expect(st.pen).toBe(true);
  });

  it("warns about drawing with the pen up, leaving the sheet and not returning home", () => {
    const s = simulate(parseGcode("G21\nG90\nG1 X10 Y10\nM3\nG0 X300 Y10\nM5\n"));
    const msgs = s.warnings.map((w) => `${w.line}:${w.message}`);
    expect(msgs.some((m) => m.startsWith("3:G1 drawing move with the pen up"))).toBe(true);
    expect(msgs.some((m) => m.startsWith("5:G0 travel move with the pen down"))).toBe(true);
    expect(msgs.some((m) => m.startsWith("5:") && m.includes("outside"))).toBe(true);
    expect(msgs.some((m) => m.includes("ends away from home"))).toBe(true);
  });
});

describe("agreement with the Python engine", () => {
  it("reproduces the engine's lengths, pen lifts and plot time from the G-code alone", () => {
    const dir = path.join(__dirname, "fixtures");
    const gcode = readFileSync(path.join(dir, "sample.gcode"), "utf8");
    const [stats] = JSON.parse(readFileSync(path.join(dir, "sample.stats.json"), "utf8"));
    const s = simulate(parseGcode(gcode));
    expect(s.warnings).toEqual([]);
    expect(s.totals.penLifts).toBe(stats.paths);
    // G-code coordinates are rounded to 0.01 mm, so allow a small difference.
    expect(s.totals.drawMm).toBeCloseTo(stats.draw_mm, 0);
    expect(s.totals.travelMm).toBeCloseTo(stats.travel_mm, 0);
    expect(s.totals.totalS).toBeCloseTo(stats.total_s, 0);
  });
});

describe("ink model", () => {
  it("widens the line at stroke ends and sharp corners", () => {
    const s = simulate(parseGcode(JOB));
    const ink = prepareInk(s);
    expect(ink).toHaveLength(2);
    expect(ink[0].touchDown).toBe(true);
    expect(ink[0].w0).toBeGreaterThan(1.5); // start of stroke: slow, wide
    expect(ink[0].w1).toBeGreaterThan(1.1); // 90° corner
    // page coordinates are y-down: machine y=20 -> page 277
    expect(ink[0].y0).toBeCloseTo(DEFAULT_MACHINE.sheetH - 20);
  });
});

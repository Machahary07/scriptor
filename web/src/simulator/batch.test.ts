// Runs every simulator validation check on a folder of G-code files.
// Skipped unless SCRIPTOR_GCODE_DIR is set, e.g. after
//   python training/evaluate.py --pages 50 --gcode-dir out/eval
//   SCRIPTOR_GCODE_DIR=../out/eval npx vitest run batch

import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { simulate } from "./CoreXY";
import { parseGcode } from "./GcodeParser";

const dir = process.env.SCRIPTOR_GCODE_DIR;

describe.skipIf(!dir)("batch validation", () => {
  it("passes every check on every file", () => {
    const files = readdirSync(dir!).filter((f) => f.endsWith(".gcode")).sort();
    expect(files.length).toBeGreaterThan(0);
    const failed: string[] = [];
    let total = 0;
    for (const f of files) {
      const sim = simulate(parseGcode(readFileSync(path.join(dir!, f), "utf8")));
      total += sim.totals.totalS;
      if (sim.warnings.length) failed.push(`${f}: ${sim.warnings[0].message}`);
    }
    console.log(`${files.length - failed.length}/${files.length} files passed; mean simulated time ${(total / files.length).toFixed(1)} s`);
    expect(failed).toEqual([]);
  });
});

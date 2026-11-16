// Parses the subset of GRBL G-code that Scriptor writes. Anything else is
// reported in `unknown` instead of being silently ignored.

export type Cmd = {
  line: number; // 1-based line number in the file
  raw: string;
  code: string; // "G0", "G1", "G4", "M3", ...
  x?: number;
  y?: number;
  f?: number;
  p?: number;
  s?: number;
};

export type Parsed = {
  cmds: Cmd[];
  unknown: { line: number; raw: string }[];
  comments: { line: number; text: string }[];
  lineCount: number;
};

const KNOWN = new Set(["G0", "G1", "G4", "G21", "G90", "M2", "M3", "M5"]);

export function parseGcode(text: string): Parsed {
  const lines = text.split(/\r?\n/);
  const cmds: Cmd[] = [];
  const unknown: Parsed["unknown"] = [];
  const comments: Parsed["comments"] = [];
  lines.forEach((raw, i) => {
    const line = i + 1;
    const semi = raw.indexOf(";");
    if (semi >= 0) {
      const c = raw.slice(semi + 1).trim();
      if (c) comments.push({ line, text: c });
    }
    const body = (semi >= 0 ? raw.slice(0, semi) : raw).replace(/\(.*?\)/g, "").trim().toUpperCase();
    if (!body) return;
    const words = body.match(/[A-Z][-+]?\d*\.?\d+/g);
    if (!words) {
      unknown.push({ line, raw });
      return;
    }
    const letter = words[0][0];
    const code = letter + String(parseFloat(words[0].slice(1)));
    if (!KNOWN.has(code)) {
      unknown.push({ line, raw });
      return;
    }
    const cmd: Cmd = { line, raw, code };
    for (const w of words.slice(1)) {
      const v = parseFloat(w.slice(1));
      switch (w[0]) {
        case "X": cmd.x = v; break;
        case "Y": cmd.y = v; break;
        case "F": cmd.f = v; break;
        case "P": cmd.p = v; break;
        case "S": cmd.s = v; break;
      }
    }
    cmds.push(cmd);
  });
  // A trailing newline does not start another line.
  const lineCount = lines.length - (lines[lines.length - 1] === "" ? 1 : 0);
  return { cmds, unknown, comments, lineCount };
}

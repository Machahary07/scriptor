// Typed client for the Scriptor engine (FastAPI). Requests go to /api/*, which
// next.config.ts rewrites to the engine (http://127.0.0.1:8000 by default).

export type ModelInfo = { name: string; label: string; description: string; available: boolean; status: string };

export type StyleInfo = { slant: number; width: number; spacing: number; word_gap: number; messiness: number; samples: number };

export type Writer = {
  id: string;
  name: string;
  consent_at: number;
  samples: number;
  photos: number;
  style: StyleInfo | null;
  adapter: boolean;
};

export type PageItem = { kind: "line"; text: string; x: number; y: number } | { kind: "box"; x: number; y: number; w: number; h: number };
export type GeneratedPage = { layout: { spec: Record<string, number>; items: PageItem[] }; paths: number[][][] };

export type GcodeStats = {
  paths: number;
  paths_before_join: number;
  draw_mm: number;
  travel_mm: number;
  travel_mm_written_order: number;
  draw_s: number;
  travel_s: number;
  pen_s: number;
  total_s: number;
  total_s_written_order: number;
  pen_share: number;
  points: number;
  gcode_lines: number;
};

export type GenerateRequest = {
  text: string;
  writer_id?: string | null;
  model: string;
  bias: number;
  seed: number;
  x_height?: number;
  line_spacing?: number;
};

export type Job = { id: string; status: string; model: string; stats: Record<string, unknown>; error: string | null; files: string[]; log_tail?: string };

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, init);
  } catch {
    throw new Error("Cannot reach the engine. Start it with: uvicorn engine.app:app --port 8000");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* not JSON */
    }
    if (res.status === 500 && detail === "Internal Server Error") detail = "Engine is not running or failed (see the engine terminal).";
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const api = {
  health: () => call<{ ok: boolean }>("/health"),
  models: () => call<ModelInfo[]>("/models"),
  consent: () => call<{ text: string }>("/consent"),
  writers: () => call<Writer[]>("/writers"),
  writer: (id: string) => call<Writer>(`/writers/${id}`),
  createWriter: (name: string, consent: boolean) => {
    const fd = new FormData();
    fd.set("name", name);
    fd.set("consent", String(consent));
    return call<Writer>("/writers", { method: "POST", body: fd });
  },
  deleteWriter: (id: string) => call<{ deleted: string }>(`/writers/${id}`, { method: "DELETE" }),
  uploadSamples: (id: string, files: (File | Blob)[], names?: string[]) => {
    const fd = new FormData();
    files.forEach((f, i) => fd.append("files", f, names?.[i] ?? (f as File).name ?? `sample${i}.json`));
    return call<{ added: number; writer: Writer }>(`/writers/${id}/samples`, { method: "POST", body: fd });
  },
  buildStyle: (id: string, mode: "measure" | "lora") => call<{ style: StyleInfo; job?: string }>(`/writers/${id}/style`, json({ mode })),
  generate: (req: GenerateRequest) => call<{ job: string; pages: GeneratedPage[] }>("/generate", json(req)),
  gcode: (paths: number[][][], opts: { job?: string; page?: number; order?: boolean; join_mm?: number }) =>
    call<{ gcode: string; stats: GcodeStats }>("/gcode", json({ paths, ...opts })),
  job: (id: string) => call<Job>(`/jobs/${id}`),
};

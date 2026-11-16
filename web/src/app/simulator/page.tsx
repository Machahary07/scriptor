"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";
import { useJob } from "@/lib/job-store";

// three.js needs the browser; load the simulator on the client only.
const SimulatorView = dynamic(() => import("@/components/SimulatorView").then((m) => m.SimulatorView), {
  ssr: false,
  loading: () => <div className="viewport empty">Loading simulator…</div>,
});

function SimulatorPage() {
  const { job } = useJob();
  const params = useSearchParams();
  const [file, setFile] = useState<{ name: string; text: string } | null>(null);
  const [page, setPage] = useState(0);

  useEffect(() => {
    const p = Number(params.get("page") ?? 0);
    if (Number.isFinite(p)) setPage(p);
  }, [params]);

  const fromJob = job && !file ? job.pages[Math.min(page, job.pages.length - 1)] : null;
  const gcode = file?.text ?? fromJob?.gcode ?? null;
  const name = file?.name ?? (job ? `${job.title || "Job " + job.id} · page ${page + 1}` : "");

  return (
    <div className="page stack">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div>
          <h1>Virtual plotter</h1>
          <p className="note">Runs the same G-code a GRBL CoreXY plotter would receive.</p>
        </div>
        <div className="row">
          {job && !file && job.pages.length > 1 && (
            <div className="seg">
              {job.pages.map((_, i) => (
                <button key={i} className={i === page ? "on" : ""} onClick={() => setPage(i)}>Page {i + 1}</button>
              ))}
            </div>
          )}
          {file && job && <button className="small" onClick={() => setFile(null)}>Back to current job</button>}
          <label className="btn small" style={{ cursor: "pointer" }}>
            Open G-code file
            <input
              type="file"
              accept=".gcode,.nc,.gc,.txt"
              hidden
              onChange={async (e) => {
                const f = e.target.files?.[0];
                if (f) setFile({ name: f.name, text: await f.text() });
                e.target.value = "";
              }}
            />
          </label>
        </div>
      </div>
      {gcode ? (
        <SimulatorView key={name} gcode={gcode} name={name} />
      ) : (
        <div className="card empty">
          <h2 style={{ marginBottom: 6 }}>Nothing to plot yet</h2>
          <p>
            Generate a page in the <Link href="/" style={{ color: "var(--accent)" }}>Studio</Link>, or open a G-code file.
          </p>
        </div>
      )}
    </div>
  );
}

export default function Page() {
  return (
    <Suspense>
      <SimulatorPage />
    </Suspense>
  );
}

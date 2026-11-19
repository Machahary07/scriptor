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
      <div className="page-head" style={{ marginBottom: 4 }}>
        <div>
          <span className="kicker">the machine</span>
          <h1>Virtual <em>plotter.</em></h1>
          <p>Runs the same G-code a GRBL CoreXY plotter would receive, so what plots here plots on the real machine.</p>
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
        <div className="viewport bay-empty">
          <svg viewBox="0 0 400 260" fill="none" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <rect x="40" y="40" width="320" height="190" rx="6" stroke="#3a4558" strokeWidth="2" />
            <rect x="120" y="70" width="160" height="140" fill="#f3efe6" opacity=".9" />
            <path d="M40 120h320" stroke="#5b6a82" strokeWidth="6" />
            <rect x="182" y="106" width="36" height="28" rx="4" fill="#d9482b" />
            <path className="draw" d="M135 95c10-2 12-14 20-14s2 14 10 14 8-10 14-10 4 8 10 8M135 125c8 0 10-10 16-10s4 10 12 10" stroke="#1f3a8a" strokeWidth="2" />
          </svg>
          <h2>The bay is empty</h2>
          <p>
            Generate a page in the <Link href="/">Studio</Link>, or open a G-code file to watch the plotter write it.
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

import { fmtTime } from "@/lib/job-store";

/** Split of plot time into drawing, pen-up travel and pen lift/lower dwells. */
export function TimeBar({ draw, travel, pen }: { draw: number; travel: number; pen: number }) {
  const total = draw + travel + pen || 1;
  const parts = [
    { k: "Drawing", v: draw, c: "var(--ink)" },
    { k: "Travel", v: travel, c: "var(--accent)" },
    { k: "Pen lifts", v: pen, c: "#e7b25c" },
  ];
  return (
    <div className="stack" style={{ gap: 8 }}>
      <div className="bar">
        {parts.map((p) => (
          <span key={p.k} style={{ width: `${(100 * p.v) / total}%`, background: p.c }} title={`${p.k}: ${fmtTime(p.v)}`} />
        ))}
      </div>
      <div className="legend">
        {parts.map((p) => (
          <span key={p.k}>
            <i style={{ background: p.c }} />
            {p.k} {fmtTime(p.v)} ({Math.round((100 * p.v) / total)}%)
          </span>
        ))}
      </div>
    </div>
  );
}

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

const LINKS = [
  { href: "/", label: "Studio" },
  { href: "/writers", label: "Writers" },
  { href: "/simulator", label: "Simulator" },
];

export function Nav() {
  const path = usePathname();
  const [up, setUp] = useState<boolean | null>(null);
  useEffect(() => {
    let alive = true;
    const check = () => api.health().then(() => alive && setUp(true)).catch(() => alive && setUp(false));
    check();
    const id = setInterval(check, 10000);
    return () => { alive = false; clearInterval(id); };
  }, []);
  return (
    <header className="topbar">
      <Link href="/" className="brand">
        <span className="brand-mark">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#1a0d04" strokeWidth="2.4" strokeLinecap="round">
            <path d="M3 17c3-1 4-9 7-9s1 8 4 8 3-6 5-6 1 3 2 3" />
          </svg>
        </span>
        Scriptor
      </Link>
      <nav className="nav">
        {LINKS.map((l) => (
          <Link key={l.href} href={l.href} className={path === l.href ? "active" : ""}>{l.label}</Link>
        ))}
      </nav>
      <div className="engine-status" title="Python engine at /api">
        <i className={`dot ${up === null ? "" : up ? "ok" : "bad"}`} />
        <span>{up === null ? "checking engine" : up ? "engine online" : "engine offline"}</span>
      </div>
    </header>
  );
}

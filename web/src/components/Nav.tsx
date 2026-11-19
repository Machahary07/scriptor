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
        <span className="brand-word">Script<i>or</i></span>
        <span className="brand-tag">writes like you</span>
      </Link>
      <nav className="nav">
        {LINKS.map((l, i) => (
          <Link key={l.href} href={l.href} className={path === l.href ? "active" : ""}>
            <span className="num">0{i + 1}</span>{l.label}
          </Link>
        ))}
      </nav>
      <div className="engine-status" title="Python engine at /api">
        <i className={`dot ${up === null ? "" : up ? "ok" : "bad"}`} />
        <span>{up === null ? "checking engine" : up ? "engine online" : "engine offline"}</span>
      </div>
    </header>
  );
}

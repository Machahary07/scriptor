import type { Metadata } from "next";
import { Caveat, Fraunces, IBM_Plex_Mono, Manrope } from "next/font/google";
import { Nav } from "@/components/Nav";
import { JobProvider } from "@/lib/job-store";
import "./globals.css";

const display = Fraunces({ subsets: ["latin"], style: ["normal", "italic"], variable: "--font-display" });
const body = Manrope({ subsets: ["latin"], variable: "--font-body" });
const mono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-mono" });
const hand = Caveat({ subsets: ["latin"], variable: "--font-hand" });

export const metadata: Metadata = {
  title: "Scriptor",
  description: "AI handwriting synthesis with a virtual pen-plotter simulator",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${mono.variable} ${hand.variable}`}>
      <body>
        <JobProvider>
          <div className="shell">
            <Nav />
            <main>{children}</main>
          </div>
        </JobProvider>
      </body>
    </html>
  );
}

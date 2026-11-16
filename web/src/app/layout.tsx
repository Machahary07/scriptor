import type { Metadata } from "next";
import { Nav } from "@/components/Nav";
import { JobProvider } from "@/lib/job-store";
import "./globals.css";

export const metadata: Metadata = {
  title: "Scriptor",
  description: "AI handwriting synthesis with a virtual pen-plotter simulator",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
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

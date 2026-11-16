import type { NextConfig } from "next";

const ENGINE_URL = process.env.ENGINE_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${ENGINE_URL}/api/:path*` }];
  },
};

export default nextConfig;

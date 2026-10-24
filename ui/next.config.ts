import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  agentRules: false, // sonst legt `next dev` bei jedem Start eine AGENTS.md an
};

export default nextConfig;

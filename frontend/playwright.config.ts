import fs from "node:fs";
import { defineConfig } from "@playwright/test";

// End-to-end tests run against a running stack in DEMO mode (see docs/TESTING.md):
//   E2E_BASE_URL (default http://localhost:3000), E2E_EMAIL, E2E_PASSWORD.
const chromium = process.env.PLAYWRIGHT_CHROMIUM_PATH || "/opt/pw-browsers/chromium";

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 60_000,
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: process.env.E2E_BASE_URL || "http://localhost:3000",
    trace: "retain-on-failure",
    ignoreHTTPSErrors: process.env.E2E_IGNORE_HTTPS_ERRORS === "1", // local Caddy uses an internal CA
    launchOptions: fs.existsSync(chromium) ? { executablePath: chromium } : {},
  },
});

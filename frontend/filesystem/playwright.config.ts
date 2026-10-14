import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests', workers: 1, timeout: 45000,
  expect: { timeout: 10000 },
  use: {
    baseURL: 'http://127.0.0.1:19620', viewport: { width: 1500, height: 1000 },
    trace: 'retain-on-failure',
    launchOptions: { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE },
  },
  webServer: {
    command: '.venv/bin/python -m streamlit run tests/filesystem_browser_app.py --server.port 19620 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false',
    cwd: '../..', url: 'http://127.0.0.1:19620/_stcore/health', reuseExistingServer: false,
  },
});

import { expect, type Page, test } from "@playwright/test";

// Credentials of an admin on the stack under test; never hard-code real ones.
const EMAIL = process.env.E2E_EMAIL ?? "";
const PASSWORD = process.env.E2E_PASSWORD ?? "";
test.skip(!EMAIL || !PASSWORD, "set E2E_EMAIL and E2E_PASSWORD");

async function login(page: Page) {
  await page.goto("/login");
  await page.getByLabel("Email").fill(EMAIL);
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: /sign in/i }).click();
  await page.waitForURL("/");
}

test("unauthenticated users are sent to the login page", async ({ page }) => {
  await page.goto("/news");
  await expect(page).toHaveURL(/\/login/);
});

test("wrong password is rejected", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Email").fill(EMAIL);
  await page.getByLabel("Password").fill("definitely-wrong-password");
  await page.getByRole("button", { name: /sign in/i }).click();
  await expect(page.getByRole("alert")).toBeVisible();
});

test.describe("signed in (demo mode)", () => {
  test.beforeEach(async ({ page }) => login(page));

  test("dashboard lists the watchlist and labels simulated data", async ({ page }) => {
    await expect(page.getByText(/DEMO MODE/)).toBeVisible();
    const table = page.getByTestId("pair-table");
    await expect(table).toBeVisible();
    await expect(table.getByRole("link", { name: /KO \/ PEP/ })).toBeVisible();
    expect(await table.getByRole("row").count()).toBeGreaterThanOrEqual(12); // header + 11 pairs
  });

  test("pair detail shows scores, charts and research tabs", async ({ page }) => {
    await page.goto("/pairs/KO-PEP");
    await expect(page.getByRole("heading", { name: /KO/ })).toBeVisible();
    for (const tab of ["Price relationship", "Splittability history", "AI research", "Prediction tracking"]) {
      await page.getByRole("tab", { name: tab }).click();
      await expect(page.getByRole("tabpanel")).toBeVisible();
    }
    await page.getByRole("tab", { name: "AI research" }).click();
    await expect(page.getByText(/SIMULATED/).first()).toBeVisible();
  });

  test("news can be filtered and expanded to show verified evidence", async ({ page }) => {
    await page.goto("/news");
    await page.getByLabel("Pair").selectOption("KO-PEP");
    const list = page.getByTestId("news-list");
    await expect(list.locator("> *").first()).toBeVisible();
    await list.locator("> *").first().locator("button").first().click();
    await expect(page.getByTestId("event-detail").first()).toBeVisible();
  });

  test("performance page separates live, backtest and reported results", async ({ page }) => {
    await page.goto("/performance");
    await expect(page.getByTestId("comparison")).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText("Original experiment (reported)")).toBeVisible();
    await page.getByRole("tab", { name: "Reported benchmark" }).click();
    await expect(page.getByTestId("reported-benchmark")).toBeVisible();
  });

  test("bot can be paused and resumed", async ({ page }) => {
    await page.goto("/bot");
    await expect(page.getByTestId("schedules")).toBeVisible();
    await page.getByTestId("pause").click();
    await expect(page.getByTestId("resume")).toBeVisible();
    await page.getByTestId("resume").click();
    await expect(page.getByTestId("pause")).toBeVisible();
  });

  test("a pair can be added, archived and removed", async ({ page, request, context }) => {
    await page.goto("/settings");
    await page.getByRole("button", { name: /add pair/i }).click();
    await page.getByLabel("Ticker A").fill("AAPL");
    await page.getByLabel("Ticker B").fill("MSFT");
    await page.getByRole("button", { name: /^add$|create|add pair/i }).last().click();
    const row = page.getByTestId("settings-pairs").getByRole("row", { name: /AAPL-MSFT/ });
    await expect(row).toBeVisible();
    await row.getByRole("button", { name: "Archive" }).click();
    await expect(row.getByText("archived", { exact: false })).toBeVisible();
    // clean up through the API (hard delete is allowed because the pair has no history yet)
    const cookies = await context.cookies();
    const res = await request.delete("/api/v1/pairs/AAPL-MSFT?hard=true", {
      headers: { "X-Requested-With": "splittability", Cookie: cookies.map((c) => `${c.name}=${c.value}`).join("; ") },
    });
    expect(res.status()).toBe(200);
  });

  test("settings never display API keys", async ({ page }) => {
    await page.goto("/settings#system");
    await page.getByRole("tab", { name: "System & account" }).click();
    await expect(page.getByText(/never sent to the browser/i)).toBeVisible();
    const html = await page.content();
    expect(html).not.toMatch(/sk-ant-|ANTHROPIC_API_KEY=/);
  });
});

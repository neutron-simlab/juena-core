import { test, expect, type Page } from '@playwright/test';
const api = 'http://127.0.0.1:19621';
const panel = (page: Page) => page.getByRole('region', { name: 'Conversation files', exact: true });
const entry = (page: Page, path: string) => panel(page).locator(`button.entry[title="${path}"]`);
const editor = (page: Page) => panel(page).locator('.editor');
const value = (page: Page) => editor(page).evaluate((el: any) => el.env.editor.getValue());
const edit = async (page: Page, text: string) => {
  await editor(page).locator('textarea').focus();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.insertText(text);
  await expect(panel(page).getByText('Unsaved changes', { exact: true })).toBeVisible();
};
const control = (page: Page, action: string) => page.request.post(`${api}/test/control/${action}`);
const stats = async (page: Page) => (await page.request.get(`${api}/test/stats`)).json();
const choose = async (page: Page, name: string) => {
  await page.getByRole('combobox').click();
  await page.getByRole('option', { name, exact: true }).click();
  await expect(panel(page)).toHaveAttribute('data-conversation', name);
};

test.beforeEach(async ({ page, context }) => {
  await context.addCookies([{ name: 'juena_session', value: 'browser-test-cookie', url: api, httpOnly: true, sameSite: 'Lax' }]);
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Streaming explorer test' })).toBeVisible();
  await control(page, 'reset');
  await page.reload();
  for (const path of ['/workspace', '/inputs', '/findings']) await expect(entry(page, path)).toBeVisible();
  await expect(panel(page).getByRole('navigation', { name: 'Folder path' })).toHaveText('/');
  await expect(panel(page).getByText('View options', { exact: true })).toHaveCount(0);
  await entry(page, '/workspace').click();
  await expect(entry(page, '/workspace/analysis.py')).toBeVisible();
});

test('uncached navigation and composer handoff during SSE, then completion refresh', async ({ page }) => {
  await page.getByPlaceholder('Message').fill('Start the slow response');
  await page.getByPlaceholder('Message').press('Enter');
  await expect.poll(async () => (await stats(page)).started).toBe(1);
  await entry(page, '/workspace/outputs').click();
  await entry(page, '/workspace/outputs/nested').click();
  await entry(page, '/workspace/outputs/nested/result.txt').click();
  await expect.poll(() => value(page)).toBe('uncached result\n');
  expect((await stats(page)).completed).toBe(0);
  await edit(page, 'unsaved during execution');
  await panel(page).getByRole('button', { name: 'Save', exact: true }).click();
  await expect(panel(page).getByText(/Files are busy/)).toBeVisible({ timeout: 3000 });
  await panel(page).getByRole('button', { name: 'Discard', exact: true }).click();
  await panel(page).getByRole('button', { name: 'Ask about this file', exact: true }).click();
  await expect(panel(page).getByText('Path queued for your next message.')).toBeVisible();
  await expect.poll(async () => (await stats(page)).completed, { timeout: 20000 }).toBe(1);
  await expect(page.getByPlaceholder('Message')).toHaveValue('Please look at `/workspace/outputs/nested/result.txt`: ');
  await expect(entry(page, '/workspace/outputs/nested/finished.txt')).toBeVisible();
  await expect(page.getByText('Composer insertions: 1')).toBeVisible();
  await page.getByRole('button', { name: 'Rerun page', exact: true }).click();
  await expect(page.getByText('Composer insertions: 1')).toBeVisible();
  expect(await stats(page)).toMatchObject({ started: 1, completed: 1, cancelled: 0 });
});

test('folder window, breadcrumbs, pagination, narrow layout, empty conversation', async ({ page }) => {
  const box = (await panel(page).boundingBox())!;
  expect(box.width).toBeLessThan(260);
  expect(await panel(page).evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
  const kinds = await panel(page).locator('.entry').allTextContents();
  expect(kinds[0]).toContain('.scratch');
  await expect(panel(page).getByRole('navigation', { name: 'Folder path' })).toHaveText('/ workspace');
  await entry(page, '/workspace/many').click();
  await expect(entry(page, '/workspace/many/file_000.txt')).toBeVisible();
  await panel(page).getByRole('button', { name: 'Next', exact: true }).click();
  await expect(entry(page, '/workspace/many/file_104.txt')).toBeVisible();
  await expect(entry(page, '/workspace/many/file_000.txt')).toHaveCount(0);
  await panel(page).getByRole('button', { name: 'Previous', exact: true }).click();
  await expect(entry(page, '/workspace/many/file_000.txt')).toBeVisible();
  await panel(page).getByRole('navigation', { name: 'Folder path' }).getByRole('button', { name: 'workspace', exact: true }).click();
  await expect(entry(page, '/workspace/analysis.py')).toBeVisible();
  await panel(page).getByRole('button', { name: '← Back', exact: true }).click();
  await expect(entry(page, '/workspace/many/file_000.txt')).toBeVisible();
  await panel(page).getByRole('button', { name: '↑ Up', exact: true }).click();
  await expect(entry(page, '/workspace/analysis.py')).toBeVisible();
  await choose(page, 'new');
  await expect(panel(page).getByText(/Send your first message/)).toBeVisible();
  const count = (await stats(page)).filesystem_requests;
  await panel(page).getByRole('button', { name: 'Refresh', exact: true }).click();
  await expect(panel(page).getByText(/Send your first message/)).toBeVisible();
  expect((await stats(page)).filesystem_requests).toBe(count);
});

test('draft isolation, save/discard guard, stale revisions and pending approval', async ({ page }) => {
  await entry(page, '/workspace/analysis.py').click();
  await expect.poll(() => value(page)).toBe("value = 'one'\n");
  await edit(page, 'value = 42\n');
  await entry(page, '/workspace/outputs').click();
  await expect(panel(page).getByRole('dialog')).toBeVisible();
  await panel(page).getByRole('button', { name: 'Stay here', exact: true }).click();
  await choose(page, 'two');
  await entry(page, '/workspace').click();
  await entry(page, '/workspace/analysis.py').click();
  await expect.poll(() => value(page)).toBe("value = 'two'\n");
  await choose(page, 'one');
  await expect.poll(() => value(page)).toBe('value = 42\n');
  await panel(page).getByRole('button', { name: 'Refresh', exact: true }).click();
  await expect.poll(() => value(page)).toBe('value = 42\n');
  await control(page, 'pause');
  await panel(page).getByRole('button', { name: 'Save', exact: true }).click();
  await expect(panel(page).getByText(/Files are busy/)).toBeVisible();
  await control(page, 'unpause');
  await control(page, 'mutate');
  await panel(page).getByRole('button', { name: 'Save', exact: true }).click();
  await expect(panel(page).getByRole('status').filter({ hasText: /changed/i })).toBeVisible();
  await expect.poll(() => value(page)).toBe('value = 42\n');
  await panel(page).getByRole('button', { name: 'Reload file', exact: true }).click();
  await panel(page).getByRole('button', { name: 'Discard and continue', exact: true }).click();
  await expect.poll(() => value(page)).toBe('agent changed this\n');
  await edit(page, 'saved by user\n');
  await entry(page, '/workspace/outputs').click();
  await panel(page).getByRole('button', { name: 'Save and continue', exact: true }).click();
  await expect(entry(page, '/workspace/outputs/nested')).toBeVisible();
  const response = await page.request.get(`${api}/threads/one/filesystem/download?path=/workspace/analysis.py`);
  expect(await response.text()).toBe('saved by user\n');
});

test('cached views under execution lock and prompt busy response for uncached requests', async ({ page }) => {
  await entry(page, '/workspace/analysis.py').click();
  await expect.poll(() => value(page)).toBe("value = 'one'\n");
  await control(page, 'lock');
  await panel(page).getByRole('button', { name: 'Refresh', exact: true }).click();
  await expect(panel(page).getByText(/Files are busy/).first()).toBeVisible({ timeout: 3000 });
  await expect.poll(() => value(page)).toBe("value = 'one'\n");
  await expect(panel(page).getByText(/Last loaded/)).toBeVisible();
  await entry(page, '/workspace/outputs').click();
  await expect(panel(page).getByText(/Files are busy/).first()).toBeVisible({ timeout: 3000 });
  await expect(entry(page, '/workspace/analysis.py')).toHaveCount(0);
  await control(page, 'unlock');
  await panel(page).getByRole('button', { name: 'Refresh', exact: true }).click();
  await expect(entry(page, '/workspace/outputs/nested')).toBeVisible();
});

test('previews, downloads, scratch, read-only inputs and notes', async ({ page }) => {
  await entry(page, '/workspace/preview.html').click();
  await expect.poll(() => value(page)).toContain('<script>');
  expect(await page.evaluate(() => (window as any).unsafePreview)).toBeUndefined();
  await entry(page, '/workspace/notebook.ipynb').click();
  await expect.poll(() => value(page)).toContain('"cells"');
  await entry(page, '/workspace/picture.png').click();
  await expect(panel(page).getByRole('img')).toBeVisible();
  await control(page, 'lock');
  await panel(page).getByRole('button', { name: 'Refresh', exact: true }).click();
  await expect(panel(page).getByText(/Files are busy/).first()).toBeVisible();
  await expect(panel(page).getByRole('img')).toBeVisible();
  await control(page, 'unlock');
  await entry(page, '/workspace/binary.bin').click();
  const downloaded = page.waitForEvent('download');
  await panel(page).getByRole('button', { name: 'Download', exact: true }).click();
  expect((await downloaded).suggestedFilename()).toBe('binary.bin');
  await entry(page, '/workspace/large.txt').click();
  await expect(panel(page).getByText(/Preview truncated/)).toBeVisible();
  await expect(panel(page).getByRole('button', { name: 'Save', exact: true })).toHaveCount(0);
  await expect(entry(page, '/workspace/.scratch')).toBeVisible();
  await panel(page).getByRole('button', { name: '↑ Up', exact: true }).click();
  await entry(page, '/inputs').click();
  await entry(page, '/inputs/message.txt').click();
  await expect.poll(() => value(page)).toBe('pasted input');
  await expect(panel(page).getByRole('button', { name: 'Save', exact: true })).toHaveCount(0);
  await panel(page).getByRole('button', { name: '↑ Up', exact: true }).click();
  await entry(page, '/findings').click();
  await entry(page, '/findings/note.md').click();
  await expect.poll(() => value(page)).toBe('specialist observation');
  await expect(panel(page).getByText(/Specialist observations/)).toBeVisible();
  await expect(panel(page).getByRole('button', { name: 'Save', exact: true })).toHaveCount(0);
});

test('late responses cannot cross conversations; auth loss and new sessions clear drafts', async ({ page }) => {
  let release!: () => void;
  const held = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/threads/one/filesystem/file?**', async route => {
    const response = await route.fetch();
    await held;
    await route.fulfill({ response });
  });
  await entry(page, '/workspace/analysis.py').click();
  await expect(panel(page).getByText('Loading file…')).toBeVisible();
  await choose(page, 'two');
  await entry(page, '/workspace').click();
  await entry(page, '/workspace/analysis.py').click();
  await expect.poll(() => value(page)).toBe("value = 'two'\n");
  release();
  await page.unrouteAll({ behavior: 'wait' });
  await edit(page, 'private draft');
  await control(page, 'expire');
  await panel(page).getByRole('button', { name: 'Refresh', exact: true }).click();
  await expect(panel(page).getByRole('alert')).toContainText('expired');
  await expect(editor(page)).toHaveCount(0);
  await control(page, 'reset');
  await page.getByRole('button', { name: 'New session', exact: true }).click();
  await entry(page, '/workspace').click();
  await entry(page, '/workspace/analysis.py').click();
  await expect.poll(() => value(page)).toBe("value = 'two'\n");
});

test('persistent header, rerun drafts and no idle polling', async ({ page }) => {
  await entry(page, '/workspace/analysis.py').click();
  await expect.poll(() => value(page)).toBe("value = 'one'\n");
  await edit(page, 'keep this draft\n');
  await panel(page).getByRole('button', { name: 'Files', exact: true }).click();
  await expect(panel(page).getByRole('button', { name: 'Files', exact: true })).toHaveAttribute('aria-expanded', 'false');
  await expect(editor(page)).toHaveCount(0);
  await page.getByRole('button', { name: 'Rerun page', exact: true }).click();
  await panel(page).getByRole('button', { name: 'Files', exact: true }).click();
  await expect.poll(() => value(page)).toBe('keep this draft\n');
  await expect(panel(page).locator('.directory')).toHaveAttribute('aria-busy', 'false');
  const count = (await stats(page)).filesystem_requests;
  // An idle interval is intentional: no timer should fetch filesystem contents.
  await page.waitForTimeout(3000);
  expect((await stats(page)).filesystem_requests).toBe(count);
});

test('permission failures keep drafts and do not invalidate other conversations', async ({ page }) => {
  await entry(page, '/workspace/analysis.py').click();
  await expect.poll(() => value(page)).toBe("value = 'one'\n");
  await edit(page, 'keep my edit');
  await page.route('**/threads/one/filesystem/file', async route => {
    if (route.request().method() === 'PUT') await route.fulfill({ status: 403, contentType: 'application/json',
      body: JSON.stringify({ detail: { code: 'origin_not_allowed', message: 'Browser origin is not allowed' } }) });
    else await route.continue();
  });
  await panel(page).getByRole('button', { name: 'Save', exact: true }).click();
  await expect(panel(page).getByText('Browser origin is not allowed', { exact: true })).toBeVisible();
  expect(await value(page)).toBe('keep my edit');
  await expect(panel(page).getByRole('alert')).toHaveCount(0);
  await choose(page, 'two');
  await entry(page, '/workspace').click();
  await entry(page, '/workspace/analysis.py').click();
  await expect.poll(() => value(page)).toBe("value = 'two'\n");
  await choose(page, 'one');
  await expect.poll(() => value(page)).toBe('keep my edit');
});

test('session expiry on Save explicitly reports discarded edits', async ({ page }) => {
  await entry(page, '/workspace/analysis.py').click();
  await expect.poll(() => value(page)).toBe("value = 'one'\n");
  await edit(page, 'private unsaved edit');
  await control(page, 'expire');
  await panel(page).getByRole('button', { name: 'Save', exact: true }).click();
  await expect(panel(page).getByRole('alert')).toContainText('Your unsaved edits were discarded');
  await expect(editor(page)).toHaveCount(0);
});

test('typing a large draft stays in the browser without Streamlit updates', async ({ page }) => {
  await entry(page, '/workspace/analysis.py').click();
  await expect.poll(() => value(page)).toBe("value = 'one'\n");
  await page.waitForTimeout(600);
  // Existing Streamlit socket: instrument component bridge traffic at the socket boundary.
  await page.evaluate(() => {
    const original = WebSocket.prototype.send;
    (window as any).draftFrames = 0;
    WebSocket.prototype.send = function (...args) { (window as any).draftFrames++; return original.apply(this, args); };
  });
  await edit(page, 'x'.repeat(700000));
  await page.waitForTimeout(800);
  expect(await page.evaluate(() => (window as any).draftFrames)).toBe(0);
  await choose(page, 'two');
  await choose(page, 'one');
  await expect.poll(() => value(page)).toBe('x'.repeat(700000));
});

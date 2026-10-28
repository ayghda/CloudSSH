import { expect, test } from './fixtures';

test('GitHub-only login preserves the custom card and its isolated theme switch', async ({ page }) => {
  await page.route('**/api/config', (route) => route.fulfill({ json: {
    turnstileEnabled: false,
    sitekey: '',
    githubAuthEnabled: true,
    githubAuthRequired: true,
    authMode: 'github',
  } }));

  await page.goto('/?lang=zh-CN');

  const auth = page.locator('#auth-section');
  const login = page.locator('#custom-github-login-btn');
  await expect(login).toBeVisible();
  await expect(login).toHaveAttribute('href', '/api/auth/github');
  await expect(page.locator('#connection-form-container')).toBeHidden();
  await expect(auth).toHaveAttribute('data-login-appearance', 'dark');

  const applicationTheme = await page.locator('html').getAttribute('data-theme');
  await page.locator('#custom-theme-toggle').click();
  await expect(auth).toHaveAttribute('data-login-appearance', 'light');
  expect(await page.locator('html').getAttribute('data-theme')).toBe(applicationTheme);

  await page.reload();
  await expect(auth).toHaveAttribute('data-login-appearance', 'light');
  await expect(login).toBeVisible();
});

test('authenticated server controls remain wired after changing only the login view', async ({ page }) => {
  await page.route('**/api/auth/me', (route) => route.fulfill({ json: {
    id: 1,
    github_id: 42,
    username: 'tester',
    avatar_url: '',
  } }));
  await page.route('**/api/servers', (route) => route.fulfill({ json: [{
    id: 1,
    user_id: 1,
    name: 'My SSH Host',
    host: 'ssh.example.com',
    port: 22,
    username: 'tester',
    auth_method: 'password',
    region: null,
    tags: [],
    created_at: '',
    updated_at: '',
  }] }));

  await page.goto('/?lang=zh-CN');
  await expect(page.locator('#auth-section')).toBeHidden();
  await expect(page.locator('#user-space-section')).toBeVisible();
  await expect(page.locator('#server-grid .server-card')).toHaveCount(1);
  await page.locator('#server-sort').selectOption('name');
  await expect(page.locator('#server-sort')).toHaveValue('name');

  await page.locator('#add-server-btn').click();
  await expect(page.locator('#server-modal')).toBeVisible();
  await page.locator('#modal-transport-tab-tunnel').click();
  await expect(page.locator('#server-cf-access-section')).toBeVisible();
  await page.locator('#modal-close-btn').click();
  await expect(page.locator('#server-modal')).toBeHidden();

  // These elements were missing in the supplied replacement HTML.
  for (const selector of [
    '#user-theme-segmented-container',
    '#mobile-shortcuts-config-btn',
    '#terminal-drawer-segmented-bar',
    '#snippet-toggle-btn',
    '#mobile-sftp-btn',
    '#mobile-agent-btn',
  ]) {
    await expect(page.locator(selector)).toHaveCount(1);
  }
});

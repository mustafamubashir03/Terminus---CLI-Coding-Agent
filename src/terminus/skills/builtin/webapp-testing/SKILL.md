---
name: webapp-testing
description: >
  Use when verifying a web application in a real browser: checking that a UI change
  actually works, reproducing a bug a user reported, walking a user flow end to end,
  taking screenshots to confirm visual output, or confirming a frontend behaves the same
  in a browser as it does in code. Triggers on "test the UI", "check this page works",
  "reproduce this bug", "take a screenshot", "verify the button does X", or on any
  frontend task whose acceptance criteria are visual or interactive.
  Do NOT use for pure unit, integration, or API testing where a test runner is the
  right tool. Do NOT claim a browser test passed if no browser tooling is available.
version: 1.0.0
license: Apache-2.0
author: Terminus (adapted from anthropics/skills)
repository: anthropics/skills
tags: [testing, browser, e2e, playwright, screenshot, verification, frontend]
---

# Web Application Testing

Verify web behaviour in a browser, because that is where the user meets the code.
Unit tests prove functions; only a browser proves the page.

## Check the prerequisites before promising anything

Browser verification needs a way to drive a browser. Check what is actually available
before planning around it:

```bash
# Is a browser automation driver installed?
npx playwright --version 2>/dev/null || echo "no playwright"
python -c "import playwright; print('python playwright present')" 2>/dev/null || echo "no python playwright"
# Is there a test command already configured?
cat package.json 2>/dev/null | grep -A5 '"scripts"'
```

Then check the application can actually be started and reached:

```bash
# what is the dev command, and what port?
grep -E '"(dev|start|serve)"' package.json 2>/dev/null
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 2>/dev/null
```

**If no browser tooling is available, say so plainly.** Report that the change was
made and that visual verification could not be performed, and name what is missing.
Do not describe a UI as verified because the code looks right. Do not write a
Playwright test and call the feature tested when it was never executed.

If a driver is available but the app does not start, that is the first bug to report.

## The loop

Work in a cycle, not in a straight line:

1. **Reproduce** - get the page or flow into the exact state where the problem shows.
   Record what you observe before changing anything.
2. **Diagnose** - is it the markup, the styles, a failed request, a state bug? Check the
   browser console and network panel; most UI bugs are one of those, not the layout.
3. **Fix the cause** - change the code that causes the symptom.
4. **Retest the same path** - the exact steps that reproduced it.
5. **Check the neighbours** - the states around the fix: loading, empty, error, and a
   narrow viewport.

Skipping straight from "it looks wrong" to an edit is the most common way an
unrelated fix gets shipped.

## What to check every time

- The specific interaction you changed actually does what it should.
- Console is free of errors and warnings introduced by the change.
- No failed network requests (4xx/5xx) on the visited pages.
- Loading states appear and are removed.
- Empty and error states render sensibly rather than as a blank page.
- Keyboard: the flow is reachable by tabbing, and focus is visible.
- One narrow viewport (around 375px) and one wide (around 1440px).

## Screenshots

Capture when visual confirmation is the point. Give the file a meaningful name, and
actually look at it before reporting - a screenshot you did not open proves nothing.

```bash
npx playwright screenshot --viewport-size=1280,800 http://localhost:3000/dashboard out/dashboard.png
```

Save screenshots to a build or temp directory, not into source control.

## Writing a durable test

When the check is worth keeping, write it as a real test rather than a manual step:

- target a user-visible outcome, not an implementation detail
- make it deterministic: wait for a specific condition, never a fixed sleep
- give it a descriptive name that says what behaviour broke if it fails
- keep the selector robust - prefer role, label, or test id over CSS class
- make it independent of other tests

```javascript
test('checkout rejects an expired card', async ({ page }) => {
  await page.goto('/checkout');
  await page.getByLabel('Card number').fill('4000000000000002');
  await page.getByRole('button', { name: 'Pay' }).click();
  await expect(page.getByText('Card declined')).toBeVisible();
});
```

## Report honestly

State what you actually did: which flows you exercised in which browser, what you
observed, and what you could not check. "Verified in Chromium at 1280px, error and
empty states not covered" is useful. "Tests pass" when you never ran them is not.

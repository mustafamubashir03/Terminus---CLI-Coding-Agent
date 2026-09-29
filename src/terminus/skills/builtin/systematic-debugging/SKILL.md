---
name: systematic-debugging
description: >
  Use when something is broken, failing, crashing, flaky, regressed, slow, or behaving
  unexpectedly, and the cause is not already known. Triggers on "this is failing", "it
  used to work", "it works locally but not in CI", "intermittent", "flaky", "why does
  this happen", "regression", a stack trace, a 500 or 502, a traceback or exception,
  production errors, "since the last deploy", or a pasted error pasted without
  explanation. Also use when a previous fix did not hold.
  Do NOT use for features that work as specified, for code review, or for adding
  behaviour that does not exist yet. Do NOT use when the cause is already established
  and the remaining work is the fix itself.
version: 1.0.0
license: Apache-2.0
author: Terminus
tags: [debugging, bugs, root-cause, regression, flaky, diagnosis]
---

# Systematic Debugging

Debugging is a process of elimination, not a sequence of guesses. The failure mode this
skill exists to prevent is:

```
guess -> edit -> looks better -> ship
```

That loop feels productive and frequently ships a symptom fix, a second bug, or both.
Work in this order instead:

```
symptom -> reproduce -> evidence -> hypotheses -> test -> root cause -> minimal fix -> regression test
```

## 1. State the symptom precisely

Before touching code, write down what is actually observed:

- the exact error, verbatim, not paraphrased
- the exact input or action that triggers it
- what you expected instead
- when it started
- whether it is constant or intermittent, and if intermittent, the rate

"I need to fix the thing" is not a symptom. "POST /orders returns 500 with
`NullPointerException` for orders created after 14:00 UTC, reproducible on every
attempt since the 09:10 deploy" is.

## 2. Reproduce it

An unreproduced bug cannot be fixed or verified. Get it to happen on demand.

- If it only happens in CI, capture the environment differences.
- If it is intermittent, find the conditions that make it more frequent: timing,
  concurrency, data volume, specific input values.
- If you genuinely cannot reproduce it, say so and stop. Guessing at a bug you cannot
  trigger is speculation, and you should say that instead of proceeding.

## 3. Collect evidence

Read the failure rather than guessing at it. In rough priority order:

- the full error and stack trace, including the frames above the visible one
- logs around the failure, with timestamps
- the inputs and state at the moment of failure
- recent changes to the code path involved (`git log`, `git diff`)
- whether the failure correlates with deploys, config, or data changes

Then deliberately narrow it:

- **Bisect the space.** Halve the inputs, the code path, or the history. Repeat.
- **Instrument.** Add logging or a breakpoint at the boundary where correct behaviour
  becomes incorrect. One good log line beats twenty theories.
- **Compare with a working case.** What differs between the input that fails and the
  one that works? That difference is the bug.

## 4. Form hypotheses and test them

Write down candidate causes, most likely first, and give each a cheap test that would
confirm or rule it out.

Rank by (likelihood x cost of being wrong). Test the cheapest discriminating case
first. Rules:

- one hypothesis at a time, so a result means something
- a test that cannot fail rules nothing out
- do not start editing to "try something" while still diagnosing

## 5. Identify the root cause

The root cause is the earliest decision that made the bug possible - not the line that
raised the exception. `NullPointerException` on line 88 is the symptom; the root cause
is whatever allowed `user` to be `None` there.

State the root cause as a sentence before fixing. If you cannot, you have a
correlation, not a cause, and the fix will not hold.

## 6. Fix minimally

- Change the cause, not the symptom. Wrapping the crash in a try/except that returns a
  default hides the bug until it corrupts data later.
- Keep the diff as small as the bug allows. Unrelated "while I was here" changes make
  the fix impossible to review and impossible to revert.
- Consider whether the same mistake exists elsewhere and is worth fixing now.

## 7. Verify and prevent recurrence

- Re-run the exact reproduction from step 2. It must now pass.
- Add a regression test that fails without the fix and passes with it. This is what
  stops it coming back.
- Run the surrounding test suite to confirm nothing else broke.
- If the root cause was a missing invariant, consider whether that invariant should be
  asserted or enforced at a boundary.

## Discipline

- **Do not** change code and observe an improvement; that is not a fix.
- **Do not** disable the failing test.
- **Do not** add a retry to paper over a race.
- **Do not** catch an exception you do not understand.
- **Do not** refactor while debugging a live bug. Fix first, then clean up.
- **When stuck**, say what you have ruled out. A well-stated dead end is progress and
  it tells the user what to try next.

## Reporting

Give the user: the root cause, the evidence that supports it, what you changed, and
what you verified. If you never found the root cause, say that clearly instead of
presenting the last change you tried as the answer.

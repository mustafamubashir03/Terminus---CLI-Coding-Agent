---
name: test-driven-development
description: >
  Use when adding new behaviour, fixing a bug, or changing existing behaviour where a
  test can meaningfully pin the result - especially when the requirement is precise and
  the change touches logic rather than layout or copy. Triggers on "add a test",
  "write tests for", "make sure this is covered", "regression test", or any change to
  branching logic, parsing, calculations, or error handling.
  Do NOT force this workflow onto exploratory work, throwaway prototypes, spikes,
  large exploratory refactors, or changes where the specification is still being
  discovered. It is a tool for when you already know what correct looks like.
negative_triggers:
  - design the schema
  - write a migration
  - row level security
version: 1.0.0
license: Apache-2.0
author: Terminus
tags: [testing, tdd, regression, unit-tests, quality, refactoring]
---

# Test-Driven Development

Write the test when you know what correct looks like. Skip it when you do not.

## First, decide whether this task wants TDD

TDD pays off when the requirement is clear and the behaviour is checkable. It is the
wrong tool when:

- you are exploring an unfamiliar codebase or an unknown design
- you are prototyping to discover whether an approach works at all
- the specification is still being negotiated with the user
- the change is a pure refactor with unchanged behaviour (characterisation tests
  aside)
- the deliverable is prose, configuration, or visual styling with no assertions to make

For those, explore or implement first, then add a test once the shape is known. Doing
test-first on a spike produces tests that encode guesses.

**Bug fixes are the strongest case.** Write the failing test before the fix: it
proves the bug existed, proves the fix works, and prevents it returning.

## The cycle

### 1. Write one failing test

Smallest thing that expresses the next slice of behaviour. Name it for the behaviour,
not the function: `rejects_an_expired_card` beats `test_process_payment_2`.

### 2. Run it and watch it fail

This step is not optional ceremony. A test that passes before you write the
implementation is not testing anything, or is testing something already true. Watch
it fail for the *right* reason - a syntax error is not a valid red.

### 3. Write the least code that passes

No speculative generality, no options you do not need yet, no abstraction for the
third implementation that does not exist. Hard-coding is acceptable while the test
drives out the real shape.

### 4. Run it and watch it pass

### 5. Refactor with the tests green

Now clean up names, remove duplication, and simplify - with the safety net in place.
This is the only safe moment to refactor.

### 6. Repeat

## What to test

Test behaviour through the public surface. A test coupled to private internals breaks
on every refactor and forces you to keep rewriting it.

Prioritise:

1. **Boundary and error paths** - empty, null, zero, negative, huge, malformed,
   duplicate, missing permissions, network failure. These are where the bugs are.
2. **Invariants** - things that must always hold regardless of input.
3. **Regression cases** - one per bug you have fixed, named after the bug.
4. **Happy paths** - still necessary, but the least interesting tests you will write.

Do not chase coverage percentage. A suite of 100% coverage that asserts nothing
protects nothing. A suite of 20 tests that pins real behaviour is worth more.

## Test doubles

- **Prefer the real thing** where it is fast and deterministic. Use a real database in
  a container over a hand-written mock of one.
- **Fake at boundaries you do not own** - network, third-party APIs, clock, filesystem.
- **Do not mock the thing under test.** Mocking the subject of the test means asserting
  that the mock was called.
- **Make time explicit.** Inject a clock rather than sleeping; tests that sleep are
  slow and flaky.

## Keep the suite fast and honest

- Fast tests get run. A 40-minute suite gets skipped.
- Isolate tests from each other; shared mutable state is the top cause of flakes.
- A test that fails intermittently should be treated as a real bug, not deleted.
- Never weaken or delete a failing test to make the suite green. Fix the code, or
  document explicitly why the test is wrong and change it deliberately.

## If you cannot test it

Some things are not unit-testable - visual layout, third-party integrations, timing.
Say so, and verify them another way (browser check, manual reproduction, contract
test) rather than pretending coverage you do not have.

---
name: frontend-design
description: >
  Use when building or substantially redesigning a user interface: web pages, landing
  pages, marketing pages, dashboards, admin panels, application screens, or component
  libraries, where visual hierarchy, typography, spacing, colour, responsive composition
  and interaction quality are the substance of the task. Triggers on requests to "design",
  "build a page", "make it look good", "restyle", "modernise the UI", "polish the
  interface", or to any task whose deliverable is a screen a person will look at.
  Do NOT use for backend, API, database, CLI, or infrastructure work. Do NOT use merely
  because a task happens to mention React, CSS, or a component file - this skill is about
  the quality of the result, not the technology used to produce it.
negative_triggers:
  - review my ui
  - check my ui
  - audit my
  - audit the ui
  - inspect my
  - which component library
  - what should this page be made of
  - empty state
  - loading skeleton
  - add animations
  - transitions are janky
version: 1.0.0
license: Apache-2.0
author: Terminus (adapted from anthropics/skills)
repository: anthropics/skills
tags: [frontend, ui, design, css, layout, typography, accessibility, responsive]
---

# Frontend Design

Build interfaces that look like someone made deliberate decisions. The default failure
mode of generated UI is competent-but-generic: default fonts, arbitrary spacing, a
gradient because gradients are easy, and a page that technically works while feeling
undifferentiated. Avoiding that is the job.

## Before writing any markup

Answer these first. Guessing produces the generic result.

1. **Who is looking at this, and what are they trying to finish?** A dense internal
   operations tool and a consumer marketing page want opposite things.
2. **What is the single most important thing on screen?** Everything else is secondary
   and should look like it. If you cannot name it, the design is not decided yet.
3. **What exists already?** Read the project's existing components, tokens, and CSS
   variables first. Consistency with the codebase beats a fresh opinion. If a design
   system or token file exists, use it - do not introduce a competing one.
4. **What is the real content?** Realistic text lengths, real numbers, real empty
   states. Placeholder "Lorem ipsum" hides every layout problem that real content
   exposes.

## Hierarchy and layout

- Establish a clear vertical rhythm. Spacing should come from a small, consistent set
  of values, not from arbitrary numbers chosen per element.
- One dominant focal point per view. Competing emphasis flattens hierarchy.
- Group related things tightly and separate unrelated things generously. Proximity is
  the cheapest way to express structure.
- Prefer asymmetry and deliberate alignment over centered-everything. Perfect
  centering is a default, not a decision.
- Constrain line length for prose. Full-width paragraphs are unreadable.

## Typography

- Pick a type scale with real ratios and apply it consistently. More than three or four
  sizes is usually a sign the hierarchy is muddled.
- Use a real font stack with fallbacks. System font stacks are legitimate and often
  the right call.
- Weight and size should carry hierarchy; colour should support it, not substitute.
- Set line-height deliberately: tighter for headings, looser for body copy.
- Use tabular numerals for columns of numbers so digits align.

## Colour

- Build on a small palette with one clear accent. More accent colours means no accent.
- Check contrast for text against its actual background, including hover and disabled
  states. Aim for WCAG AA (4.5:1 body text, 3:1 large text and UI boundaries).
- Support dark mode deliberately if the project has it, rather than by inverting
  colours at the last moment.
- Never encode meaning in colour alone. Pair colour with text, icon, or shape.

## Responsive behaviour

- Design mobile-first: establish what works at the narrowest width, then add capacity.
- Decide per component what reflows (stack, wrap, scroll) and what is genuinely
  unreachable on small screens.
- Test at least one narrow and one wide breakpoint; do not assume the middle works.
- Make touch targets large enough and keep interactive elements far enough apart.

## Interaction and states

Every interactive element needs more than a hover style. Define and implement:

- default, hover, focus-visible, active, disabled
- loading (skeleton or spinner appropriate to the size change)
- empty (explain what would appear here and how to create it)
- error (say what failed and what to do next)

Keyboard focus must always be visible. Removing an outline without a replacement is an
accessibility bug, not a style choice.

## Motion

- Animate to explain a change in state or spatial relationship. Animation that
  decorates is noise.
- Prefer compositor-friendly properties: `transform` and `opacity`.
- Keep durations short (roughly 150-400ms) and easing consistent across the product.
- Respect `prefers-reduced-motion`.

## Before you call it done

- [ ] Does the visual hierarchy match the priority you identified?
- [ ] Does it reuse the project's existing components and tokens?
- [ ] Does it work at a narrow width and a wide width?
- [ ] Are hover, focus, active, disabled, loading, empty, and error all defined?
- [ ] Is text contrast adequate on every background it appears on?
- [ ] Is focus visible, and is the whole flow keyboard operable?
- [ ] Does it use realistic content rather than placeholder text?
- [ ] Is motion reduced for users who ask for it?

## Anti-patterns

- Default framework styling left untouched.
- A rainbow of accent colours with no hierarchy.
- `box-shadow` used to fake elevation on flat elements inconsistently.
- Icon-only controls with no accessible name.
- Placeholder text as a substitute for labels.
- Animations on first load that delay content.

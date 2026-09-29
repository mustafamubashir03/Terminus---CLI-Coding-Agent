---
name: web-design-guidelines
description: >
  Use when REVIEWING or AUDITING an existing user interface against web interface
  standards: accessibility, keyboard operation, focus handling, form behaviour, error
  messaging, animation and reduced-motion, typography and localisation, image handling,
  performance signals, or interface consistency. Triggers on "review my UI", "audit
  this page", "check accessibility", "is this screen usable", "review the UX", "check
  contrast", "keyboard accessible", or a request for critique of a frontend.
  This skill INSPECTS AND REPORTS. It does not redesign. For creating or improving a
  UI use frontend-design; for React performance use react-best-practices. Do NOT use
  it to build new interfaces.
negative_triggers:
  - build
  - implement
  - create a
  - redesign
  - style
  - add a page
version: 1.0.0
license: MIT
author: Terminus (adapted from vercel-labs/agent-skills, Web Design Guidelines)
repository: vercel-labs/agent-skills
tags: [accessibility, a11y, review, audit, ux, forms, keyboard, frontend]
---

# Web Interface Guidelines Review

An audit pass over an existing interface. The output is a list of concrete, located,
specific findings - not a redesign.

Check the current guidelines before relying on this list. If the project's own
conventions or an updated published standard disagree with a rule here, the project
and the current source win; note the discrepancy rather than enforcing a stale rule.

## Accessibility

- Every interactive element has an accessible name. Icon-only controls need
  `aria-label` or visually-hidden text.
- Semantic elements first: `button` for actions, `a` for navigation, real headings in
  order, `label` bound to every form control.
- Text contrast meets WCAG AA (4.5:1 body, 3:1 large text and UI boundaries). Check
  the actual rendered colours, including hover, disabled, and placeholder.
- Colour is never the only carrier of meaning.
- Form errors are associated with their field (`aria-describedby`), announced, and
  stated in text - not only shown in red.
- Decorative images have empty `alt`; meaningful images describe their content.
- Dialogs trap focus, close on Escape, and return focus to their trigger.

## Keyboard

- Every action is reachable by keyboard alone.
- `focus-visible` styling exists and is clearly visible - never removed without a
  replacement.
- Logical tab order matches visual order. A positive `tabindex` above 0 is almost
  always a bug.
- No keyboard traps; `Escape` closes overlays.
- Custom controls expose correct roles and state (`aria-expanded`, `aria-selected`).

## Focus states

- Visible focus indicator on every focusable element, with adequate contrast.
- `:focus-visible` rather than `:focus`, so mouse users do not see rings but keyboard
  users always do.
- Focus is not removed after an action; move it somewhere sensible.

## Forms

- Correct `type` and `autocomplete` attributes on every field.
- Labels are persistent, not placeholders that vanish on input.
- Inline validation on blur or submit, with a clear submit-time summary when there are
  several errors.
- Error text says what to do, not just that something is wrong.
- Required fields marked programmatically as well as visually.
- The submit button shows progress and prevents double submission.

## Interaction and state

- Every data view defines loading, empty, error, and partial states.
- Empty states explain what would be here and how to create it.
- Errors are specific and actionable, and survive a page refresh where relevant.
- Destructive actions are distinguishable and, where costly, confirm.

## Animation

- Honours `prefers-reduced-motion`; disable or simplify non-essential motion.
- Animations use compositor-friendly properties (`transform`, `opacity`).
- Nothing animates on first load that delays or distracts from content.
- Durations are short and consistent; no bounce or elastic easing on functional UI.

## Typography and text

- Sensible line length; body measure around 60-75 characters.
- Real typographic characters where applicable: curly quotes, real ellipsis, proper
  dashes.
- Tabular numerals in numeric columns so digits align.
- Respect user text size; do not fix heights in `px` on text containers.
- Truncation degrades gracefully and is not the only way to see the value.

## Images and media

- Explicit dimensions or `aspect-ratio` to avoid layout shift.
- Below-the-fold images lazy-loaded; the largest above-the-fold image is not.
- Responsive `srcset`/`sizes` so phones do not download desktop assets.
- Video has captions; autoplay video is muted and respects reduced-motion.

## Performance signals

- Requests are parallelised, not serialised (see react-best-practices for detail).
- Expensive lists are virtualised; the DOM is not enormous for no reason.
- No layout thrashing in scroll or resize handlers.
- Fonts are preloaded, subset, and have fallbacks; no invisible text while loading.

## Navigation and state

- The URL reflects the view where it should be, and views are deep-linkable.
- Back and forward behave as the user expects.
- Destructive or long flows are not traps created by a missing back path.

## Internationalisation and locale

- Dates, times, and numbers formatted with `Intl`, not string concatenation.
- Layout accommodates longer translations and larger text.
- Text direction handled where the locale requires it.

## Touch and mobile

- Touch targets large enough and adequately spaced.
- `touch-action` set where custom gestures are used; no double-tap zoom surprises.
- Hover-only affordances have a touch equivalent.

## Dark mode and theming

- `color-scheme` declared so form controls and scrollbars match.
- `theme-color` meta present where the browser chrome should follow the theme.
- Both themes are designed, and images/icons exist for both.

## Output format

For each finding:

1. **What** - the issue, in one line.
2. **Where** - file and line, or a specific element.
3. **Why it matters** - the concrete user consequence (a keyboard user cannot reach
   this; this fails AA; this causes a 200ms layout shift).
4. **What to do** - the specific fix.

Order findings by user impact, not by file order. Say plainly if an area is fine -
that is useful information, not filler. If you could not check something, say which
part and why.

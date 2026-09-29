---
name: tailwindcss-development
description: >
  Use when writing, fixing, or refactoring Tailwind CSS utility classes, or when the
  task mentions tailwind: responsive grid and flex layouts, spacing and typography
  utilities, a dark mode variant, a component's class list, or a Tailwind v3 to v4
  upgrade. Triggers on "tailwind", "tailwind classes", "grid layout", "responsive
  breakpoints", "dark mode variant", "tailwind config", "deprecated tailwind utility",
  or a utility class that is not taking effect.
  Do NOT use for deciding what a page should look like or how it should be composed,
  for plain CSS or CSS modules with no Tailwind, for a server-rendered Blade or
  Django template with no Tailwind, or for an accessibility and semantics audit.
  This skill is about utility-class craft, not design decisions.
version: 1.0.0
license: MIT
author: Terminus (adapted from the Tailwind CSS development skill authored by Laravel,
  distributed in BostjanOb/MoneyCloud)
repository: BostjanOb/MoneyCloud
tags: [tailwind, css, utility-classes, grid, flex, responsive, dark-mode, v4]
negative_triggers:
  - make it look good
  - visual design
  - audit
  - review my ui
---

# Tailwind CSS Development

Utility classes are verbose in the source and short in review. The discipline is in
where they live and what they are allowed to contain.

## Follow the project's conventions first

Before writing a single class, read what is already there.

```bash
# Does the project use Tailwind, and which version?
grep -E '"tailwindcss"' package.json
# What does the config look like?
ls tailwind.config.* 2>/dev/null
# How is it configured in v4 (CSS-first)?
grep -rn "@import \"tailwindcss\"\|@theme" --include=*.css . 2>/dev/null | head
```

- Match the existing spacing, colour, and breakpoint scales. Introducing a
  one-off value because it looked right on that screen is how a design system
  erodes.
- Use the project's own tokens. If `bg-muted` exists, do not add a raw hex.
- Prefer `cn()`/class-merging helpers for conditional classes so a base and an
  override do not fight each other in the source order.

## Order classes for readability

The order does not change behaviour, but it changes whether a human can review it:

1. Layout (`flex`, `grid`, `position`)
2. Sizing (`w-`, `h-`, `min-`, `max-`)
3. Spacing (`p-`, `m-`, `gap-`)
4. Typography (`text-`, `font-`)
5. Colour (`bg-`, `text-`, `border-`)
6. Effects (`shadow-`, `opacity-`, `ring-`)
7. Responsive and state variants last

## Extract repetition deliberately

Repeated classes are a candidate for a component, but a repeated *class string* is
not automatically one. Extract when the repetition is a component, not when it is
coincidence.

- Prefer a real component over a `@apply` blob, which hides the class names from
  tooling and makes the CSS harder to read than the markup it replaced.
- When you do extract, keep the variants (hover, dark, responsive) at the call
  site so the caller can see the states.
- Combine utilities only when the combination is meaningless apart, such as a
  fixed button size reused everywhere.

## Layout that does not collapse

- **Grid**: `grid-cols-1 sm:grid-cols-2 lg:grid-cols-3` - mobile-first, so the
  base case is the narrow one.
- **Flex**: `min-w-0` on a flex child that holds text, otherwise it refuses to
  shrink and overflows. This is the most common Tailwind layout bug.
- Long unbroken strings need `break-words` or `truncate`.
- A flex parent with a scrolling child needs `min-h-0` on the child.

## Tailwind v4 notes

v4 is CSS-first: configuration lives in CSS via `@theme`, and there is no
`tailwind.config.js` by default.

- `@import "tailwindcss";` replaces the old three-layer directive stack.
- `@theme { --color-brand-500: ... }` defines design tokens usable as utilities.
- `corePlugins` is gone; configure with `@plugin`.
- Content scanning is automatic in v4 - a stale `content` array is usually why
  classes are not being generated.
- Check the current documentation for a utility before using one; several were
  renamed or removed.

## Dark mode

- `dark:` must be configured (`@custom-variant dark` in v4, or `darkMode` in the
  config) or the variant silently does nothing.
- Prefer semantic tokens (`bg-surface`, `text-muted`) over a `dark:` variant on
  every element. Then the theme is defined once.
- Test both themes. A border that works on light often disappears on dark.

## When a class is not working

1. Is the class name valid for the installed version? (v3 to v4 renames)
2. Is the content source picking up the file?
3. Is a later class in the source overriding it? Same specificity, so order in
   the generated CSS decides, not order in the class attribute.
4. Is a `dark:` or responsive variant active? It may be applying and overriding
   what you expect.
5. Is an arbitrary value malformed - `[#fff]` not `[#ffffff]` will not generate.

## Reporting

State the conventions you followed, the tokens you used, any new token you had to
add and why, and the breakpoints you verified. If you could not run the build to
confirm the classes generate, say so rather than implying the stylesheet was
verified.

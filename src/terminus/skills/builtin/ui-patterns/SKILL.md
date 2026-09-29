---
name: ui-patterns
description: >
  Use when choosing components and composing a page structure: picking a component
  library, deciding how to lay out a dashboard, settings page, data table, list, form,
  or empty screen, and implementing the states those screens need (loading, empty,
  error, partial, permission-denied). Triggers on "which component library", "build a
  dashboard layout", "settings page", "data table", "paginated table", "empty state",
  "loading skeleton", "form layout", "sidebar with topbar", "responsive grid", or
  "what should this page be made of".
  This skill is about composition and state, not visual taste. Do NOT use for deciding
  typography, colour, spacing aesthetics, or whether a design looks good - that is
  frontend-design. Do NOT use for auditing an existing UI for accessibility, which is
  web-design-guidelines, or for React performance, which is react-best-practices.
version: 1.0.0
license: MIT
author: Terminus (adapted from whawkinsiv/solo-founder-skills)
repository: whawkinsiv/solo-founder-skills
tags: [ui, components, layout, dashboard, table, forms, states, responsive, library]
negative_triggers:
  - polished
  - make it look good
  - look ugly
  - polish the ui
  - visual design
  - aesthetic
---

# UI Patterns

Pick the right parts and assemble them correctly. The look comes from
`frontend-design`; this is about *what the page is made of* and what it does in
every state.

## Before choosing anything

1. **Read what the project already uses.** Existing components, a design system, a
   `package.json`. Consistency with the codebase beats a better library.
2. **Check whether a dependency is even needed.** A select, a dialog, and a
   tooltip are a day of work before they are right, and a bad dependency is a
   permanent cost.
3. **Decide the states up front.** Every data view has loading, empty, error, and
   partial. Designing only the populated case means the states get invented later,
   in a hurry, under pressure.

## Composition: a page is a region layout

Most application screens are the same skeleton with different contents:

```
+------------------------------------------------------+
| topbar: breadcrumb, search, primary action, account |
+----------+-------------------------------------------+
| sidebar  |  page header: title, description, actions  |
| (nav)    +-------------------------------------------+
|          |  content: filters, then table/grid/form   |
|          |                                           |
|          |  pagination or infinite scroll            |
+----------+-------------------------------------------+
```

- **Dashboard**: a short row of summary metrics, then the one chart or table that
  matters. A wall of equal-weight cards communicates nothing - decide the primary
  question and answer it.
- **Settings**: group by mental model, not by database table. One column on
  desktop, stacked on mobile. Destructive settings separated at the bottom.
- **List/table**: a toolbar (search, filters, view switch) that stays put, then
  rows, then pagination. Decide sort order and make it explicit.
- **Form**: one column, labels above, help text under the field, errors tied to the
  field. Group long forms by section with headings.
- **Detail**: identity and status first, then the attributes, then actions, then
  history.

## States every data view needs

| State | What it must do |
|---|---|
| Loading | Match the final layout's dimensions so nothing shifts |
| Empty (never had any) | Say what would be here and how to create the first one |
| Empty (filtered to nothing) | Say the filter excluded everything, and offer to clear it |
| Error | Say what failed, whether it is retryable, and what to do next |
| Partial | Show what loaded; mark clearly what did not |
| Forbidden | Distinct from error: not a bug, and not retryable the same way |

The two empty states are different and conflating them is a common and visible
mistake. "No results for this filter" and "you have not created anything yet"
call for opposite actions.

## Data tables

- Keep the header visible while scrolling if the body scrolls independently.
- Right-align and tabular-align numbers; left-align text.
- Keep column count low. Past about seven, the table wants to be a list.
- Make the whole row clickable *and* keep a real focusable control inside it, so
  the row target does not break keyboard access.
- Sort state belongs in the URL if the view is linkable.
- Show a per-row action only where it applies; a row of disabled buttons is noise.
- Bulk selection needs a count and a clear way to clear it.

## Forms

- `autocomplete` and the right `type` on every field; it is a large accessibility
  and conversion win for near-zero cost.
- Validate on blur or submit, not on every keystroke. Error on a field the user
  has not finished is hostile.
- Put a summary of errors at the top on submit, each linking to its field.
- Say how to fix it: "Password must be at least 12 characters", not "invalid
  password".
- Disable submit while in flight, and keep the user's input if the request fails.
- Never lose a partially completed form on an error.

## Responsive

- Decide per component what reflows - stacks, wraps, or scrolls - and what is
  genuinely unavailable on a narrow screen.
- Tables are the hard case. Options in order of preference: hide low-priority
  columns; collapse to a stacked card; allow horizontal scroll on the table only.
  Do not shrink type until it fits.
- Sidebars become a drawer, not a squeezed column.
- Test one narrow (~375px) and one wide (~1440px) viewport, not just the middle.

## Dark mode

- Declared as a deliberate second theme, not an inversion applied at the end.
- If the project uses semantic tokens (`--surface`, `--text-muted`), use them and
  the two themes come almost free. If it uses raw hex, dark mode is a real project.
- Set `color-scheme` so native controls and scrollbars match.
- Shadows do not read on dark backgrounds; use surface elevation instead.

## Composition checklist

- [ ] Reused existing components rather than adding new ones
- [ ] Loading, empty, error, partial and forbidden all defined
- [ ] Empty-filtered distinguished from empty-never
- [ ] One primary action per region, clearly the primary
- [ ] Table sorted explicitly, numbers aligned
- [ ] Form fields typed and autocomplete set, errors actionable
- [ ] Verified at one narrow and one wide viewport

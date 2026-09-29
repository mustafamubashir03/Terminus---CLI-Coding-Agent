---
name: figma-implement-design
description: >
  Use when turning an EXISTING Figma design into working code: a Figma file, frame,
  node, or design link has been provided and the task is to implement it faithfully in
  the existing project. Triggers on "implement this Figma design", "build this from the
  Figma file", "match this mockup", a Figma node URL in the request, or an exported
  design the user wants turned into components.
  This is design-to-code ONLY. It does not create, edit, write to, or generate designs
  in Figma - that is a different workflow and a different tool. Do NOT use it when the
  user has no design and is asking you to design something. If Figma tooling is not
  available, report that instead of inventing design data.
version: 1.0.0
license: Apache-2.0
author: Terminus (adapted from openai/skills)
repository: openai/skills
tags: [figma, design-to-code, components, tokens, ui, implementation]
---

# Figma Design to Code

Implement an existing design faithfully, inside the conventions of the codebase it
lands in. This skill reads a design and writes code. It does not write to Figma.

## Prerequisite: design access

You need real design data. Check what is actually available before planning:

- a Figma MCP server or Figma API tool in this session
- a Figma file URL, node/frame ID, or exported assets
- images or annotated screenshots of the design

**If none of these exist, stop and say so.** Do not reconstruct a design from its
name, and do not proceed with a generic layout "in the spirit of" the design. Report
exactly what is missing - a Figma link, an export, or MCP access - and offer to work
from screenshots or a written spec instead.

## The pipeline

```
Figma design
  -> design context (layout, type, colour, spacing, states)
  -> visual reference (screenshot to compare against)
  -> tokens (the design's values mapped onto the project's existing system)
  -> project conventions (existing components win)
  -> implementation
  -> visual verification against the reference
```

Each step has a failure mode worth naming.

## 1. Extract design context

For the frames in scope, collect:

- **Structure**: frames, auto-layout direction, sizing mode (fill/hug/fixed), gaps
- **Typography**: family, size, weight, line height, letter spacing, and the actual
  text content
- **Colour**: fills, strokes, and opacity. Note whether a colour is a token or a
  one-off
- **Spacing and radius**: padding, item spacing, corner radius, border width
- **States**: variants, hover/active/disabled, and every frame the design includes
- **Assets**: icons, images, illustrations - export at the size actually used

A 1440px desktop frame is not the whole design. Find and implement the mobile frame
too if the design has one; shipping desktop-only from a design that has mobile is a
common and visible miss.

## 2. Capture a visual reference

Screenshot the design before writing code. This is the thing you will compare against
at the end, and without it "matches the design" is unfalsifiable.

## 3. Map values onto project tokens

This is where a faithful implementation usually goes wrong.

- **Read the project's existing tokens first** - CSS variables, Tailwind config, theme
  files, a design-system package.
- The design's values are the source of truth for *what it looks like*. The project's
  tokens are the source of truth for *how it is expressed*.
- If the design uses `#1A73E8` and the project has a primary token that resolves to the
  same colour, use the token. If the colour is genuinely new, add a token rather than
  a hard-coded hex scattered through components.
- Reuse existing components before creating new ones. Most designs are assembled from
  the same primitives the project already has.
- Never overwrite the repository's architecture, naming, or component patterns to match
  the design's internal structure. The design is a target, not a template.

## 4. Implement

- Build the component tree to match the design's structure, not your preferred pattern.
- Extract a component when the design repeats a structure; do not extract speculatively.
- Handle the real text from the design, including long strings - a design that fits
  with short labels must still work with long ones.
- Implement every state the design defines. A design that shows only the default is a
  gap to fill by convention, not an excuse to ship only the default.
- Prefer the project's existing patterns for data fetching, state, and styling.
- Make it responsive from the frames available, using the design's own breakpoints.

## 5. Verify visually

Compare the implementation against the reference screenshot:

- spacing, alignment, and overall proportions
- type scale and line heights
- colour values
- behaviour at the design's breakpoints
- every state the design specified

Fix what differs, then re-compare. Report which frames you verified and which you
could not.

## Reporting

State: which frames/nodes you implemented, which project components you reused or
added, which tokens you introduced, which states and breakpoints you covered, and what
you verified visually. If you could not access part of the design, say which part and
why rather than implying full coverage.

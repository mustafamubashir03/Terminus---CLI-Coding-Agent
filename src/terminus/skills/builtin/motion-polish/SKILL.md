---
name: motion-polish
description: >
  Use when adding or improving animation in a web interface: page and route transitions,
  micro-interactions, hover and press feedback, loading and skeleton animations, scroll
  and reveal effects, or when something already animated feels janky, laggy, or static.
  Triggers on "add animations", "make it feel smoother", "the transitions are janky",
  "add some life to this", "make it feel like Linear or Stripe", "polish the motion",
  "stagger the list", or a named effect such as a shared-element transition.
  Do NOT use for deciding what a screen should look like, for choosing colours, type,
  or layout, for accessibility and semantics review, or for React rendering and bundle
  performance. Animation is a layer on top of a finished design, not a substitute.
version: 1.0.0
license: MIT
author: Terminus (adapted from whawkinsiv/solo-founder-skills)
repository: whawkinsiv/solo-founder-skills
tags: [animation, motion, transitions, micro-interactions, css, reduced-motion, polish]
negative_triggers:
  - polished
  - make it look better
  - make it look good
  - choose a color
  - color scheme
  - typography
  - audit
  - review my ui
---

# Motion Polish

The test for motion is whether it explains something. Movement that decorates is
noise the user has to look past; movement that shows where a thing came from, or
that something is now ready, is felt rather than seen.

## The restraint rule

Most interfaces need less motion than you want to add. The failure mode is a page
where everything slides, fades, and bounces, and the user can no longer tell what
changed.

- One idea per view. If everything moves, nothing is emphasised.
- Never animate on first load in a way that delays or blocks content.
- If a user turns off motion, the interface must still be complete and usable.

## Choose the right property

Animation cost follows the property, not the effect.

| Property | Cost | Use for |
|---|---|---|
| `transform`, `opacity` | Compositor only - cheap | Almost everything |
| `width`, `height`, `top`, `left` | Layout on every frame - expensive | Rarely; use transform |
| `filter`, `box-shadow` | Paint - moderate | Occasional |

Animating layout properties at 60fps is the usual reason an interface feels heavy.
`height: auto` cannot be transitioned; animate `transform: scaleY()` or use
`grid-template-rows: 0fr -> 1fr`, or measure and set a fixed height.

## Timing

- **Micro-feedback** (hover, press, toggle): 100-200ms.
- **Element enter/exit**: 200-300ms.
- **Page or route transition**: 300-500ms.
- Anything slower than ~500ms is a delay, and the user is waiting.

- **Easing.** Entering elements decelerate (`ease-out`); leaving elements
  accelerate (`ease-in`); elements moving within view use a symmetric ease. A
  linear ease reads as mechanical.
- **Distance.** Small elements travel 4-16px. Large panels maybe 24-40px. Long
  slides feel slow even at the same duration.
- **Stagger** list items by 20-40ms and cap the total: a 50-item list staggered
  40ms each takes two seconds to settle, so stagger the first handful only.

## State transitions that need motion

Motion earns its place when the change is otherwise invisible:

- A modal or drawer should animate from the trigger, so its origin is obvious.
- An expanding panel should grow from where the click happened.
- A list item entering or leaving should not cause the rest of the list to jump.
- A value that changes should briefly indicate the change.

Equally: a hover state needs 150ms. A focus ring needs no animation at all.

## Loading and skeletons

- Prefer a skeleton that matches the real layout's dimensions, so nothing shifts
  when data arrives.
- If the wait is under ~300ms, show nothing. A spinner for a fast request is a
  flash of noise.
- Above ~1s, show progress structure. Above ~10s, say what is happening and offer
  a way to continue elsewhere.
- A spinner drawn as a border-spinner on a rounded square jitters. A rotating arc
  with a fixed size and a consistent start angle does not.

## Scroll and reveal

- Reveal on scroll: intersect once, then unobserve. Re-animating on every pass
  is noise.
- Never animate anything the user must read before it appears.
- Prefer `IntersectionObserver` to scroll handlers; scroll handlers that write
  layout cause exactly the jank they are meant to decorate.

## Reduced motion is required

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
```

Note what this preserves: opacity changes and colour changes still communicate
state, and that is enough. The information must never be carried *only* by
movement.

## Diagnosing jank

1. Record a performance profile in the browser and look for long tasks.
2. Check the animated properties - anything on the layout list is suspect.
3. Check for a `transition` on a property also being changed by JavaScript every
   frame; that re-triggers layout repeatedly.
4. Check `will-change`. It helps before an interaction and *hurts* memory if left
   on permanently; scope it to the interaction.
5. Check for several simultaneous heavy animations; a list of 50 independently
   animated items will not hold 60fps.

## Before you call it done

- Does every animation explain a state change or a spatial relationship?
- Is anything on the layout-property list animating?
- Is the total stagger under half a second?
- Does `prefers-reduced-motion` leave the interface complete?
- Did you profile it, or are you guessing that it is smooth?

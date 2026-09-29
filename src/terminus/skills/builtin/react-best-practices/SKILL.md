---
name: react-best-practices
description: >
  Use when writing, reviewing, or optimising React and Next.js code: creating or
  refactoring components, hooks, or pages; implementing data fetching; fixing slow
  re-renders; reducing bundle size or time to interactive; or auditing a React app for
  performance problems. Triggers on "this component re-renders too much", "reduce bundle
  size", "improve load time", "server component vs client component", "waterfall",
  "React performance", or any non-trivial React or Next.js implementation.
  Do NOT use for non-React frontends (Vue, Svelte, Angular, plain HTML), for backend or
  Node API work, or for CSS and visual design questions - that is frontend-design.
  Do NOT load this into unrelated Python, Go, Rust, or infrastructure tasks.
negative_triggers:
  - tailwind
  - tailwind class
  - dark mode variant
version: 1.0.0
license: MIT
author: Terminus (adapted from vercel-labs/agent-skills, React Best Practices)
repository: vercel-labs/agent-skills
tags: [react, nextjs, performance, frontend, hooks, rendering, bundle]
---

# React and Next.js Performance

Optimise for the thing users feel: time to first useful paint, and smoothness after.
Work through the categories in impact order - the top two are most of the win.

## 1. Eliminate waterfalls (critical)

The most common performance defect in React apps. Independent requests are issued
serially because each waits for the previous one to resolve.

```jsx
// Sequential: three round trips, one after another.
const user = await fetchUser(id);
const prefs = await fetchPrefs(id);
const orders = await fetchOrders(id);
```

```jsx
// Parallel: one round trip's worth of latency.
const [user, prefs, orders] = await Promise.all([
  fetchUser(id), fetchPrefs(id), fetchOrders(id),
]);
```

In Next.js App Router, fetch in parallel components rather than awaiting inside one
component. Move a `await` down into a child component so siblings start together.

Also avoid: awaiting a fetch before rendering static content; sequential
`await` in a loop where `Promise.all` would do; awaiting a promise to compute
something the first paint does not need.

## 2. Bundle size (critical)

- Import only what you use. A barrel file that re-exports everything can pull in a
  whole library for one function; import from the specific path.
- Check whether a dependency can be replaced with a few lines of native code.
  `date-fns` for one `format` call, or a moment-like library, is usually not worth it.
- Dynamically import heavy, below-the-fold, or conditionally-needed components:
  `const Chart = dynamic(() => import('./Chart'), { ssr: false })`.
- Never ship a full icon library for the four icons you use.
- Watch the client boundary: `'use client'` opts a whole subtree out of server
  rendering. Push it as deep as possible.

## 3. Server-side performance (high)

- Cache what is stable. In Next.js use the framework's caching primitives deliberately
  rather than re-fetching per request by default.
- Stream slow parts with Suspense so the shell paints immediately.
- Keep server-only secrets and heavy computation off the client.
- Prefer streaming HTML to fetching a page's data client-side and rendering a spinner.

## 4. Client-side data fetching (medium-high)

- Fetch on the server when the data is needed for first paint.
- Avoid client-side waterfalls; see section 1.
- Deduplicate identical concurrent requests.
- Do not poll when a websocket, SSE, or revalidation would do.
- Stop polling when a component unmounts.

## 5. Re-render optimisation (medium)

- `React.memo` is a last resort, not a default. It only helps when a component
  actually re-renders with unchanged props *and* is expensive, and it adds comparison
  cost.
- Fix the cause first: unstable props and new inline objects/functions are what cause
  the re-renders memo is used to paper over.
- Stabilise callbacks with `useCallback` only when the identity matters (a memoised
  child, or an effect dependency) - otherwise it is noise.
- Split state: two independent pieces of state in one component re-render everything
  together.
- Derive, do not store: a value computable during render does not belong in state.
- Use functional updates (`setX(prev => ...)`) so you do not depend on a stale closure.
- Keep effects honest. If an effect only computes something from props and state, render
  it instead. An effect that synchronises with a prop is usually a missing key or a
  derived value.
- Do not wrap context values in objects created inline; memoise the value.

## 6. Rendering performance (medium)

- Virtualise long lists (hundreds of rows). Rendering 10,000 nodes is the cost, not the
  data.
- Give lists stable `key`s. An array index as key makes React re-render everything on
  reorder, and silently corrupts state when items move.
- Avoid layout thrashing: batch reads before writes, and do not interleave
  `getBoundingClientRect`/`offsetWidth` reads with style writes in a loop.
- `content-visibility: auto` defers rendering of off-screen sections cheaply.

## 7. JavaScript cost (low-medium)

- Move per-frame work out of render; use a ref for hot values.
- Prefer CSS transform and opacity for animation - they stay on the compositor.
- Debounce or throttle expensive handlers (resize, scroll, input).
- Look for accidental O(n^2): `.includes` inside `.map`, `.find` inside a loop, deep
  clones inside render.

## 8. Images and assets (high, cheap)

- Set explicit `width` and `height` (or `aspect-ratio`) so the browser reserves space
  and does not shift layout.
- Use the framework's image component with responsive `sizes`.
- Lazy-load below-the-fold images; never lazy-load the LCP image - it should be
  eager and preloaded.

## Diagnosing before optimising

Measure first. A change that does not improve a measured metric is not an optimisation,
it is churn.

- React DevTools Profiler: which components render, how often, and why
- Chrome DevTools Performance: long tasks, layout shifts, paint times
- Network panel: request count, size, and where the time goes
- `next build` output: route sizes, first-load JS, and what dominated it

## Reporting

State the measurement you improved, before and after, and the change that caused it.
"Optimised performance" with no numbers is not a result.

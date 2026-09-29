---
name: skill-creator
description: >
  Use when creating a new skill, improving or fixing an existing one, writing a skill
  description or trigger, splitting one skill into several, or judging whether a skill
  is actually worth having. Triggers on "create a skill", "write a skill", "add a
  skill", "my skill does not get selected", "improve this skill", "why is this skill
  never triggered", or "SKILL.md".
  A skill is not finished documentation: this skill treats it as a testable artefact
  that must be evaluated against realistic prompts and iterated on. Do NOT use it to
  write ordinary project documentation or a README.
version: 1.0.0
license: Apache-2.0
author: Terminus (adapted from anthropics/skills)
repository: anthropics/skills
tags: [skills, skill-authoring, evaluation, prompts, triggers, meta]
---

# Skill Creator

A skill only earns its place if it changes behaviour on the prompts it claims. Treat
authoring as measurement, not writing.

## The contract

```
<skill-name>/
  SKILL.md        required - frontmatter + instructions
  references/     optional - supporting documentation
  scripts/        optional - helpers (never run automatically)
  assets/         optional - templates, fixtures
```

```markdown
---
name: skill-name
description: >
  What it does, what triggers it, and when NOT to use it.
version: 1.0.0
license: Apache-2.0
tags: [a, b, c]
---

# Title

Instructions the agent follows.
```

Only `name` and `description` are required. Do not invent metadata that carries no
information.

## The description is the trigger, not documentation

This is the highest-leverage part of a skill and the most commonly botched. The
description is what the matcher reads to decide relevance. It must state three things:

1. **What it does.**
2. **What kind of request should trigger it** - the concrete situations, phrased the
   way a user would phrase the request.
3. **What it should NOT be used for** - the near-misses, stated explicitly.

```yaml
# Too weak. Cannot be matched reliably.
description: Frontend design skill.

# Usable. Names situations, and the boundary.
description: >
  Use when building or substantially redesigning a user interface: pages, dashboards,
  application screens, where visual hierarchy, typography, spacing and interaction
  quality are the substance of the task. Triggers on "design a page", "restyle",
  "polish the UI". Do NOT use for backend or API work, and do NOT use merely because a
  task mentions React.
```

Write triggers in the vocabulary of the *request*, not of the domain. Users say "make
the dashboard look less ugly", not "apply visual hierarchy principles".

## Writing the body

- Lead with the decision or procedure, not with background. The agent already knows
  what React is.
- Give checkable steps, not encouragement. "Check the console for errors" beats "be
  thorough".
- Include the anti-patterns. What to avoid is often more actionable than what to do.
- State prerequisites explicitly, and what to do when they are missing.
- Prefer one skill per coherent job. If a skill needs "and also", consider splitting.
- Keep it inside the budget (8000 characters). A skill that must be truncated teaches
  the model to guess.

## Evaluation: the part everyone skips

Write evaluation prompts before you write the skill, or at minimum before you call it
done. Include **negative** cases - a skill that is selected for everything is worse
than no skill, because it spends context and dilutes attention.

```markdown
Positive:
- "Build a polished SaaS dashboard in React"      -> react-best-practices, frontend-design
- "This table re-renders on every keystroke"       -> react-best-practices
- "It works locally but fails in CI"               -> systematic-debugging

Negative (must select nothing):
- "Fix MongoDB connection timeout in Python"      -> no frontend skills
- "Add a REST endpoint for invoices"               -> no frontend skills
- "Bump the dependency versions"                   -> no skills
```

For each prompt, record which skills were selected and why. Then:

1. **Does the right skill fire?** Missing a trigger is a description problem.
2. **Does the wrong skill fire?** A false positive is a description boundary problem.
3. **Do the instructions change the output?** Compare baseline (no skill) against
   skill-assisted on the same prompt. If the outputs are indistinguishable, the skill
   is decoration and should be deleted or rewritten.

## Iterate

- Version the skill and note what changed and why.
- When a skill is never selected, the fix is almost always the description, not the
  body. The body is never read if selection fails.
- When a skill is always selected, it is too broad - narrow the trigger or split it.
- Delete skills that no longer earn their place. A curated library with 9 sharp skills
  beats 40 vague ones.

## Test it in Terminus

Terminus ships a built-in trigger evaluation. Use the same idea when you add your own:

- `/skills` lists what is installed and from where.
- `/skill <name>` shows the description, source, version, references, and any
  executable helpers.
- Explicitly requesting a skill always wins over automatic selection, so you can
  compare the two directly.

## Provenance

Record where a skill came from and under what licence. If you adapted one, say so in
`repository`. If it ships an executable helper, that is called out explicitly and is
never run automatically - a user running it is a normal, approved tool call.

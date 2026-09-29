"""Deterministic, testable skill selection.

Design rule: the matcher must be explainable and reproducible. A skill that gets
loaded costs context, and a skill loaded for the wrong reason actively degrades
output, so precision matters more than recall. Every selection is therefore a
scored decision with a recorded reason rather than a black box, and a user can
always override the outcome by naming a skill explicitly.

Why not embeddings
------------------
Choosing between a dozen curated skills and a user's own additions does not
need semantic similarity. A well-written ``description`` already states the
trigger, and lexical scoring over that description is inspectable, free, and
testable offline. Embeddings would add a model call and a dependency to make a
decision a human can read. If this matcher is ever shown to be insufficient, the
escalation path is an LLM selector layered *on top* of these signals, not a
replacement - the scores below remain the floor and the ceiling respectively.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Iterable

from terminus.skills.registry import (
    MAX_SKILLS_PER_TASK,
    MAX_SKILLS_TOTAL_CHARS,
    SkillRef,
)

# A candidate must clear this to be selected automatically. Tuned against the
# trigger evaluation suite in tests/test_skill_triggers.py, which asserts both
# that real skills are recalled and that unrelated prompts select nothing.
DEFAULT_THRESHOLD = 2.0

# Words that carry no signal about which skill applies. Removing them stops
# "the" in a long sentence from matching a skill that happens to be called
# "the-..." and keeps scoring about the words the author chose.
STOPWORDS = frozenset("""
a an and are as at be by can could do does for from get give go has have how i
if in into is it its just like make me my need not of on or our should so some
that the their them then there these they this to use used using want was we
were what when where which will with would you your
""".split())


def _tokens(text: str) -> list[str]:
    """Lowercase word tokens, lightly stemmed so inflected forms still match.

    Skill descriptions are written as prose and requests are written as intent,
    so the two routinely disagree on inflection: a description says "redesigning"
    where the request says "redesign", or "audits" where the request says "audit".
    A matcher that cannot see that those are the same word will miss obvious
    triggers. Each token is therefore emitted with its own form plus a crude
    stem, and matching happens on the union.
    """
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    out: list[str] = []
    for word in words:
        if word in STOPWORDS or len(word) < 3:
            continue
        out.append(word)
        if word.endswith("ies") and len(word) > 5:
            out.append(word[:-3] + "y")
        elif word.endswith("ing") and len(word) > 6:
            out.append(word[:-3])
        elif word.endswith("ed") and len(word) > 5:
            out.append(word[:-2])
        elif (
            word.endswith("s")
            and len(word) > 4
            # Only the plain "-s" plural. Words ending "es" are mostly singular
            # nouns that merely end in s - "kubernetes" must not become
            # "kubernete" - and the "ss"/"us"/"is" endings are never plurals.
            and not word.endswith(("ss", "us", "is", "es"))
        ):
            out.append(word[:-1])
    return out


def _phrases(text: str) -> list[str]:
    """Multi-word signals, which matter more than isolated tokens.

    "accessibility" and "contrast ratio" both appearing in a description is a
    stronger hint than the word "design" appearing once, and multi-word trigger
    phrases in a description are usually the author telling us the boundary.
    """
    return re.findall(r"[a-z][a-z ]{4,40}", (text or "").lower())


@dataclass
class SkillMatch:
    """One scored candidate and the reason it scored what it did."""

    name: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    ref: SkillRef | None = None
    selected: bool = False
    explicit: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 2),
            "reasons": list(self.reasons),
            "selected": self.selected,
            "explicit": self.explicit,
            "source": self.ref.source if self.ref else "",
        }


def _negative_triggers(ref: SkillRef) -> list[str]:
    """Phrases that mean "this request is the other skill's job".

    A description can say in prose that a skill should not fire on a certain kind
    of request, but prose negation is not something a lexical matcher can read:
    "Do NOT use for building interfaces" contains the very words "building" and
    "interfaces" that would otherwise score. Declaring the boundary as data makes
    the non-trigger enforceable, and keeps the intent in the skill where the author
    can see it. Entirely optional - a skill without it simply has no non-triggers.
    """
    raw = (ref.meta or {}).get("negative_triggers") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(item).strip().lower() for item in raw if str(item).strip()]


# Domain vocabulary, not identifiers.
#
# A skill name containing "design" or "polish" or "test" does not become the
# obvious answer to a request containing those words - they are the shared
# language of software work, so a request like "build a polished dashboard"
# says nothing about whether a skill named "motion-polish" is wanted. Only a name
# token that is actually distinctive earns the name bonus, otherwise every
# generic adjective silently selects whatever skill happens to contain it.
GENERIC_NAME_TOKENS = frozenset("""
app build building code coding create creating database design dev develop
generate make motion polish software test testing tool tool ui ux web
""".split())


def _score_one(
    request_text: str,
    ref: SkillRef,
    signals: dict[str, Any],
    name_token_counts: dict[str, int] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> SkillMatch:
    """Score a single skill. Additive and readable on purpose.

    ``name_token_counts`` records how many installed skills carry each name
    token, so a word that is not a discriminator can be recognised as one.
    """
    match = SkillMatch(name=ref.name, ref=ref)
    request_tokens = set(_tokens(request_text))
    if not request_tokens:
        return match

    # Description is the trigger contract, so it is weighted highest.
    meta = ref.meta or {}
    description = f"{ref.description} {meta.get('when_to_use', '')}"
    description_tokens = set(_tokens(description))
    overlap = request_tokens & description_tokens
    if overlap:
        match.score += min(4.0, 0.8 * len(overlap))
        match.reasons.append(
            f"request mentions {', '.join(sorted(overlap)[:6])} from its description"
        )

    # Name matches are a strong signal - someone asking about "react" almost
    # certainly wants the react skill - but only for words that actually
    # discriminate. "design" appears in the name of more than one installed skill,
    # so matching it tells us nothing about *which* design skill is meant, and
    # awarding it as a name hit is how a generic word started selecting skills.
    counts = name_token_counts or {}
    name_hits = {
        token
        for token in request_tokens & set(_tokens(ref.name))
        if counts.get(token, 1) <= 1 and token not in GENERIC_NAME_TOKENS
    }
    if name_hits:
        match.score += 3.0 * len(name_hits)
        match.reasons.append(f"request names '{' '.join(sorted(name_hits))}'")

    # Explicit tags are the author pre-committing to a vocabulary.
    for tag in ref.tags:
        if _tokens(tag) and set(_tokens(tag)) & request_tokens:
            match.score += 1.0
            match.reasons.append(f"tag '{tag}' matches")
            break

    # Framework/language signals supplied by the caller (task type, detected
    # stack). These are facts about the work, not guesses.
    for key, weight in (("frameworks", 2.0), ("languages", 1.0), ("task_type", 0.5)):
        value = signals.get(key)
        if not value:
            continue
        wanted = {str(v).lower() for v in (value if isinstance(value, Iterable) and not isinstance(value, str) else [value])}
        haystack = f"{ref.name} {ref.description} {' '.join(ref.tags)}".lower()
        for item in sorted(wanted):
            if item and item in haystack:
                match.score += weight
                match.reasons.append(f"{key} '{item}' applies")
                break

    # A declared non-trigger is applied last, and only against weak evidence.
    #
    # A blunt veto is wrong: "redesign the page and audit it for accessibility"
    # names a non-trigger *and* the skill's own job, and vetoing there would
    # refuse the reviewer on a request that explicitly asked for a review. So a
    # non-trigger suppresses a marginal match, while strong positive evidence -
    # a name hit, or a clear pile-up of description overlap - survives and records
    # the tension instead of hiding it.
    for phrase in _negative_triggers(ref):
        if phrase not in (request_text or "").lower():
            continue
        if name_hits or match.score >= threshold * 1.5:
            match.reasons.append(
                f"request also contains its non-trigger '{phrase}', but the "
                "request clearly asks for this skill's own work - kept"
            )
        else:
            match.score = 0.0
            match.reasons.append(
                f"not selected: request matches its non-trigger '{phrase}'"
            )
        break

    return match


def match_skills(
    registry: Any,
    request: str = "",
    *,
    explicit: Iterable[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    limit: int = MAX_SKILLS_PER_TASK,
    include_all: bool = False,
    **signals: Any,
) -> list[SkillMatch]:
    """Rank skills for a request, explicit selections always included.

    With ``include_all``, every candidate is returned with its score and
    selection flag, so a caller can inspect why something was *not* chosen -
    which is what makes a wrong trigger debuggable rather than mysterious.

    ``explicit`` names are the user saying "use this skill". They are added
    regardless of score, because a user who asks for a skill by name knows
    something the matcher does not - and that is the whole point of allowing an
    explicit request to override automatic discovery.
    """
    refs: dict[str, SkillRef] = getattr(registry, "refs", {}) or {}
    wanted = {str(name).strip() for name in (explicit or []) if str(name).strip()}
    wanted_lower = {name.lower(): name for name in wanted}

    results: list[SkillMatch] = []
    name_token_counts: dict[str, int] = {}
    for ref in refs.values():
        for token in set(_tokens(ref.name)):
            name_token_counts[token] = name_token_counts.get(token, 0) + 1

    for ref in refs.values():
        canonical = wanted_lower.get(ref.name.lower())
        match = _score_one(request or "", ref, signals, name_token_counts, threshold)
        if canonical:
            match.explicit = True
            match.selected = True
            match.reasons.insert(0, "explicitly requested by the user")
        elif match.score >= threshold:
            match.selected = True
        results.append(match)

    # An explicitly requested skill that does not exist is reported rather than
    # silently dropped, so a typo surfaces instead of quietly changing behaviour.
    for lowered, original in wanted_lower.items():
        if lowered not in {ref.name.lower() for ref in refs.values()}:
            results.append(
                SkillMatch(
                    name=original,
                    score=0.0,
                    reasons=["explicitly requested, but no such skill is installed"],
                    selected=False,
                    explicit=True,
                )
            )

    results.sort(
        key=lambda m: (not m.explicit, -m.score, m.name),
    )
    if include_all:
        return results
    if limit is None:
        return [m for m in results if m.selected]
    selected = [m for m in results if m.selected]
    if len(selected) > limit:
        for extra in selected[limit:]:
            extra.selected = False
            extra.reasons.append(
                f"not loaded: beyond the {limit}-skill budget for one execution"
            )
        selected = selected[:limit]

    # Anything the caller asked for and did not get is still returned, marked
    # unselected and carrying the reason. Silently dropping an explicit request
    # or a budget overflow would make a skill look "not applicable" when the
    # truth is "not loaded", which is a very different thing to debug.
    not_loaded = [
        m
        for m in results
        if not m.selected
        and (m.explicit or any("budget" in r for r in m.reasons))
    ]
    return selected + not_loaded


def render_selection(
    matches: Iterable[SkillMatch],
    registry: Any,
    budget: int = MAX_SKILLS_TOTAL_CHARS,
) -> str:
    """The bounded instructions block injected for the selected skills.

    The whole block is capped at *budget*, not just each skill, and the budget is
    shared in proportion to body size. Selecting a second skill therefore has to
    fit inside what is left rather than adding its full length, which is what
    keeps prompt growth bounded as the library grows.
    """
    chosen = [m for m in matches if m.selected and m.ref is not None]
    if not chosen:
        return ""

    bodies = [(m, m.ref.truncated_body()) for m in chosen if m.ref is not None]
    # Headers, reasons and separators are overhead, not skill content, so they
    # are reserved out of the budget before the bodies are divided. Without this
    # the block lands slightly over the cap, which is how a "bounded" budget
    # quietly stops being one.
    overhead = sum(
        64 + len(m.name) + len(m.ref.source if m.ref else "") for m, _ in bodies
    )
    overhead += 32 * len(bodies) + 64
    body_budget = max(300, budget - overhead)

    total = sum(len(body) for _m, body in bodies)
    share = body_budget
    if total > body_budget and total > 0:
        share = max(300, int(body_budget * (body_budget / total)))

    parts: list[str] = ["## Skills in effect", ""]
    for match, body in bodies:
        ref = match.ref
        assert ref is not None
        reason = "; ".join(match.reasons[:3]) or "selected"
        if len(body) > share:
            body = ref.truncated_body(share)
        parts.append(
            f"### {ref.name}  (source: {ref.source} | {reason})\n\n{body}"
        )
    text = "\n\n".join(parts)
    if len(text) > budget:
        text = text[: budget - 40].rstrip() + "\n\n[skills block truncated]"
    return text


def detect_conflicts(matches: Iterable[SkillMatch]) -> list[str]:
    """Report skills whose instructions compete, without merging them blindly.

    Two skills touching the same concern is not a conflict - "implement a
    dashboard" legitimately wants both a design skill and a framework skill.
    A conflict is two skills claiming the same job, which would leave the model
    following contradictory advice. It is reported so precedence can be applied
    and visible, never resolved by concatenation.
    """
    chosen = [m for m in matches if m.selected and m.ref is not None]
    conflicts: list[str] = []
    for i, a in enumerate(chosen):
        for b in chosen[i + 1:]:
            assert a.ref is not None and b.ref is not None
            a_tokens = set(_tokens(f"{a.ref.name} {a.ref.description}"))
            b_tokens = set(_tokens(f"{b.ref.name} {b.ref.description}"))
            shared = a_tokens & b_tokens
            if len(shared) >= 3:
                conflicts.append(
                    f"{a.ref.name} and {b.ref.name} overlap on "
                    f"{', '.join(sorted(shared)[:4])}"
                )
    return conflicts

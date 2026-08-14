# Guardrails

Three layers, placed where the evidence for the decision actually exists.

| Layer | Where | Evidence it uses | Source |
|---|---|---|---|
| **Input** | Before retrieval | Rules over the utterance | [`guardrails/input.ts`](../src/core/guardrails/input.ts) |
| **Abstention** | After retrieval, after synthesis | Cross-encoder logits | [`guardrails/policy.ts`](../src/core/guardrails/policy.ts) |
| **Grounding** | After synthesis | The answer against its cited spans | [`guardrails/grounding.ts`](../src/core/guardrails/grounding.ts) |

The bias throughout is deliberate: **a confident wrong answer is worth less than
an honest "I do not have that."**

---

## Input

The design constraint is **precision, not recall**. This corpus is web text about
medicine, chemistry, law and history, so a keyword blocklist would refuse *"what
is the lethal dose of acetaminophen"* — a question whose answer is on a
poison-control page and which MS MARCO actually contains.

Every rule therefore matches an **intent pattern**: an instruction frame plus an
object that has no benign reading inside that frame.

```js
// weapons — the frame is what makes it a rule rather than a blocklist
/\b(how (do i|to|can i)|steps? to|guide to|instructions? for|teach me to)\b
 [^?]{0,60}\b(build|make|construct|assemble|synthesi[sz]e)\b
 [^?]{0,40}\b(bomb|explosive|ied|nerve agent|chemical weapon|…)\b/i
```

Seven categories: weapons, drug synthesis, self-harm, targeted harm, CSAM,
malware, credential theft. The false-positive rate is asserted against a list of
benign medical and chemistry questions in
[`input.test.ts`](../src/core/guardrails/input.test.ts).

### Evaluation order

```
empty            → EMPTY_QUERY      ABSTAINED   "the audio came through empty"
filler / mic noise → EMPTY_QUERY    ABSTAINED   um, uh, hm, "testing 1 2 3"
small talk       → NOT_A_QUESTION   ABSTAINED   answered without a lookup
personal state   → PERSONAL_CONTEXT ABSTAINED   ← see below
harmful intent   → UNSAFE_INPUT     REFUSED
prompt injection → strip and continue           ← not a refusal
  └─ nothing left → PROMPT_INJECTION REFUSED
over 500 chars   → truncate, flag, continue
```

**Injection is stripped, not refused.** A transcript is untrusted input that
reaches a generation stage, so an instruction hidden in speech has to be
neutralised — but a real question wrapped in an injection attempt is still a
real question. The verdict returns a cleaned `query`, and only an utterance that
is *nothing but* injection is refused.

### The personal-state rule

This one came out of testing, and it is the clearest case for why an input rule
is needed at all:

> *"what is my bank account balance"* retrieves a genuinely on-topic passage
> about trial balances, the cross-encoder scores it **positively**, and the
> pipeline confidently answers a question about accounting to someone asking
> about their own money.

No retrieval score can catch that, because the retrieval is not wrong — no
corpus can hold that answer. So the check has to happen before retrieval.

The rule is narrow on purpose. It fires only on interrogatives asking for the
**value** of a first-person-possessed private thing:

```
^(what|when|where|which|who|how much|how many|how long) …
  \b(my|our)\s+ (next|last|current|bank|email|…){0,2}
  (account|balance|password|salary|appointment|prescription|inbox|…)
```

`how do i …` deliberately does not match, so *"how do I change my password"* —
which this corpus does answer — passes straight through. Two tests pin exactly
that boundary.

---

## Abstention

Gated on cross-encoder logits because, unlike cosine, they do not saturate.
Cosine returns ~0.45 for an off-topic corpus; a logit goes strongly negative.

### The two gates

```js
// checkRetrieval — after retrieval, reads RetrievalSignals
if (reranked && isFinite(crossTop)) {
  if (crossTop < -7.0) → OUT_OF_CORPUS
  if (crossTop < -6.0) → WEAK_RETRIEVAL
  → answer
}
// reranker unavailable: fall back to bi-encoder signals
if (topDense < 0.34 && lexicalCoverage < 0.45) → OUT_OF_CORPUS
if (topDense < 0.42 && margin < 0.035)         → WEAK_RETRIEVAL

// checkAnswer — after synthesis, reads the answer sentence's own logit
if (answerLogit !== null) {
  if (answerLogit < 1.0) → NO_ANSWER_IN_CONTEXT
  → answer                                  // it decides alone
}
// reranker unavailable only:
if (coverage < 0.3 || confidence < 0.2) → NO_ANSWER_IN_CONTEXT
```

Two structural notes:

- `outOfCorpusCross` (−7.0) does not change the accept/reject call — anything
  under `weakCross` is already rejected. It only decides **which of two messages
  the user gets**, and it sits near the off-corpus median so a nonsense question
  is told it is off-corpus rather than merely unanswered.
- The lexical `coverage`/`confidence` gates apply **only** when the logit is
  null. Layering them on top of the logit gate cost recall in the sweep without
  catching anything the logit missed, so they became the fallback path rather
  than a second opinion.

### The separation that justifies the gates

Measured over three populations by `npm run bench:quality`:

| Population | passage logit (p10 / p50) | **answer-sentence logit (p10 / p50)** |
|---|---:|---:|
| answerable | 5.7 / 8.6 | **4.3 / 7.5** |
| MS MARCO "no answer present" | 1.2 / 5.8 | **−0.1 / 4.3** |
| off-corpus probes | −10.3 / −7.2 | **−11.0 / −7.7** |

The answer-sentence gate sits at **+1.0** — just above the "no answer present"
population's 10th percentile (−0.1) and well below the answerable population's
(4.3). That separation is what makes it worth gating on, and it is what took
off-corpus refusal from 11/12 to **12/12**.

The sentence-level signal is sharper than the passage-level one for a structural
reason: a passage can be entirely on-topic while no sentence in it actually
answers the question, and only a scorer that reads the *candidate answer* can
see that.

### Where the thresholds came from

`benchAbstention` in [`tools/bench.ts`](../src/tools/bench.ts) enumerates ~320
threshold combinations. Re-running retrieval for each would take half an hour, so
it runs retrieval **once per query**, caches `{signals, coverage, confidence,
answerLogit}`, and replays the gate functions over the cache — legitimate because
retrieval is deterministic and independent of the thresholds.

The shipped defaults **are** the sweep's optimum (precision 0.697, recall 0.968,
F1 0.811), not a nearby round number.

### What it catches, and what it does not

| Population | Answered | Wanted |
|---|---|---|
| answerable questions | 242 / 250 | all |
| **off-corpus probes** | **0 / 12** | none |
| MS MARCO "No Answer Present" | 105 / 125 | none |

The third row is a genuine limitation, reported rather than averaged away. Those
passages *are* topically relevant; the human annotator simply could not find an
answer in them, which is a property of the answer rather than of the retrieval.
The answer-sentence gate moved it from 119/125 to 105/125 — real, but not a
solution. Closing the rest needs a dedicated answerability model, not a
threshold.

---

## Grounding

Three checks, because they fail differently.

### 1. Lexical support — a ratio

Share of the answer's content words (non-stopword, >2 chars) present in the
cited spans. Threshold **0.72**. Near 1.0 for the extractive path by
construction; it exists for the rewrite path, where it catches a fluent sentence
that quietly invented a clause.

### 2. Numeric and entity containment — a hard veto

Every number in the answer must appear in the context. Not a score — a veto.

> A rewrite that turns *"about 105 calories"* into *"about 150 calories"* is
> still **96% lexically supported**. That is exactly the failure a support ratio
> cannot see.

`extractNumerics` normalises to the digit run (so `$1,200.00` and `1200.00`
compare equal); `extractEntities` is a capitalised-multiword proxy, reported but
not vetoed, because proper-noun detection by regex has too many false positives
to hang a rejection on.

### 3. Semantic support — cosine

Cosine between the answer embedding and the cited-context embedding, threshold
**0.45**. Catches negation and attribution flips that keep every word but
reverse the claim.

### Two entry points

```js
verifyExtractive(answer, citations)    // sync, ~0.03 ms — the pipeline default
  → semanticSupport: 1 (by construction: the answer is a substring of a cited span)
  → passed = lexicalSupport ≥ 0.72 && unsupportedNumerics.length === 0

verifyGrounding(answer, citations)     // async, ~4 ms — two encoder calls
  → passed = lexicalSupport ≥ 0.72 && no unsupported numerics
             && semanticSupport ≥ 0.45
```

Recording `semanticSupport: 1` for the extractive path is honest rather than
lazy: the answer *is* a cited substring, so recomputing the cosine would only
measure the encoder's self-similarity against itself.

The rewrite path applies a **looser** containment check (`lexicalSupport ≥ 0.6`,
zero unsupported numerics) because re-voicing legitimately changes wording — but
the numeric veto is identical, and a failed check discards the rewrite and keeps
the verified extractive answer.

---

## Test coverage

26 unit tests, all runnable with `npm test` (no index required):

| File | Tests | Pins |
|---|---:|---|
| [`input.test.ts`](../src/core/guardrails/input.test.ts) | 8 | Harmful intent refused · benign medical/chemistry questions **not** refused · injection stripped but question kept · injection-only refused · private-state declined · how-to questions not mistaken for private state · greetings and mic noise handled · ordinary questions untouched |
| [`chunking.test.ts`](../src/core/chunking/chunking.test.ts) | 8 | Spans round-trip exactly · uniqueness and strategy-bit union · pre-dedup proposal counting · single-strategy restriction · propositions finer than passages · sentence splitting over abbreviations/decimals/initials · clause offsets aligned · numeric extraction |
| [`harness.test.ts`](../src/core/harness/harness.test.ts) | 10 | Retry accounting · non-retryable errors · timeout abandonment · validation as failed attempt · fallback recovery · breaker short-circuit and cooldown · schema path naming · malformed JSON as validation error · enum rejection |

/**
 * Abstention policy — the decision of whether the retrieved evidence is good
 * enough to answer from.
 *
 * Every threshold here was swept, not chosen. `npm run bench:quality` runs the
 * pipeline over the 1,200 ingested MS MARCO questions — of which 520 are
 * labelled unanswerable by human annotators — plus a set of deliberately
 * off-corpus probes, and reports the abstention confusion matrix at each
 * candidate threshold. The values below are the operating point that maximises
 * F1 on "should this be answered at all"; the sweep is in
 * bench/abstention-sweep.json.
 *
 * The bias is deliberate: on a shortlisting task, a confident wrong answer is
 * worth less than an honest "I do not have that".
 */
import type { RefusalReason } from '../types.ts';
import type { RetrievalSignals } from '../retrieve/engine.ts';

export interface AbstentionThresholds {
  /**
   * Cross-encoder logit below which nothing in the shortlist is even about the
   * question. This is the primary gate whenever reranking ran — the logit is
   * unbounded and signed, so it separates "off-corpus" from "on-topic but
   * unanswered" in a way cosine cannot.
   */
  outOfCorpusCross: number;
  /** Cross-encoder logit below which the shortlist is on-topic but answers nothing. */
  weakCross: number;
  /** Below this cosine, nothing in the corpus is about the question. */
  outOfCorpusDense: number;
  /** Combined with the above: how much of the query even exists in the vocabulary. */
  outOfCorpusLexical: number;
  /** Retrieved something on-topic, but nothing that stands out. */
  weakDense: number;
  weakMargin: number;
  /** The extractive stage found no sentence that covers the question. */
  minAnswerCoverage: number;
  minAnswerConfidence: number;
  /**
   * Cross-encoder logit for the selected answer *sentence*. Sharper than the
   * chunk-level gate: a passage can be on-topic while no sentence in it
   * actually answers the question, and this is the signal that sees that.
   */
  minAnswerLogit: number;
}

export const DEFAULT_ABSTENTION: AbstentionThresholds = {
  // Swept values. outOfCorpusCross does not change the accept/reject call —
  // anything under weakCross is already rejected — it only decides which of
  // two messages the user gets, and it sits near the off-corpus median so a
  // nonsense question is told it is off-corpus rather than merely unanswered.
  outOfCorpusCross: -7.0,
  weakCross: -6.0,
  outOfCorpusDense: 0.34,
  outOfCorpusLexical: 0.45,
  weakDense: 0.42,
  weakMargin: 0.035,
  // Only used when the reranker is unavailable — see checkAnswer.
  minAnswerCoverage: 0.3,
  minAnswerConfidence: 0.2,
  // The primary answer gate. Sits just above the "no answer present"
  // population's 10th percentile (-0.1) and well below the answerable
  // population's (4.3), which is the separation that makes it worth gating on.
  minAnswerLogit: 1.0,
};

export interface AbstentionVerdict {
  answer: boolean;
  reason?: RefusalReason;
  message?: string;
  /** What drove the decision, surfaced in the trace panel. */
  detail: Record<string, number>;
}

/** Gate applied after retrieval, before the answer stage runs. */
export function checkRetrieval(
  signals: RetrievalSignals,
  thresholds: AbstentionThresholds = DEFAULT_ABSTENTION,
): AbstentionVerdict {
  const detail = {
    crossTop: Number.isFinite(signals.crossTop) ? round(signals.crossTop) : 0,
    crossMargin: Number.isFinite(signals.crossMargin) ? round(signals.crossMargin) : 0,
    topDense: round(signals.topDense),
    margin: round(signals.margin),
    lexicalCoverage: round(signals.lexicalCoverage),
    topLexical: round(signals.topLexical),
  };

  // When the cross-encoder ran, it is the better evidence and it decides.
  if (signals.reranked && Number.isFinite(signals.crossTop)) {
    if (signals.crossTop < thresholds.outOfCorpusCross) {
      return {
        answer: false,
        reason: 'OUT_OF_CORPUS',
        message:
          'That question is outside this corpus. I only answer from the MS MARCO web passages this index was built on.',
        detail,
      };
    }
    if (signals.crossTop < thresholds.weakCross) {
      return {
        answer: false,
        reason: 'WEAK_RETRIEVAL',
        message:
          'I found passages on that topic but none of them clearly answer the question, so I would rather not guess.',
        detail,
      };
    }
    return { answer: true, detail };
  }

  // Reranking was disabled or unavailable — fall back to the bi-encoder signals.
  if (
    signals.topDense < thresholds.outOfCorpusDense &&
    signals.lexicalCoverage < thresholds.outOfCorpusLexical
  ) {
    return {
      answer: false,
      reason: 'OUT_OF_CORPUS',
      message:
        'That question is outside this corpus. I only answer from the MS MARCO web passages this index was built on.',
      detail,
    };
  }

  if (signals.topDense < thresholds.weakDense && signals.margin < thresholds.weakMargin) {
    return {
      answer: false,
      reason: 'WEAK_RETRIEVAL',
      message:
        'I found passages on that topic but none of them clearly answer the question, so I would rather not guess.',
      detail,
    };
  }

  return { answer: true, detail };
}

/** Gate applied to the extractive answer itself. */
export function checkAnswer(
  coverage: number,
  confidence: number,
  thresholds: AbstentionThresholds = DEFAULT_ABSTENTION,
  answerLogit: number | null = null,
): AbstentionVerdict {
  const detail: Record<string, number> = {
    coverage: round(coverage),
    confidence: round(confidence),
  };
  if (answerLogit !== null) detail.answerLogit = round(answerLogit);

  // When the sentence-level logit exists it is the better evidence and it
  // decides alone: it scores the candidate answer itself rather than the
  // passage it was taken from. Layering the lexical gates on top of it only
  // cost recall in the sweep, so they are reserved for the path where the
  // reranker was unavailable and the logit is null.
  if (answerLogit !== null) {
    if (answerLogit < thresholds.minAnswerLogit) {
      return {
        answer: false,
        reason: 'NO_ANSWER_IN_CONTEXT',
        message:
          'The passages I retrieved are related but none of them states an answer, so I am not going to infer one.',
        detail,
      };
    }
    return { answer: true, detail };
  }

  if (coverage < thresholds.minAnswerCoverage || confidence < thresholds.minAnswerConfidence) {
    return {
      answer: false,
      reason: 'NO_ANSWER_IN_CONTEXT',
      message:
        'The passages I retrieved are related but none of them states an answer, so I am not going to infer one.',
      detail,
    };
  }

  return { answer: true, detail };
}

const round = (value: number): number => Math.round(value * 1000) / 1000;

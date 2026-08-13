/**
 * Input guardrails — everything that can be decided before retrieval runs.
 *
 * The design constraint is precision, not recall. This corpus is web text about
 * medicine, chemistry, law and history, so a keyword blocklist would refuse
 * "what is the lethal dose of acetaminophen" — a question whose answer is on a
 * poison-control page and which MS MARCO actually contains. Every rule here
 * therefore matches an *intent* pattern (an instruction verb plus a harmful
 * object), not a topic word, and the false-positive rate is measured against a
 * held-out set of benign medical and chemistry questions in
 * `src/core/guardrails/input.test.ts`.
 */
import type { RefusalReason } from '../types.ts';

export interface InputVerdict {
  ok: boolean;
  reason?: RefusalReason;
  message?: string;
  /** The query as it should proceed downstream — trimmed, injection stripped. */
  query: string;
  flags: string[];
}

const MIN_CHARS = 3;
const MAX_CHARS = 500;

/**
 * Instruction-shaped requests for genuinely harmful capability. Each pattern
 * pairs a "teach me to do it" frame with an object that has no benign reading
 * in that frame.
 */
const HARMFUL_INTENT: Array<{ pattern: RegExp; label: string }> = [
  {
    label: 'weapons',
    pattern: /\b(how (do i|to|can i)|steps? to|guide to|instructions? for|teach me to)\b[^?]{0,60}\b(build|make|construct|assemble|synthesi[sz]e)\b[^?]{0,40}\b(bomb|explosive|ied|pipe bomb|nerve agent|chemical weapon|biological weapon|dirty bomb|napalm|grenade)\b/i,
  },
  {
    label: 'drug-synthesis',
    pattern: /\b(how (do i|to|can i)|steps? to|recipe for|synthesi[sz]e|cook)\b[^?]{0,50}\b(meth(amphetamine)?|fentanyl|heroin|cocaine|lsd|mdma|krokodil)\b/i,
  },
  {
    label: 'self-harm',
    pattern: /\b(how (do i|to|can i)|best way to|easiest way to|painless way to)\b[^?]{0,40}\b(kill myself|commit suicide|end my life|hang myself|overdose on|cut myself)\b/i,
  },
  {
    label: 'targeted-harm',
    pattern: /\b(how (do i|to|can i)|best way to)\b[^?]{0,40}\b(kill|poison|murder|hurt|stalk|dox)\b\s+(my|his|her|their|the)?\s*\b(wife|husband|neighbou?r|boss|teacher|classmate|coworker|ex|someone|a person|people)\b/i,
  },
  {
    label: 'csam',
    pattern: /\b(child|minor|underage|preteen|kid)\b[^?]{0,30}\b(porn|nude|sexual|sex|explicit|erotica)\b|\b(porn|nude|sexual|explicit)\b[^?]{0,20}\b(child|minor|underage|preteen)\b/i,
  },
  {
    label: 'malware',
    pattern: /\b(write|create|build|generate|code)\b[^?]{0,40}\b(ransomware|keylogger|botnet|rootkit|computer virus|trojan|spyware)\b[^?]{0,40}\b(to (steal|encrypt|infect|attack)|that (steals|encrypts|infects))\b/i,
  },
  {
    label: 'credential-theft',
    pattern: /\b(how (do i|to|can i)|help me)\b[^?]{0,40}\b(hack|break into|gain access to|steal (the )?(password|credentials|account))\b[^?]{0,40}\b(someone'?s?|my (ex|neighbou?r|friend|wife|husband)|their|his|her)\b/i,
  },
];

/**
 * Prompt-injection markers. A transcript is untrusted input: it reaches a
 * generation stage, so an instruction hidden in speech ("ignore your rules and
 * ...") has to be neutralised. These are stripped rather than refused when the
 * rest of the utterance still reads as a question.
 */
const INJECTION: RegExp[] = [
  /\b(ignore|disregard|forget|override)\b[^.?!]{0,30}\b(previous|prior|above|earlier|all)\b[^.?!]{0,20}\b(instructions?|prompts?|rules?|context)\b/i,
  /\b(you are now|from now on you are|act as|pretend to be|roleplay as)\b[^.?!]{0,40}\b(dan|jailbroken|unrestricted|developer mode|no rules)\b/i,
  /\b(reveal|show|print|repeat|output)\b[^.?!]{0,25}\b(system prompt|your instructions|initial prompt|the prompt above)\b/i,
  /\b(do not|don'?t)\b[^.?!]{0,20}\b(cite|use|rely on)\b[^.?!]{0,20}\b(the )?(context|sources?|passages?|retrieval)\b/i,
];

/**
 * Questions about the asker's own private state.
 *
 * These are the one class of unanswerable question that retrieval scores
 * cannot catch, and testing showed exactly why: "what is my bank account
 * balance" retrieves a genuinely on-topic passage about trial balances, the
 * cross-encoder scores it positively, and the pipeline confidently answers a
 * question about accounting to someone asking about their own money. No corpus
 * can hold that answer, so the check belongs here, before retrieval runs.
 *
 * The rule is narrow on purpose. It fires only on interrogatives asking for
 * the *value* of a first-person-possessed private thing, so "how do I change
 * my password" and similar how-to questions — which this corpus does answer —
 * pass straight through.
 */
const PERSONAL_STATE = new RegExp(
  // asks for a value, not a procedure — "how do i…" deliberately does not match
  String.raw`^(?:what|when|where|which|who|how much|how many|how long)\b[^?]{0,30}` +
    String.raw`\b(?:my|our)\s+` +
    // an enumerated qualifier may sit between the possessive and the noun
    // ("my next appointment"); anything else does not count
    String.raw`(?:(?:next|last|latest|first|current|upcoming|previous|monthly|annual|total|exact|bank|savings|checking|email|user|home|work)\s+){0,2}` +
    String.raw`(?:account|balance|password|passcode|pin|salary|paycheck|appointment|booking|reservation|order|prescription|schedule|calendar|address|phone number|inbox|credit score|policy number|itinerary|meeting|grade|score|ticket|refund|invoice|subscription)\b`,
  'i',
);

/**
 * Utterances that are conversation, not retrieval. Matched as a whole
 * utterance, allowing the filler that trails a spoken greeting ("hi there",
 * "thanks a lot") — a transcript rarely arrives as a bare token.
 */
const SMALL_TALK =
  /^\s*(?:hi|hey|hello|yo|namaste|hola|thanks?|thank you|thankyou|ok(?:ay)?|cool|nice|great|bye|goodbye|good (?:morning|afternoon|evening|night)|how are you(?: doing)?|what'?s up|who are you|what can you do)\b(?:\s+(?:there|again|a lot|so much|man|buddy|friend|everyone|guys))*[\s!.?]*$/i;

/** Text with no interrogative shape and no content — usually a mis-fired mic. */
const FILLER = /^[\s.,!?—–-]*(?:u+m+|u+h+|a+h+|e+r+|hm+|mm+|huh|test(ing)?(\s+1\s*2\s*3)?)[\s.,!?]*$/i;

export function checkInput(raw: string): InputVerdict {
  const flags: string[] = [];
  const trimmed = (raw ?? '').replace(/\s+/g, ' ').trim();

  if (trimmed.length === 0) {
    return {
      ok: false,
      reason: 'EMPTY_QUERY',
      message: 'I did not catch a question — the audio came through empty.',
      query: '',
      flags: ['empty'],
    };
  }

  if (FILLER.test(trimmed) || trimmed.length < MIN_CHARS) {
    return {
      ok: false,
      reason: 'EMPTY_QUERY',
      message: 'That came through as filler rather than a question. Try asking again.',
      query: trimmed,
      flags: ['filler'],
    };
  }

  if (SMALL_TALK.test(trimmed)) {
    return {
      ok: false,
      reason: 'NOT_A_QUESTION',
      message:
        'I answer questions from a MS MARCO web-passage corpus — ask me something factual and I will look it up.',
      query: trimmed,
      flags: ['small-talk'],
    };
  }

  if (PERSONAL_STATE.test(trimmed)) {
    return {
      ok: false,
      reason: 'PERSONAL_CONTEXT',
      message:
        'I can only answer from a fixed corpus of public web passages — I have no access to your accounts, calendar or records.',
      query: trimmed,
      flags: ['personal-state'],
    };
  }

  for (const rule of HARMFUL_INTENT) {
    if (rule.pattern.test(trimmed)) {
      return {
        ok: false,
        reason: 'UNSAFE_INPUT',
        message: 'I will not help with that. Ask me something else and I will search the corpus for it.',
        query: trimmed,
        flags: [`unsafe:${rule.label}`],
      };
    }
  }

  // Strip injection spans rather than refusing: a real question wrapped in an
  // injection attempt is still a real question.
  let cleaned = trimmed;
  for (const pattern of INJECTION) {
    if (pattern.test(cleaned)) {
      flags.push('injection-stripped');
      cleaned = cleaned.replace(pattern, ' ').replace(/\s+/g, ' ').trim();
    }
  }

  if (flags.includes('injection-stripped') && cleaned.length < MIN_CHARS) {
    return {
      ok: false,
      reason: 'PROMPT_INJECTION',
      message: 'That request was an instruction to me rather than a question about the corpus.',
      query: trimmed,
      flags,
    };
  }

  if (cleaned.length > MAX_CHARS) {
    cleaned = cleaned.slice(0, MAX_CHARS);
    flags.push('truncated');
  }

  return { ok: true, query: cleaned, flags };
}

/**
 * Text primitives shared by chunking, indexing, retrieval and grounding.
 *
 * Everything here works in character offsets against the original passage
 * string. Chunks are stored as (passageId, start, end) triples rather than
 * copies of the text, so the whole 117k-chunk index costs the same text memory
 * as the 11.9k passages it was derived from.
 */

export interface Span {
  start: number;
  end: number;
}

/** Abbreviations that end in a period but do not end a sentence. */
const ABBREVIATIONS = new Set([
  'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'mt', 'ft', 'rev', 'hon', 'gen', 'col', 'lt',
  'sgt', 'capt', 'cmdr', 'adm', 'gov', 'sen', 'rep', 'pres', 'supt', 'det', 'insp',
  'inc', 'ltd', 'co', 'corp', 'llc', 'plc', 'dept', 'univ', 'assn', 'bros', 'est',
  'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec',
  'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
  'no', 'vol', 'pp', 'ed', 'eds', 'fig', 'al', 'etc', 'vs', 'approx', 'min', 'max', 'avg',
  'e.g', 'i.e', 'cf', 'viz', 'sq', 'cu', 'oz', 'lb', 'lbs', 'kg', 'km', 'cm', 'mm', 'ml',
  'u.s', 'u.k', 'u.n', 'a.m', 'p.m', 'ph.d', 'b.a', 'm.a', 'm.d', 'b.s', 'm.s',
]);

/**
 * Splits text into sentence spans.
 *
 * Handles the three things that break naive `split('.')` on MS MARCO passages:
 * abbreviations ("Dr. Smith"), decimals and version numbers ("3.14", "v2.1"),
 * and enumerations ("1. First item"). Falls back to hard-wrapping any sentence
 * that runs past `maxChars` so a passage with no terminal punctuation still
 * produces usable units.
 */
export function splitSentences(text: string, maxChars = 400): Span[] {
  const spans: Span[] = [];
  let start = 0;
  let i = 0;

  const pushSpan = (from: number, to: number): void => {
    // trim whitespace at both edges without losing offset alignment
    let s = from;
    let e = to;
    while (s < e && /\s/.test(text[s])) s++;
    while (e > s && /\s/.test(text[e - 1])) e--;
    if (e - s >= 2) spans.push({ start: s, end: e });
  };

  while (i < text.length) {
    const ch = text[i];

    if (ch === '.' || ch === '!' || ch === '?' || ch === '\n') {
      if (ch === '.' && isNonTerminalPeriod(text, i)) {
        i++;
        continue;
      }
      // absorb trailing quotes/brackets and repeated terminators
      let end = i + 1;
      while (end < text.length && /["'”’)\]}.!?]/.test(text[end])) end++;
      // a sentence must be followed by whitespace or the end of the text
      if (end < text.length && !/\s/.test(text[end]) && ch !== '\n') {
        i++;
        continue;
      }
      pushSpan(start, end);
      start = end;
      i = end;
      continue;
    }
    i++;
  }
  if (start < text.length) pushSpan(start, text.length);

  // hard-wrap over-long sentences at the nearest clause or word boundary
  const wrapped: Span[] = [];
  for (const span of spans) {
    if (span.end - span.start <= maxChars) {
      wrapped.push(span);
      continue;
    }
    let from = span.start;
    while (from < span.end) {
      let to = Math.min(from + maxChars, span.end);
      if (to < span.end) {
        const slice = text.slice(from, to);
        const cut = Math.max(slice.lastIndexOf('; '), slice.lastIndexOf(', '), slice.lastIndexOf(' '));
        if (cut > maxChars * 0.5) to = from + cut + 1;
      }
      wrapped.push({ start: from, end: to });
      from = to;
    }
  }
  return wrapped;
}

/** True when the period at `i` belongs to an abbreviation, decimal or initial. */
function isNonTerminalPeriod(text: string, i: number): boolean {
  const next = text[i + 1];
  const prev = text[i - 1];

  // decimals and version numbers: 3.14, v2.1
  if (/\d/.test(prev ?? '') && /\d/.test(next ?? '')) return true;
  // single-letter initials: J. R. R. Tolkien
  if (/[A-Za-z]/.test(prev ?? '') && !/[A-Za-z]/.test(text[i - 2] ?? ' ')) {
    if (next === ' ' && /[A-Z]/.test(text[i + 2] ?? '')) return true;
  }
  // ellipsis
  if (next === '.' || prev === '.') return true;

  // known abbreviation immediately before the period
  let wordStart = i;
  while (wordStart > 0 && /[A-Za-z.]/.test(text[wordStart - 1])) wordStart--;
  const word = text.slice(wordStart, i).toLowerCase().replace(/^\.+/, '');
  if (word && ABBREVIATIONS.has(word)) return true;

  // enumerations: "1. " / "a) "
  if (/^\d{1,2}$/.test(word) && wordStart === 0) return true;

  return false;
}

/** Clause boundaries inside a sentence — the raw material for propositions. */
const CLAUSE_SPLIT =
  /(?:,\s+(?:which|who|whom|whose|where|when|while|although|though|because|since|so that|such that|and|but|or|yet)\s)|(?:;\s)|(?:\s+—\s+)|(?:\s+--\s+)/gi;

/** Splits a sentence span into clause spans, keeping character offsets. */
export function splitClauses(text: string, span: Span, minChars = 45): Span[] {
  const slice = text.slice(span.start, span.end);
  const cuts: number[] = [];
  CLAUSE_SPLIT.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = CLAUSE_SPLIT.exec(slice)) !== null) {
    // cut after the separator, keeping the connective with the following clause
    const at = match.index + (match[0].startsWith(',') ? 2 : match[0].length);
    if (at > minChars && slice.length - at > minChars) cuts.push(at);
  }
  if (cuts.length === 0) return [span];

  const spans: Span[] = [];
  let from = 0;
  for (const cut of [...cuts, slice.length]) {
    if (cut - from >= minChars) {
      spans.push({ start: span.start + from, end: span.start + cut });
      from = cut;
    }
  }
  if (from < slice.length && spans.length > 0) {
    spans[spans.length - 1].end = span.end; // absorb a short tail
  }
  return spans.length > 0 ? spans : [span];
}

// --------------------------------------------------------------------------
// Tokenization
// --------------------------------------------------------------------------

export const STOPWORDS = new Set([
  'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'of', 'to', 'in', 'on', 'at', 'by',
  'for', 'with', 'about', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am', 'do',
  'does', 'did', 'doing', 'have', 'has', 'had', 'having', 'i', 'you', 'he', 'she', 'it', 'we',
  'they', 'this', 'that', 'these', 'those', 'from', 'into', 'over', 'under', 'their', 'its',
  'his', 'her', 'them', 'there', 'here', 'not', 'no', 'so', 'than', 'too', 'very', 's', 't',
]);

/**
 * Question words carry the intent of a query even though they are stopwords for
 * scoring, so routing looks at them separately.
 */
export const QUESTION_WORDS = new Set([
  'what', 'when', 'where', 'who', 'whom', 'whose', 'which', 'why', 'how', 'is', 'are', 'was',
  'were', 'do', 'does', 'did', 'can', 'could', 'should', 'would', 'will', 'define', 'meaning',
]);

const TOKEN_RE = /[\p{L}\p{N}]+(?:['’][\p{L}]+)?/gu;

/** Lowercased alphanumeric tokens. Unicode-aware so Indic display text tokenizes too. */
export function tokenize(text: string): string[] {
  return text.toLowerCase().match(TOKEN_RE) ?? [];
}

/** Tokens with their character offsets, for highlighting and proximity scoring. */
export function tokenizeWithOffsets(text: string): Array<{ token: string; start: number; end: number }> {
  const out: Array<{ token: string; start: number; end: number }> = [];
  TOKEN_RE.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = TOKEN_RE.exec(text)) !== null) {
    out.push({ token: match[0].toLowerCase(), start: match.index, end: match.index + match[0].length });
  }
  return out;
}

/**
 * Light suffix stemmer. Not Porter — deliberately conservative, because over-
 * stemming hurts exact-match precision on the numeric and entity queries that
 * make up a third of MS MARCO, and every rule here costs query latency.
 */
export function stem(token: string): string {
  if (token.length <= 3) return token;
  if (token.endsWith('ies') && token.length > 4) return `${token.slice(0, -3)}y`;
  if (token.endsWith('sses')) return token.slice(0, -2);
  if (token.endsWith('ss')) return token;
  if (token.endsWith('s') && !token.endsWith('us') && !token.endsWith('is')) return token.slice(0, -1);
  if (token.endsWith('ing') && token.length > 5) {
    const base = token.slice(0, -3);
    return /[aeiou]/.test(base) ? base : token;
  }
  if (token.endsWith('ed') && token.length > 4) {
    const base = token.slice(0, -2);
    return /[aeiou]/.test(base) ? base : token;
  }
  return token;
}

/** Analysis pipeline used on both sides of the lexical index. */
export function analyze(text: string, { keepStopwords = false } = {}): string[] {
  const out: string[] = [];
  for (const token of tokenize(text)) {
    if (!keepStopwords && STOPWORDS.has(token)) continue;
    out.push(stem(token));
  }
  return out;
}

/** Collapses whitespace and trims — used for display and hashing. */
export function normalizeWhitespace(text: string): string {
  return text.replace(/\s+/g, ' ').trim();
}

/**
 * Numbers, dates, money and units mentioned in a text. Grounding treats these
 * as hard constraints: a generated answer may not contain a figure that does
 * not appear in the retrieved context.
 */
export function extractNumerics(text: string): string[] {
  const out = new Set<string>();
  const re = /(?:\$\s?)?\d[\d,]*(?:\.\d+)?\s?(?:%|percent|million|billion|trillion|thousand|kg|lb|lbs|km|mi|miles|ft|feet|inches|in|cm|mm|°|degrees|f|c|years?|months?|days?|hours?|minutes?|seconds?)?/gi;
  let match: RegExpExecArray | null;
  while ((match = re.exec(text)) !== null) {
    const raw = match[0].trim().toLowerCase();
    // keep only tokens that actually contain a digit run of interest
    const digits = raw.replace(/[^\d.]/g, '');
    if (digits.length > 0) out.add(digits.replace(/\.$/, ''));
  }
  return [...out];
}

/** Capitalised multi-word sequences — a cheap proper-noun proxy for grounding. */
export function extractEntities(text: string): string[] {
  const out = new Set<string>();
  const re = /\b[A-Z][a-z]{1,}(?:\s+(?:of|the|and|de|van|von)?\s*[A-Z][a-z]{1,})*/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(text)) !== null) {
    const value = match[0].trim();
    if (value.length > 2 && !STOPWORDS.has(value.toLowerCase())) out.add(value.toLowerCase());
  }
  return [...out];
}

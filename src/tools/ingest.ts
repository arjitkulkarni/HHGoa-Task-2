/**
 * Pulls a working corpus out of ai4bharat/MSMARCO-XI without downloading it.
 *
 * The full dataset is 55.6 GB. We stream a bounded byte window of the primary
 * language file to get complete rows (query, answer, 10 passages, relevance
 * labels), then pull only the two small query columns from the other language
 * files so the same question is available in several Indic languages.
 *
 * Output (data/raw/):
 *   passages.jsonl  one deduplicated passage per line, with its provenance
 *   queries.jsonl   one question per line, with gold passage ids and translations
 *   manifest.json   what was fetched, how much was transferred
 *
 * Usage: node src/tools/ingest.ts [--queries 1200] [--primary hin] [--extra tam,ben,mar,tel]
 */
import { createWriteStream } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { readParquetWindow } from './parquet-window.ts';

const REPO = 'https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main';
const fileUrl = (lang: string) => `${REPO}/validation/${lang}val.parquet`;

/** BCP-47 tags for the Indic language codes MSMARCO-XI uses in its filenames. */
export const LANGUAGES: Record<string, { name: string; bcp47: string }> = {
  asm: { name: 'Assamese', bcp47: 'as-IN' },
  ben: { name: 'Bengali', bcp47: 'bn-IN' },
  guj: { name: 'Gujarati', bcp47: 'gu-IN' },
  hin: { name: 'Hindi', bcp47: 'hi-IN' },
  kan: { name: 'Kannada', bcp47: 'kn-IN' },
  mal: { name: 'Malayalam', bcp47: 'ml-IN' },
  mar: { name: 'Marathi', bcp47: 'mr-IN' },
  nep: { name: 'Nepali', bcp47: 'ne-IN' },
  ori: { name: 'Odia', bcp47: 'od-IN' },
  pan: { name: 'Punjabi', bcp47: 'pa-IN' },
  san: { name: 'Sanskrit', bcp47: 'sa-IN' },
  tam: { name: 'Tamil', bcp47: 'ta-IN' },
  tel: { name: 'Telugu', bcp47: 'te-IN' },
  urd: { name: 'Urdu', bcp47: 'ur-IN' },
};

interface PrimaryRow {
  query_id: bigint;
  query_type: string;
  Eng_Query: string;
  Eng_Answer: string;
  query: string;
  Answer: string;
  passages: {
    English_passages: string[];
    Translated_passages: string[];
    is_selected: bigint[];
  };
}

interface ExtraRow {
  query_id: bigint;
  query: string;
  Answer: string;
}

export interface PassageRecord {
  id: number;
  text: string;
  /** Same passage in the primary Indic language, for display. */
  alt: string;
  /** Every (query, rank, label) this passage was retrieved for. */
  refs: Array<{ qid: number; rank: number; selected: boolean }>;
}

export interface QueryRecord {
  qid: number;
  /** MS MARCO's own taxonomy: DESCRIPTION | NUMERIC | ENTITY | LOCATION | PERSON. */
  type: string;
  question: string;
  answer: string;
  /** MS MARCO marks unanswerable questions explicitly; we keep them as abstention tests. */
  noAnswer: boolean;
  gold: number[];
  translations: Record<string, { question: string; answer: string }>;
}

const NO_ANSWER = /^no answer present\.?$/i;

function parseArgs(argv: string[]): { queries: number; primary: string; extra: string[] } {
  const get = (flag: string, fallback: string): string => {
    const i = argv.indexOf(flag);
    return i >= 0 && argv[i + 1] ? argv[i + 1] : fallback;
  };
  return {
    queries: Number(get('--queries', '1200')),
    primary: get('--primary', 'hin'),
    extra: get('--extra', 'tam,ben,mar,tel')
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean),
  };
}

/** Collapses whitespace so near-identical passages dedupe against each other. */
function normalizeKey(text: string): string {
  return text.toLowerCase().replace(/\s+/g, ' ').replace(/[^\w\s]/g, '').trim();
}

async function main(): Promise<void> {
  const { queries: wanted, primary, extra } = parseArgs(process.argv.slice(2));
  if (!LANGUAGES[primary]) throw new Error(`unknown primary language "${primary}"`);

  const outDir = resolve(process.cwd(), 'data/raw');
  await mkdir(outDir, { recursive: true });

  // Windows are sized from the file's own bytes-per-row, with headroom for the
  // fact that pages only truncate on complete boundaries.
  const bytesPerRow = { english: 1800, translated: 2850 };
  const window = (perRow: number) => Math.max(4e6, Math.ceil(wanted * perRow * 2.4));

  console.log(`\n▸ primary language: ${LANGUAGES[primary].name} (${primary})`);
  console.log(`  target: ${wanted} queries\n`);

  const primaryResult = await readParquetWindow<PrimaryRow>({
    url: fileUrl(primary),
    columns: ['query_id', 'query_type', 'Eng_Query', 'Eng_Answer', 'query', 'Answer', 'passages'],
    maxRows: wanted,
    windows: {
      'passages.English_passages.list.element': window(bytesPerRow.english),
      'passages.Translated_passages.list.element': window(bytesPerRow.translated),
    },
    onProgress: (m) => console.log(m),
  });

  console.log(
    `  read ${primaryResult.rows.length} rows for ${(primaryResult.bytesFetched / 1e6).toFixed(1)} MB ` +
      `of ${(primaryResult.totalFileBytes / 1e6).toFixed(0)} MB ` +
      `(${((primaryResult.bytesFetched / primaryResult.totalFileBytes) * 100).toFixed(1)}% of the file)\n`,
  );

  let bytesTotal = primaryResult.bytesFetched;

  // The language files share MS MARCO's row order, so the small query columns
  // line up by query_id — a few MB each buys the same question in 5 languages.
  const extraByLang = new Map<string, Map<number, { question: string; answer: string }>>();
  for (const lang of extra) {
    if (!LANGUAGES[lang]) {
      console.log(`  skipping unknown language "${lang}"`);
      continue;
    }
    try {
      const result = await readParquetWindow<ExtraRow>({
        url: fileUrl(lang),
        columns: ['query_id', 'query', 'Answer'],
        maxRows: wanted,
      });
      const map = new Map<number, { question: string; answer: string }>();
      for (const row of result.rows) {
        if (row?.query_id === undefined || !row.query) continue;
        map.set(Number(row.query_id), { question: row.query, answer: row.Answer ?? '' });
      }
      extraByLang.set(lang, map);
      bytesTotal += result.bytesFetched;
      console.log(
        `  + ${LANGUAGES[lang].name.padEnd(10)} ${String(map.size).padStart(5)} queries ` +
          `(${(result.bytesFetched / 1e6).toFixed(1)} MB)`,
      );
    } catch (error) {
      console.log(`  ! ${lang} failed: ${(error as Error).message}`);
    }
  }

  // ---- assemble ----
  const passages: PassageRecord[] = [];
  const passageIdByKey = new Map<string, number>();
  const queries: QueryRecord[] = [];

  for (const row of primaryResult.rows) {
    const english = row?.passages?.English_passages;
    if (!row?.Eng_Query || !Array.isArray(english) || english.length === 0) continue;

    const qid = Number(row.query_id);
    const translated = row.passages.Translated_passages ?? [];
    const selectedFlags = row.passages.is_selected ?? [];
    const gold: number[] = [];

    for (let rank = 0; rank < english.length; rank++) {
      const text = (english[rank] ?? '').trim();
      if (text.length < 40) continue;
      const selected = Number(selectedFlags[rank] ?? 0) === 1;
      const key = normalizeKey(text);

      let id = passageIdByKey.get(key);
      if (id === undefined) {
        id = passages.length;
        passageIdByKey.set(key, id);
        passages.push({ id, text, alt: (translated[rank] ?? '').trim(), refs: [] });
      }
      passages[id].refs.push({ qid, rank, selected });
      if (selected) gold.push(id);
    }

    const answer = (row.Eng_Answer ?? '').trim();
    const translations: QueryRecord['translations'] = {
      [primary]: { question: (row.query ?? '').trim(), answer: (row.Answer ?? '').trim() },
    };
    for (const [lang, map] of extraByLang) {
      const hit = map.get(qid);
      if (hit?.question) translations[lang] = hit;
    }

    queries.push({
      qid,
      type: (row.query_type ?? 'DESCRIPTION').toUpperCase(),
      question: row.Eng_Query.trim(),
      answer: NO_ANSWER.test(answer) ? '' : answer,
      noAnswer: !answer || NO_ANSWER.test(answer),
      gold: [...new Set(gold)],
      translations,
    });
  }

  const writeJsonl = async (name: string, records: unknown[]): Promise<void> => {
    const stream = createWriteStream(resolve(outDir, name));
    for (const record of records) stream.write(`${JSON.stringify(record)}\n`);
    await new Promise<void>((res, rej) => stream.end(() => res()).on('error', rej));
  };

  await writeJsonl('passages.jsonl', passages);
  await writeJsonl('queries.jsonl', queries);

  const answerable = queries.filter((q) => !q.noAnswer).length;
  const withGold = queries.filter((q) => q.gold.length > 0).length;
  const byType = new Map<string, number>();
  for (const q of queries) byType.set(q.type, (byType.get(q.type) ?? 0) + 1);

  const manifest = {
    source: 'ai4bharat/MSMARCO-XI (validation split)',
    fetchedAt: new Date().toISOString(),
    primaryLanguage: primary,
    extraLanguages: [...extraByLang.keys()],
    queries: queries.length,
    answerable,
    unanswerable: queries.length - answerable,
    withGoldPassage: withGold,
    passages: passages.length,
    passageChars: passages.reduce((sum, p) => sum + p.text.length, 0),
    queryTypes: Object.fromEntries([...byType].sort((a, b) => b[1] - a[1])),
    bytesTransferred: bytesTotal,
    datasetBytes: 55_619_599_557,
    fractionOfDatasetTransferred: bytesTotal / 55_619_599_557,
  };
  await writeFile(resolve(outDir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);

  console.log(`\n▸ corpus`);
  console.log(`  ${passages.length} unique passages (${(manifest.passageChars / 1e6).toFixed(1)} M chars)`);
  console.log(`  ${queries.length} queries — ${answerable} answerable, ${queries.length - answerable} explicitly unanswerable`);
  console.log(`  ${withGold} queries have at least one human-labelled relevant passage`);
  console.log(`  types: ${[...byType].map(([t, n]) => `${t} ${n}`).join(', ')}`);
  console.log(`\n  transferred ${(bytesTotal / 1e6).toFixed(1)} MB of a 55.6 GB dataset (${(manifest.fractionOfDatasetTransferred * 100).toFixed(3)}%)`);
  console.log(`  → data/raw/\n`);
}

await main();

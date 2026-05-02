#!/usr/bin/env tsx
/**
 * Read a JSONL produced by `seed-trigger-keywords.ts` and emit a
 * ranked list of candidate keywords for a group's
 * `trigger_pattern.patterns[]`.
 *
 * Gold-cell filter: stage1.decision != 'allow'
 *                  AND stage2 != null
 *                  AND stage2.intent == 'yes'
 *                  AND botReplied == true.
 *
 * (Stage 1 only `pass`-es when the group config has only
 * forward-looking pattern kinds — sender_tier / regex. Real groups
 * with a keyword/mention list reach `deny` instead of `pass` when
 * the list misses; both outcomes are equally "Stage 1 missed" for
 * keyword-discovery purposes, hence `!= allow`.)
 *
 * Tokenisation uses `Intl.Segmenter` (Node 22 supports it) so mixed
 * RU/EN messages tokenise correctly — a naive `\w+` regex corrupts
 * Cyrillic.
 *
 * READ-ONLY tool. Writes the rendered table to
 * `<input>.candidates.md` next to the JSONL; never touches the DB.
 *
 * Usage:
 *   tsx scripts/extract-keyword-candidates.ts \
 *     --input <jsonl> [--top-n 50] [--min-frequency 3] \
 *     [--existing-patterns-jid <jid>] [--db <path>]
 */
import fs from 'fs';
import path from 'path';
import readline from 'readline';
import { pathToFileURL } from 'url';

import Database from 'better-sqlite3';

import type { TriggerPattern, TriggerPatternConfig } from '../src/types.js';

interface Args {
  input: string;
  topN: number;
  minFrequency: number;
  existingPatternsJid: string | null;
  dbPath: string;
}

function parseArgs(argv: string[]): Args {
  const out: Partial<Args> = {
    topN: 50,
    minFrequency: 3,
    existingPatternsJid: null,
    dbPath: path.resolve(process.cwd(), 'store/messages.db'),
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    switch (a) {
      case '--input':
        out.input = argv[++i];
        break;
      case '--top-n':
        out.topN = Number(argv[++i]);
        break;
      case '--min-frequency':
        out.minFrequency = Number(argv[++i]);
        break;
      case '--existing-patterns-jid':
        out.existingPatternsJid = argv[++i];
        break;
      case '--db':
        out.dbPath = path.resolve(argv[++i]);
        break;
      case '--help':
      case '-h':
        printHelp();
        process.exit(0);
        break;
      default:
        throw new Error(`Unknown argument: ${a}`);
    }
  }
  if (!out.input) throw new Error('--input <path> is required');
  if (!Number.isFinite(out.topN) || (out.topN as number) <= 0) {
    throw new Error('--top-n must be a positive number');
  }
  if (!Number.isFinite(out.minFrequency) || (out.minFrequency as number) <= 0) {
    throw new Error('--min-frequency must be a positive number');
  }
  return out as Args;
}

function printHelp(): void {
  process.stdout.write(
    [
      'Usage: tsx scripts/extract-keyword-candidates.ts [options]',
      '',
      'Required:',
      '  --input <path>                 JSONL file from seed-trigger-keywords',
      '',
      'Optional:',
      '  --top-n <N>                    Top N candidates (default 50)',
      '  --min-frequency <N>            Drop n-grams below this count (default 3)',
      '  --existing-patterns-jid <jid>  Dedupe against current group patterns',
      '  --db <path>                    Path to messages.db (default store/messages.db)',
      '',
    ].join('\n'),
  );
}

// ---------------------------------------------------------------------------
// Stopwords. Small inline list — this is a one-off.
// ---------------------------------------------------------------------------
const STOPWORDS_EN = new Set([
  'a',
  'an',
  'the',
  'is',
  'are',
  'was',
  'were',
  'be',
  'been',
  'being',
  'have',
  'has',
  'had',
  'do',
  'does',
  'did',
  'of',
  'in',
  'on',
  'at',
  'to',
  'for',
  'with',
  'by',
  'from',
  'as',
  'and',
  'or',
  'but',
  'if',
  'so',
  'not',
  'no',
  'yes',
  'i',
  'you',
  'he',
  'she',
  'it',
  'we',
  'they',
  'me',
  'him',
  'her',
  'us',
  'them',
  'my',
  'your',
  'his',
  'its',
  'our',
  'their',
  'this',
  'that',
  'these',
  'those',
  'what',
  'which',
  'who',
  'whom',
  'whose',
  'when',
  'where',
  'why',
  'how',
  'will',
  'would',
  'should',
  'could',
  'can',
  'may',
  'might',
  'must',
  'just',
  'now',
  'also',
  'too',
  'very',
  'really',
  'okay',
  'ok',
  'yeah',
  'yep',
  'nope',
  'lol',
]);

const STOPWORDS_RU = new Set([
  'и',
  'в',
  'во',
  'не',
  'что',
  'он',
  'на',
  'я',
  'с',
  'со',
  'как',
  'а',
  'то',
  'все',
  'она',
  'так',
  'его',
  'но',
  'да',
  'ты',
  'к',
  'у',
  'же',
  'вы',
  'за',
  'бы',
  'по',
  'только',
  'ее',
  'мне',
  'было',
  'вот',
  'от',
  'меня',
  'еще',
  'нет',
  'о',
  'из',
  'ему',
  'теперь',
  'когда',
  'даже',
  'ну',
  'вдруг',
  'ли',
  'если',
  'уже',
  'или',
  'ни',
  'быть',
  'был',
  'него',
  'до',
  'этот',
  'этого',
  'этом',
  'эту',
  'эти',
  'есть',
  'для',
  'раз',
  'тоже',
  'себе',
  'под',
  'будет',
  'ж',
  'тогда',
  'кто',
  'этот',
  'того',
  'потому',
  'этого',
  'какой',
  'совсем',
  'ничего',
  'там',
  'может',
  'надо',
  'нее',
  'сейчас',
  'были',
  'куда',
  'зачем',
  'всех',
  'никогда',
  'можно',
  'при',
  'наконец',
  'два',
  'об',
  'другой',
  'хоть',
  'после',
  'над',
  'больше',
  'тот',
  'через',
  'эти',
  'нас',
  'про',
  'всего',
  'них',
  'какая',
  'много',
  'разве',
  'три',
  'эту',
  'моя',
  'впрочем',
  'хорошо',
  'свою',
  'этой',
  'перед',
  'иногда',
  'лучше',
  'чуть',
  'том',
  'нельзя',
  'такой',
  'им',
  'более',
  'всегда',
  'конечно',
  'всю',
  'между',
]);

const STOPWORDS = new Set([...STOPWORDS_EN, ...STOPWORDS_RU]);

// ---------------------------------------------------------------------------
// Tokeniser. Intl.Segmenter is unicode-aware; we treat any segment
// classified as `isWordLike` as a candidate token after lowercasing
// and stripping leading/trailing punctuation. Strips bracket-prefixes
// like `[Replying to ...]` that the orchestrator inserts (those are
// noise for keyword discovery).
// ---------------------------------------------------------------------------
const SEGMENTER = new Intl.Segmenter(['en', 'ru'], { granularity: 'word' });

function stripBracketPrefix(text: string): string {
  // Remove leading `[Replying to ...]\n` blocks the orchestrator
  // synthesises in `resolveReply`. They make every reply look
  // identical and would dominate frequency counts.
  let t = text;
  while (t.startsWith('[')) {
    const end = t.indexOf(']');
    if (end === -1) break;
    const after = t.slice(end + 1).replace(/^\s+/, '');
    t = after;
  }
  return t;
}

function tokenize(text: string): string[] {
  const cleaned = stripBracketPrefix(text);
  const tokens: string[] = [];
  for (const seg of SEGMENTER.segment(cleaned)) {
    if (!seg.isWordLike) continue;
    const word = seg.segment
      .toLowerCase()
      .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, '');
    if (!word) continue;
    if (word.length < 2) continue;
    // Drop pure numbers — not useful as keyword triggers.
    if (/^\d+$/.test(word)) continue;
    if (STOPWORDS.has(word)) continue;
    tokens.push(word);
  }
  return tokens;
}

function ngrams(tokens: string[], n: number): string[] {
  if (tokens.length < n) return [];
  const out: string[] = [];
  for (let i = 0; i + n <= tokens.length; i++) {
    out.push(tokens.slice(i, i + n).join(' '));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Existing-patterns dedupe (read-only DB lookup, only when requested)
// ---------------------------------------------------------------------------
function loadExistingPatterns(
  dbPath: string,
  jid: string,
): TriggerPatternConfig | null {
  if (!fs.existsSync(dbPath)) {
    throw new Error(`messages.db not found at ${dbPath}`);
  }
  const db = new Database(dbPath, { readonly: true, fileMustExist: true });
  const row = db
    .prepare(`SELECT trigger_pattern FROM registered_groups WHERE jid = ?`)
    .get(jid) as { trigger_pattern: string } | undefined;
  db.close();
  if (!row) return null;
  const trimmed = row.trigger_pattern.trim();
  if (!trimmed) return null;
  if (!trimmed.startsWith('{')) {
    // Legacy single-string trigger.
    const mentionMatch = /^@([a-zA-Z0-9_]+)$/.exec(trimmed);
    const pattern: TriggerPattern = mentionMatch
      ? {
          pattern: mentionMatch[1],
          kind: 'mention',
          source: 'owner-set',
          precision: 0,
          sample_count: 0,
          last_matched_at: null,
          last_updated_at: null,
        }
      : {
          pattern: row.trigger_pattern,
          kind: 'keyword',
          source: 'owner-set',
          precision: 0,
          sample_count: 0,
          last_matched_at: null,
          last_updated_at: null,
        };
    return { version: 1, patterns: [pattern] };
  }
  try {
    const parsed = JSON.parse(trimmed);
    if (parsed && parsed.version === 1 && Array.isArray(parsed.patterns)) {
      return parsed as TriggerPatternConfig;
    }
  } catch (err) {
    if (!(err instanceof SyntaxError)) throw err;
  }
  return null;
}

function buildExistingPatternSet(
  config: TriggerPatternConfig | null,
): Set<string> {
  const set = new Set<string>();
  if (!config) return set;
  for (const p of config.patterns) {
    if (p.kind === 'keyword' || p.kind === 'mention') {
      set.add(p.pattern.toLowerCase().trim());
    }
  }
  return set;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
interface JsonlRow {
  messageId: string;
  groupJid: string;
  timestamp: string;
  senderJid: string;
  senderName: string;
  text: string;
  stage1: { decision: string };
  stage2: null | { decision: string; intent: string | null };
  botReplied: boolean;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv);
  const inputPath = path.resolve(args.input);
  if (!fs.existsSync(inputPath)) {
    throw new Error(`input not found: ${inputPath}`);
  }

  const existingPatterns = args.existingPatternsJid
    ? loadExistingPatterns(args.dbPath, args.existingPatternsJid)
    : null;
  const existingSet = buildExistingPatternSet(existingPatterns);

  const stream = fs.createReadStream(inputPath, { encoding: 'utf-8' });
  const rl = readline.createInterface({ input: stream });

  // n-gram -> { count, examples (first 3 messages it appeared in) }
  const counts = new Map<string, { count: number; examples: string[] }>();
  let totalRows = 0;
  let goldRows = 0;

  for await (const line of rl) {
    if (!line.trim()) continue;
    totalRows++;
    let row: JsonlRow;
    try {
      row = JSON.parse(line);
    } catch (err) {
      if (!(err instanceof SyntaxError)) throw err;
      process.stderr.write(`Skipping malformed line: ${err.message}\n`);
      continue;
    }
    if (
      row.stage1.decision === 'allow' ||
      row.stage2 == null ||
      row.stage2.intent !== 'yes' ||
      row.botReplied !== true
    ) {
      continue;
    }
    goldRows++;

    const tokens = tokenize(row.text);
    const seenInRow = new Set<string>();
    for (const n of [1, 2, 3]) {
      for (const g of ngrams(tokens, n)) {
        if (existingSet.has(g)) continue;
        if (seenInRow.has(g)) continue; // count once per message
        seenInRow.add(g);
        const cur = counts.get(g);
        if (!cur) {
          counts.set(g, { count: 1, examples: [row.text] });
        } else {
          cur.count++;
          if (cur.examples.length < 3) cur.examples.push(row.text);
        }
      }
    }
  }

  const ranked = [...counts.entries()]
    .filter(([, v]) => v.count >= args.minFrequency)
    .sort((a, b) => {
      if (b[1].count !== a[1].count) return b[1].count - a[1].count;
      return a[0].localeCompare(b[0]);
    })
    .slice(0, args.topN);

  // Markdown render
  const lines: string[] = [];
  lines.push(`# Trigger keyword candidates`);
  lines.push('');
  lines.push(`- Source: \`${path.basename(inputPath)}\``);
  lines.push(`- Total JSONL rows: ${totalRows}`);
  lines.push(
    `- Gold-cell rows (stage1!=allow, stage2.intent=yes, botReplied=true): ${goldRows}`,
  );
  lines.push(`- Min frequency: ${args.minFrequency}`);
  lines.push(`- Top N: ${args.topN}`);
  if (args.existingPatternsJid) {
    lines.push(
      `- Deduped against existing patterns for jid \`${args.existingPatternsJid}\` (${existingSet.size} entries)`,
    );
  }
  lines.push('');
  lines.push('| rank | n-gram | frequency | example_messages |');
  lines.push('|---:|---|---:|---|');
  ranked.forEach(([gram, v], idx) => {
    const examples = v.examples
      .slice(0, 3)
      .map((e) => {
        const trimmed = e.replace(/\s+/g, ' ').trim();
        return trimmed.length > 80 ? trimmed.slice(0, 77) + '...' : trimmed;
      })
      .map((s) => s.replace(/\|/g, '\\|'))
      .join(' / ');
    lines.push(
      `| ${idx + 1} | \`${gram.replace(/\|/g, '\\|')}\` | ${v.count} | ${examples} |`,
    );
  });
  lines.push('');

  const md = lines.join('\n');
  const outPath = `${inputPath}.candidates.md`;
  fs.writeFileSync(outPath, md);

  process.stdout.write(md);
  process.stderr.write(`\nWrote: ${outPath}\n`);
}

// ESM entry-point guard per `jbaruch/coding-policy: file-hygiene`.
// Without this, importing the module for tests/reuse would execute
// the CLI at import time. `path.resolve` matters — a relative
// `argv[1]` would crash `pathToFileURL`.
if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  // Outer-boundary process contract: an entry-point catch is the only
  // place a CLI script CAN catch every unhandled rejection — if we
  // don't, Node prints the stack trace and exits 1 anyway. Catching
  // here lets us format a single-line FATAL diagnostic for the
  // operator with a full stack still attached. Per
  // `coding-policy: error-handling` the bare catch is allowed
  // exactly here because the next tick is `process.exit` — no
  // recovery, no defect-hiding, just better operator UX on the
  // stderr that's about to terminate the process.
  main().catch((err: unknown) => {
    const e = err instanceof Error ? err : new Error(String(err));
    process.stderr.write(`FATAL: ${e.message}\n${e.stack ?? ''}\n`);
    process.exit(1);
  });
}

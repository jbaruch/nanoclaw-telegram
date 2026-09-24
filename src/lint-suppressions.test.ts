import fs from 'fs';
import path from 'path';
import { describe, expect, it } from 'vitest';

// The needles are assembled at runtime so this file never carries the literal
// directives it inventories.
const ESLINT_DISABLE = ['eslint', 'disable'].join('-');
const NEXT_LINE_DIRECTIVE = `${ESLINT_DISABLE}-next-line`;
const TS_DIRECTIVES = ['ignore', 'nocheck', 'expect-error'].map(
  (directive) => `@ts-${directive}`,
);

const REPO_ROOT = process.cwd();
const SRC_DIR = path.join(REPO_ROOT, 'src');

interface Occurrence {
  file: string;
  line: number;
  text: string;
}

interface SourceFile {
  file: string;
  lines: string[];
}

function listSourceFiles(dir: string): string[] {
  const files: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...listSourceFiles(full));
    } else if (entry.isFile() && /\.(ts|js)$/.test(entry.name)) {
      files.push(full);
    }
  }
  return files.sort();
}

function readSourceFiles(): SourceFile[] {
  return listSourceFiles(SRC_DIR).map((file) => ({
    file: path.relative(REPO_ROOT, file),
    lines: fs.readFileSync(file, 'utf-8').split('\n'),
  }));
}

function findOccurrences(pattern: RegExp): Occurrence[] {
  const hits: Occurrence[] = [];
  for (const { file, lines } of readSourceFiles()) {
    lines.forEach((text, index) => {
      if (pattern.test(text)) {
        hits.push({ file, line: index + 1, text: text.trim() });
      }
    });
  }
  return hits;
}

describe('lint suppression inventory', () => {
  it('has no file-wide, region, or same-line ESLint disables', () => {
    // Everything except the next-line form: the bare file-wide directive,
    // the block-comment region form, and the same-line form.
    expect(
      findOccurrences(new RegExp(`${ESLINT_DISABLE}(?!-next-line)`)),
    ).toEqual([]);
  });

  it('has no TypeScript check suppressions', () => {
    expect(findOccurrences(new RegExp(TS_DIRECTIVES.join('|')))).toEqual([]);
  });

  it('uses no lint suppressions', () => {
    expect(findOccurrences(new RegExp(NEXT_LINE_DIRECTIVE))).toEqual([]);
  });

  it('keeps the ESLint configuration and lint script unweakened', () => {
    const config = fs.readFileSync(
      path.join(REPO_ROOT, 'eslint.config.js'),
      'utf-8',
    );
    expect(config).toContain("{ files: ['src/**/*.{js,ts}'] }");
    expect(config).toContain(
      "{ ignores: ['node_modules/', 'dist/', 'container/', 'groups/'] }",
    );
    expect(config).toContain("'no-catch-all/no-catch-all': 'warn'");
    expect(config).toContain("'@typescript-eslint/no-explicit-any': 'warn'");
    expect(config).toContain(
      "'preserve-caught-error': ['error', { requireCatchParameter: true }]",
    );

    const pkg = JSON.parse(
      fs.readFileSync(path.join(REPO_ROOT, 'package.json'), 'utf-8'),
    ) as { scripts: Record<string, string> };
    expect(pkg.scripts.lint).toBe('eslint src/');

    for (const stray of [
      '.eslintignore',
      '.eslintrc',
      '.eslintrc.json',
      '.eslintrc.js',
      '.eslintrc.cjs',
    ]) {
      expect(fs.existsSync(path.join(REPO_ROOT, stray)), stray).toBe(false);
    }
  });
});

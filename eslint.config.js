import globals from 'globals'
import pluginJs from '@eslint/js'
import tseslint from 'typescript-eslint'
import noCatchAll from 'eslint-plugin-no-catch-all'

export default [
  {
    // container/ is mostly non-TS (skills, shell, python, Dockerfiles);
    // the one TS tree inside it — agent-runner/src — is linted because
    // vitest runs its tests (#733). Its own node_modules/dist stay out.
    ignores: [
      'node_modules/',
      'dist/',
      'groups/',
      'container/agent-runner/node_modules/',
      'container/agent-runner/dist/',
      'container/skills/',
      'container/audible-backup/',
    ],
  },
  {
    files: [
      'src/**/*.{js,ts}',
      'scripts/**/*.{js,ts}',
      'setup/**/*.{js,ts}',
      'container/agent-runner/src/**/*.{js,ts}',
    ],
  },
  { languageOptions: { globals: globals.node } },
  pluginJs.configs.recommended,
  ...tseslint.configs.recommended,
  {
    plugins: { 'no-catch-all': noCatchAll },
    rules: {
      'preserve-caught-error': ['error', { requireCatchParameter: true }],
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          args: 'all',
          argsIgnorePattern: '^_',
          caughtErrors: 'all',
          caughtErrorsIgnorePattern: '^_',
          destructuredArrayIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          ignoreRestSiblings: true,
        },
      ],
      // Error-level gate (#735 burn-down complete: zero findings across the
      // lint scope). A reintroduced catch-all fails CI. Narrow the catch to a
      // specific exception type, or use the documented
      // `outer-boundary-process-contract` carve-out
      // (`jbaruch/coding-policy: error-handling`).
      'no-catch-all/no-catch-all': 'error',
      // Still advisory — the no-explicit-any track is a separate, un-started
      // burn-down; flipping it to error is deferred until those sites clear.
      '@typescript-eslint/no-explicit-any': 'warn',
    },
  },
]

/**
 * ContextStrategy seam for Stage 2 (#83).
 *
 * The Haiku classifier prompt is split into a frozen prefix (the
 * group-agnostic task definition + few-shot examples) and a volatile
 * suffix (per-group context). Strategies produce the volatile suffix.
 *
 * #82's self-improvement loop will plug in a learned-context strategy
 * here without changes to the framework. The framework's contract
 * with strategies is intentionally narrow:
 *
 *   - `name` is the registry key.
 *   - `buildContext(ctx)` returns the suffix string.
 */
import type { GateContext } from './index.js';
import { logger } from '../logger.js';

export interface ContextStrategy {
  readonly name: string;
  buildContext(ctx: GateContext): Promise<string>;
}

const registry: Record<string, ContextStrategy> = {};

export function registerContextStrategy(strategy: ContextStrategy): void {
  if (Object.prototype.hasOwnProperty.call(registry, strategy.name)) {
    throw new Error(`ContextStrategy "${strategy.name}" already registered`);
  }
  registry[strategy.name] = strategy;
}

export function getContextStrategy(name: string): ContextStrategy | undefined {
  return registry[name];
}

export function listContextStrategies(): string[] {
  return Object.keys(registry).sort();
}

/**
 * Test-only: drop a strategy from the registry. Production code
 * never calls this — strategies are registered once at module load
 * and never unregistered.
 */
export function _unregisterContextStrategyForTesting(name: string): void {
  delete registry[name];
}

export const DEFAULT_CONTEXT_STRATEGY = 'static-group-context';

/**
 * Resolve a strategy by name with default fallback. A missing
 * registration logs ERROR and falls back to the default — never
 * crash. If the default itself isn't registered (the
 * static-group-context module didn't load), throws to surface the
 * misconfiguration.
 */
export function resolveContextStrategy(
  name: string | undefined,
  groupFolder: string,
): ContextStrategy {
  const requested = name ?? DEFAULT_CONTEXT_STRATEGY;
  const found = registry[requested];
  if (found) return found;
  if (name) {
    logger.error(
      {
        groupFolder,
        requested: name,
        registered: listContextStrategies(),
      },
      'unknown context strategy — falling back to default',
    );
  }
  const fallback = registry[DEFAULT_CONTEXT_STRATEGY];
  if (!fallback) {
    throw new Error(
      `Default ContextStrategy "${DEFAULT_CONTEXT_STRATEGY}" not registered`,
    );
  }
  return fallback;
}

// Built-in strategy registration. Side-effect import — the registry
// is populated once at module load and stays constant for the
// process lifetime.
import { staticGroupContextStrategy } from './strategies/static-group-context.js';
registerContextStrategy(staticGroupContextStrategy);

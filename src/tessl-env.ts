import fs from 'fs';
import path from 'path';

import { STORE_DIR } from './config.js';
import { isExpectedFsError } from './fs-errors.js';
import { logger } from './logger.js';
import {
  ONECLI_MANAGED_PLACEHOLDER,
  getOneCliOutboundConfig,
  isOneCliConfigured,
} from './onecli-client.js';

/**
 * Filename the OneCLI MITM CA is written to for the tessl child process.
 * `NODE_EXTRA_CA_CERTS` takes a PATH, but `getOneCliOutboundConfig`
 * returns the CA CONTENT, so it has to land on disk somewhere stable.
 * Under `STORE_DIR` rather than `/tmp` so it shares the container's
 * lifetime and is inspectable when a run misbehaves.
 */
const TESSL_CA_BASENAME = 'onecli-tessl-ca.pem';

/**
 * Environment additions that route a `tessl` child process through the
 * OneCLI gateway (#887).
 *
 * Why the registry credential goes through OneCLI at all: tessl's auth
 * lived in a session file under `~/.tessl`, bind-mounted into the
 * orchestrator. That session silently expired on 2026-07-27 and
 * `tessl update` reported `✔ All plugins are up-to-date` while
 * unauthenticated — a stale-tile deploy that only `deploy.sh` step 3c
 * caught. A vaulted credential has no session to decay, and keeps the
 * token out of `.env`, which was deliberately emptied of credentials.
 *
 * Verified end-to-end before this was written:
 * - tessl honors `HTTPS_PROXY` (a dead proxy fails the registry fetch)
 * - tessl trusts a MITM CA via `NODE_EXTRA_CA_CERTS` (a live call
 *   succeeded through the gateway)
 * - tessl authenticates from `TESSL_TOKEN` alone and puts it on the
 *   wire as `Authorization: Bearer <token>` to `api.tessl.io`, so the
 *   gateway has a request to rewrite (before the vault entry existed, a
 *   bogus token returned 401 from the server rather than a client-side
 *   refusal — that is what proved the credential reaches the wire)
 *
 * The placeholder and the proxy env MUST travel together: a placeholder
 * token without the gateway is a dead credential that 401s, and the
 * gateway without the placeholder has nothing to swap. That is why this
 * returns both or neither.
 *
 * Returns `{}` when OneCLI is unconfigured or unreachable — the caller
 * then runs tessl on the direct path with whatever ambient auth exists
 * (the `~/.tessl` session, in a dev checkout). Failing open here is
 * deliberate: a dev machine with no gateway must still be able to run
 * `tessl update`, and a gateway outage in production surfaces as a LOUD
 * tessl failure plus `deploy.sh` step 3c, never as silent staleness.
 */
export async function buildTesslChildEnv(): Promise<NodeJS.ProcessEnv> {
  if (!isOneCliConfigured()) return {};
  let cfg: { proxyUrl: string; ca: string } | null;
  try {
    cfg = await getOneCliOutboundConfig('main');
  } catch (err: unknown) {
    // `getOneCliOutboundConfig` already returns null for a configured-
    // but-unavailable gateway, so the only failure worth absorbing here is
    // a filesystem errno from its CA read (the bundle is written by
    // another component and can be mid-rotation). Anything else — a
    // TypeError, a bug in the client — is not a gateway outage and
    // propagates, per `coding-policy: error-handling` Specific Exceptions.
    if (!isExpectedFsError(err)) throw err;
    logger.warn(
      { err: (err as NodeJS.ErrnoException).code },
      'OneCLI CA read failed while resolving outbound config — running tessl on the direct path with ambient auth',
    );
    return {};
  }
  if (!cfg) {
    logger.warn(
      'OneCLI configured but the gateway returned no outbound config — running tessl on the direct path with ambient auth',
    );
    return {};
  }
  const caPath = path.join(STORE_DIR, TESSL_CA_BASENAME);
  try {
    fs.mkdirSync(STORE_DIR, { recursive: true });
    fs.writeFileSync(caPath, cfg.ca);
  } catch (err: unknown) {
    // Without the CA on disk the MITM leg fails TLS, so a proxied run
    // would break outright — degrade rather than hand tessl a proxy it
    // cannot validate. Narrowed to filesystem errnos (a full disk, a
    // read-only mount); a non-fs Error is a programming defect and
    // propagates rather than being laundered into a silent fallback.
    if (!isExpectedFsError(err)) throw err;
    logger.warn(
      { err: (err as NodeJS.ErrnoException).code, caPath },
      'Could not write the OneCLI CA for tessl — running tessl on the direct path',
    );
    return {};
  }
  return {
    HTTPS_PROXY: cfg.proxyUrl,
    HTTP_PROXY: cfg.proxyUrl,
    NODE_EXTRA_CA_CERTS: caPath,
    // Any non-empty value would do: the gateway's injection config
    // OVERWRITES the `Authorization` header on every api.tessl.io
    // request rather than matching this specific placeholder. The value
    // exists only so the CLI believes it has a credential and issues the
    // request — with `TESSL_TOKEN` unset it refuses client-side and the
    // gateway never sees anything to inject into. The placeholder name
    // is kept for consistency with `ONECLI_MANAGED_VARS`.
    TESSL_TOKEN: ONECLI_MANAGED_PLACEHOLDER,
  };
}

/**
 * Render `buildTesslChildEnv()` as `docker exec -e` arguments so
 * `scripts/deploy.sh` can run its in-container tessl steps through the
 * same path the orchestrator uses. Emitted one `KEY=VALUE` per line;
 * the caller wraps each in `-e`.
 *
 * Printed rather than returned because the consumer is a shell script.
 * The proxy URL carries a gateway credential, so the caller must not
 * echo these lines into build logs.
 *
 * @internal — consumed by scripts/deploy.sh, not part of the API surface.
 */
export async function printTesslChildEnv(): Promise<void> {
  const env = await buildTesslChildEnv();
  for (const [k, v] of Object.entries(env)) {
    if (v !== undefined) process.stdout.write(`${k}=${v}\n`);
  }
}

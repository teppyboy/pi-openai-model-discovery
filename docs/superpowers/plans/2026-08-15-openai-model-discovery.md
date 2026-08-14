# OpenAI-Compatible Model Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a standalone Pi package that discovers `/v1/models` for model-less custom OpenAI-compatible providers listed in `models.json`.

**Architecture:** A single TypeScript extension reads the user JSONC model configuration, registers only providers with `baseUrl` and no static models, and supplies Pi's `refreshModels` callback. The callback restores persisted models, fetches the provider catalog with Pi's resolved credential, maps records into Pi model definitions, and persists successful results. No Pi core changes or runtime dependencies are required.

**Tech Stack:** Pi extension API, Node built-ins (`fs/promises`, `os`, `path`), native `fetch`, Node's built-in test runner, TypeScript executed by Pi/Jiti.

---

### Task 1: Create the package manifest

**Files:**

- Create: `package.json`
- Create: `.gitignore`

- [ ] **Step 1: Add the Pi package manifest**

```json
{
  "name": "pi-openai-model-discovery",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "keywords": ["pi-package"],
  "files": ["extensions", "README.md"],
  "scripts": {
    "test": "node --test --experimental-strip-types test/openai-model-discovery.test.ts",
    "check": "node --experimental-strip-types --check extensions/openai-model-discovery.ts"
  },
  "peerDependencies": {
    "@earendil-works/pi-coding-agent": "*"
  },
  "devDependencies": {
    "@types/node": "^22.0.0"
  },
  "pi": {
    "extensions": ["./extensions/openai-model-discovery.ts"]
  }
}
```

- [ ] **Step 2: Ignore installed dependencies**

Create `.gitignore`:

```gitignore
node_modules/
```

- [ ] **Step 3: Check the manifest parses**

Run: `node -e "JSON.parse(require('fs').readFileSync('package.json', 'utf8')); console.log('package.json OK')"`

Expected: `package.json OK`

### Task 2: Implement dynamic provider discovery

**Files:**

- Create: `extensions/openai-model-discovery.ts`

- [ ] **Step 1: Implement JSONC parsing, provider selection, URL construction, and model mapping**

The module must export pure helpers for the test file and a default Pi extension factory. Use these defaults:

```ts
export const DEFAULT_API = "openai-completions" as const;
export const DEFAULT_CONTEXT_WINDOW = 128_000;
export const DEFAULT_MAX_TOKENS = 16_384;
```

`stripJsonComments()` must preserve quoted strings, remove `//` comments, and remove trailing commas. `readDynamicProviders()` must read `${PI_CODING_AGENT_DIR ?? join(homedir(), ".pi", "agent")}/models.json`, return an empty list for `ENOENT`, and warn/return an empty list for other read or parse errors.

Select entries from `providers` only when `baseUrl` is a non-empty string and `models` is absent or an empty array. Preserve the provider id and configured `api`, defaulting to `openai-completions`.

`mapModelRecord(record, provider)` must require a non-empty string `id` and return a complete Pi provider model definition with these fallbacks:

```ts
{
  name: typeof record.name === "string" && record.name ? record.name : record.id,
  reasoning: firstBoolean(record.reasoning, record.supports_reasoning, record.capabilities?.reasoning) ?? false,
  input: validInput(record.input ?? record.input_types) ?? ["text"],
  contextWindow: positiveNumber(record.contextWindow, record.context_window, record.context_length, record.max_model_len) ?? DEFAULT_CONTEXT_WINDOW,
  maxTokens: positiveNumber(record.maxTokens, record.max_tokens, record.max_output_tokens) ?? DEFAULT_MAX_TOKENS,
  cost: {
    input: numberOrZero(record.cost?.input),
    output: numberOrZero(record.cost?.output),
    cacheRead: numberOrZero(record.cost?.cacheRead ?? record.cost?.cache_read),
    cacheWrite: numberOrZero(record.cost?.cacheWrite ?? record.cost?.cache_write),
  },
}
```

Include `id`, `api`, and `baseUrl` in the returned definition. Normalize input to `text`/`image` entries only and reject records whose id is invalid.

- [ ] **Step 2: Implement refresh and persistence**

The provider callback must follow this flow:

```ts
async function refreshProvider(provider, context) {
  if (!context.allowNetwork) {
    return (context.stored?.models ?? [])
      .filter((model) => model.provider === provider.id)
      .map((model) => toProviderModelDefinition(model, provider));
  }

  const response = await fetch(`${provider.baseUrl.replace(/\/+$/, "")}/models`, {
    signal: context.signal,
    headers: {
      accept: "application/json",
      ...(context.credential?.type === "api_key" && context.credential.key
        ? { authorization: `Bearer ${context.credential.key}` }
        : {}),
    },
  });

  if (!response.ok) {
    throw new Error(`Model discovery failed for ${provider.id}: HTTP ${response.status}`);
  }

  const payload = await response.json();
  if (!Array.isArray(payload?.data)) {
    throw new Error(`Model discovery failed for ${provider.id}: response.data is not an array`);
  }

  const definitions = payload.data.map((record) => mapModelRecord(record, provider));
  await context.publish({
    persist: {
      checkedAt: Date.now(),
      models: definitions.map((model) => ({ ...model, provider: provider.id })),
    },
  });
  return definitions;
}
```

The default extension factory must call `readDynamicProviders()` and register each provider with `baseUrl`, `api`, and `refreshModels`. It must not pass `apiKey`, allowing Pi's existing `models.json`, stored credential, environment, and runtime-key resolution to remain authoritative.

- [ ] **Step 3: Run a type/syntax check**

Run: `node --experimental-strip-types --check extensions/openai-model-discovery.ts`

Expected: exit code 0 and no syntax errors.

### Task 3: Add focused verification

**Files:**

- Create: `test/openai-model-discovery.test.ts`

- [ ] **Step 1: Add built-in test cases**

Use `node:test` and `node:assert/strict`. Test the exported pure functions and the extension factory with a temporary `models.json`:

1. JSONC parsing accepts comments and trailing commas while preserving URL strings.
2. A model-less provider is selected and a provider with a non-empty `models` array is ignored.
3. Mapping uses record metadata and the documented defaults.
4. The registered refresh callback requests `/v1/models` and sends `Authorization: Bearer test-key`.
5. The callback publishes a provider-scoped persisted catalog on success.
6. Cache-only refresh converts persisted full models back into provider definitions.
7. HTTP failure rejects before publishing a replacement catalog.

Mock `globalThis.fetch`, set `PI_CODING_AGENT_DIR` to a temporary directory, and restore both globals in `finally` blocks. Do not contact a live server.

- [ ] **Step 2: Run the focused test**

Run: `npm test`

Expected: all tests pass with zero failures.

### Task 4: Document installation and configuration

**Files:**

- Create: `README.md`

- [ ] **Step 1: Document installation**

Document:

```bash
pi install ./path/to/pi-openai-model-discovery
```

and the alternative temporary load:

```bash
pi -e ./path/to/pi-openai-model-discovery/extensions/openai-model-discovery.ts
```

Explain that the package reads `~/.pi/agent/models.json` (or `PI_CODING_AGENT_DIR`) and discovers providers with `baseUrl` plus no static `models` list.

- [ ] **Step 2: Document the minimal configuration and limitations**

Include:

```json
{
  "providers": {
    "my-server": {
      "baseUrl": "http://localhost:1234/v1",
      "api": "openai-completions",
      "apiKey": "local"
    }
  }
}
```

State that the extension fetches `/models`, uses standard Bearer auth, persists successful catalogs, keeps stale catalogs on refresh failure, leaves explicit model lists untouched, and uses conservative metadata defaults. Mention that local unauthenticated servers still need a harmless configured `apiKey` such as `local` so Pi considers the provider authenticated.

- [ ] **Step 3: Re-run verification**

Run: `npm test`

Expected: all tests pass.

### Task 5: Review, commit, and prepare push

**Files:**

- Modify: none beyond the files above

- [ ] **Step 1: Inspect the final diff**

Run: `git diff --check && git status --short && git diff --stat HEAD`

Expected: no whitespace errors; only `.gitignore`, package files, extension/test/README files, plan, and spec files are present. `node_modules/` is ignored.

- [ ] **Step 2: Run the final test**

Run: `npm test`

Expected: all tests pass.

- [ ] **Step 3: Commit the implementation**

Run:

```bash
git add .gitignore package.json package-lock.json extensions/openai-model-discovery.ts test/openai-model-discovery.test.ts README.md docs/superpowers/plans/2026-08-15-openai-model-discovery.md
git commit -m "feat: discover OpenAI-compatible models"
```

Expected: one new commit on `main`.

- [ ] **Step 4: Push when a remote is configured**

Run:

```bash
git push -u origin main
```

Expected: `main` is pushed to the configured `origin`. If no remote exists, ask the user for the repository URL before running this command.

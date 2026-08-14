# OpenAI-Compatible Model Discovery Extension

## Goal

Provide a standalone Pi package that discovers models from custom OpenAI-compatible providers listed in `models.json`, so those providers do not need a static `models` array.

## Scope

- Read `~/.pi/agent/models.json`, honoring `PI_CODING_AGENT_DIR`.
- Select explicitly configured providers with a `baseUrl` and no models or an empty `models` array.
- Register each selected provider with Pi's extension provider API.
- Fetch `{baseUrl}/models` during Pi's model-catalog refresh flow.
- Use Pi's resolved API-key credential for a standard Bearer header.
- Map OpenAI model records into Pi model definitions.
- Persist successful catalogs and restore them before attempting network refresh.
- Preserve a stale catalog when a later fetch fails.
- Leave providers with explicit model lists unchanged.

Out of scope: non-OpenAI APIs, custom discovery paths or headers, UI changes, modifying Pi core, and automatic inference of provider-specific reasoning or pricing semantics beyond common metadata fields.

## Architecture

The package contains one extension module and a package manifest:

- `package.json` declares the Pi package and its extension entry point.
- `extensions/openai-model-discovery.ts` parses the user model configuration, registers dynamic providers, and owns discovery/mapping/persistence logic.
- `README.md` documents installation, configuration, defaults, and limitations.
- A small mocked-fetch check validates the non-trivial mapping and refresh behavior without network access.

The extension uses the legacy config form of `pi.registerProvider`. Pi's current runtime invokes its `refreshModels` callback during startup and interactive model-catalog refreshes, including `/model`. The callback returns provider model definitions; Pi composes them with the provider configuration and built-in compatibility behavior.

## Provider Selection

The extension reads only the `providers` object. A provider is dynamic when:

- `baseUrl` is a non-empty string; and
- `models` is absent or an empty array.

The configured provider id is preserved. The configured `api` is used when present; otherwise it defaults to `openai-completions`. Existing provider entries with non-empty `models` are not registered by this extension.

The extension passes the configured `baseUrl` and `api` through to Pi. It does not copy or resolve `apiKey`; Pi's provider runtime resolves environment variables, commands, stored credentials, and runtime keys before invoking the network refresh.

## Discovery Flow

1. Load JSONC-compatible `models.json` during extension initialization.
2. Register one dynamic provider for each eligible provider entry.
3. On the cache-only refresh phase, return the persisted provider catalog if one exists.
4. On a network refresh, request `GET {baseUrl}/models` with the refresh abort signal.
5. Require a successful response and an array at `data`; reject malformed records without replacing the current catalog. An empty `data` array produces an empty catalog.
6. Map each valid record by id and publish a provider-scoped persisted catalog with `checkedAt`.
7. Return the mapped definitions to Pi, which makes them available to `/model`.

The URL builder removes trailing slashes and appends `/models`; a base URL ending in `/v1` therefore becomes `/v1/models`.

## Model Mapping

Each discovered record must have a non-empty string `id`. The mapper uses these fields when present:

- `name` or `id` for display name.
- `reasoning`, then `supports_reasoning`, then `capabilities.reasoning`; default `false`.
- `input` or `input_types`; default `["text"]`.
- `contextWindow`, `context_window`, `context_length`, or `max_model_len`; default `128000`.
- `maxTokens`, `max_tokens`, or `max_output_tokens`; default `16384`.
- `cost` fields (`input`, `output`, `cacheRead`, `cacheWrite`) when numeric; otherwise zero.

The mapper normalizes invalid or unsupported values to the documented defaults and does not guess that a model reasons merely from its name.

## Authentication

For a resolved API-key credential, send:

```http
Authorization: Bearer <credential key>
```

Providers using local unauthenticated servers should configure a harmless `apiKey` value such as `local`, matching Pi's existing custom-provider behavior. Providers using `/login` or stored credentials continue to use Pi's credential resolution.

## Failure Behavior

- Missing, unreadable, or malformed `models.json`: the extension reports a warning and registers no dynamic providers.
- Missing `baseUrl`: the provider is skipped.
- HTTP failure, malformed JSON, missing `data`, or invalid model records: the refresh rejects; Pi keeps the previous in-memory/persisted catalog and reports the refresh error.
- Empty successful `data`: treated as a valid empty catalog.
- Abort signals are passed to `fetch` and stop the refresh without replacing the current catalog.

## Verification

Run the package's local mocked-fetch check. It must verify:

1. a provider with no static models is discovered;
2. `/v1/models` is requested with the Bearer credential;
3. model ids and supported metadata are mapped with defaults;
4. a persisted catalog is restored during cache-only refresh; and
5. a failed refresh leaves the prior catalog intact.

No live endpoint or third-party test dependency is required.

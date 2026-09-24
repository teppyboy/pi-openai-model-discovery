# pi-openai-model-discovery

A Pi package that discovers models from custom OpenAI-compatible providers without a static `models` array.

## Install

From this directory:

```bash
pi install .
```

Or load the extension for one run:

```bash
pi -e ./extensions/openai-model-discovery.ts
```

## Configuration

Add providers to `~/.pi/agent/models.json` without listing `models`:

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

The extension also honors `PI_CODING_AGENT_DIR`. Providers with a non-empty `models` array are left unchanged. If `api` is omitted, it defaults to `openai-completions`.

During Pi's model-catalog refresh, the extension requests `GET {baseUrl}/models` (for the example, `/v1/models`) and maps the response into Pi models. A resolved API-key credential is sent as `Authorization: Bearer ...`.

The mapper understands 9router's `context_length`, `max_completion_tokens`, and nested `capabilities` fields. Every discovered model is exposed as reasoning-capable with `xhigh` and `max` mapped directly to the provider. Other levels from explicit `supported_reasoning_levels` metadata are preserved; unknown levels remain unsupported.

## Defaults and failures

The extension uses endpoint metadata when available and conservative defaults otherwise:

- text-only input
- reasoning enabled with `xhigh` and `max`
- zero token cost
- 128,000-token context window
- 16,384 maximum output tokens

Successful non-empty catalogs are persisted in Pi's `models-store.json`. On startup, cached models are registered immediately so model selection (including subagents) works before refresh. Failed or empty refreshes reuse the last successful catalog; first-ever discovery failures still report an error. Invalid or non-HTTP(S) base URLs are rejected.

Local unauthenticated servers should still set a harmless `apiKey` such as `local`, so Pi considers the provider configured. Providers using stored credentials or `/login` continue to use Pi's normal credential resolution.

## Development

```bash
npm test
npm run check
```

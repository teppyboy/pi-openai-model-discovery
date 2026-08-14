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

During Pi's model-catalog refresh, the extension requests `GET {baseUrl}/models` (for the example, `/v1/models`) and maps each `data[].id` into a Pi model. A resolved API-key credential is sent as `Authorization: Bearer ...`.

## Defaults and failures

The extension uses endpoint metadata when available and conservative defaults otherwise:

- text-only input
- non-reasoning
- zero token cost
- 128,000-token context window
- 16,384 maximum output tokens

Successful catalogs are persisted and restored before refresh. If a later request fails, Pi keeps the previous catalog and reports the refresh error. Invalid or non-HTTP(S) base URLs are rejected.

Local unauthenticated servers should still set a harmless `apiKey` such as `local`, so Pi considers the provider configured. Providers using stored credentials or `/login` continue to use Pi's normal credential resolution.

## Development

```bash
npm test
npm run check
```

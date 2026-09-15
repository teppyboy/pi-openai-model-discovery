import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import type {
	ExtensionAPI,
	ProviderConfig,
} from "@earendil-works/pi-coding-agent";
import openAIModelDiscovery, {
	DEFAULT_CONTEXT_WINDOW,
	DEFAULT_MAX_TOKENS,
	mapModelRecord,
	modelDiscoveryUrl,
	readDynamicProviders,
	refreshProvider,
	stripJsonComments,
	type DynamicProvider,
} from "../extensions/openai-model-discovery.ts";

type RefreshContext = Parameters<
	NonNullable<ProviderConfig["refreshModels"]>
>[0];
type Publication = Parameters<RefreshContext["publish"]>[0];

type Registration = {
	name: string;
	config: ProviderConfig;
};

const dynamicProvider = {
	id: "local-server",
	baseUrl: "http://127.0.0.1:1234/v1",
	api: "openai-completions",
} as DynamicProvider;

function context(overrides: Partial<RefreshContext> = {}): RefreshContext {
	return {
		allowNetwork: true,
		signal: new AbortController().signal,
		publish: async () => true,
		...overrides,
	} as RefreshContext;
}

async function writeConfig(
	content: string,
): Promise<{ directory: string; filePath: string }> {
	const directory = await mkdtemp(join(tmpdir(), "pi-openai-discovery-"));
	const filePath = join(directory, "models.json");
	await writeFile(filePath, content, "utf8");
	return { directory, filePath };
}

test("stripJsonComments preserves URLs and removes comments/trailing commas", () => {
	const parsed = JSON.parse(
		stripJsonComments(`{
		"url": "http://localhost:1234/v1//models",
		// comment
		"value": 1,
	}`),
	);

	assert.equal(parsed.url, "http://localhost:1234/v1//models");
	assert.equal(parsed.value, 1);
});

test("readDynamicProviders selects only providers without static models", async () => {
	const { directory, filePath } = await writeConfig(`{
		"providers": {
			"dynamic": {
				"baseUrl": "http://localhost:1234/v1",
			},
			"empty": {
				"baseUrl": "https://example.test/v1",
				"models": [],
			},
			"static": {
				"baseUrl": "https://example.test/v1",
				"models": [{"id": "fixed"}],
			},
			"missing-url": {"api": "openai-completions"},
		}
	}`);

	try {
		assert.deepEqual(await readDynamicProviders(filePath), [
			{
				id: "dynamic",
				baseUrl: "http://localhost:1234/v1",
				api: "openai-completions",
			},
			{
				id: "empty",
				baseUrl: "https://example.test/v1",
				api: "openai-completions",
			},
		]);
	} finally {
		await rm(directory, { recursive: true, force: true });
	}
});

test("mapModelRecord uses metadata and conservative defaults", () => {
	assert.deepEqual(
		mapModelRecord(
			{
				id: "vision-model",
				name: "Vision Model",
				supports_reasoning: true,
				input_types: ["text", "image", "audio"],
				context_length: 32_000,
				max_output_tokens: 2_000,
				cost: { input: 1, output: 2, cache_read: 0.1, cache_write: 0.2 },
			},
			dynamicProvider,
		),
		{
			id: "vision-model",
			name: "Vision Model",
			api: "openai-completions",
			baseUrl: dynamicProvider.baseUrl,
			reasoning: true,
			thinkingLevelMap: { xhigh: "xhigh", max: "max" },
			input: ["text", "image"],
			cost: { input: 1, output: 2, cacheRead: 0.1, cacheWrite: 0.2 },
			contextWindow: 32_000,
			maxTokens: 2_000,
		},
	);

	const defaults = mapModelRecord({ id: "basic" }, dynamicProvider);
	assert.equal(defaults.reasoning, true);
	assert.deepEqual(defaults.thinkingLevelMap, {
		xhigh: "xhigh",
		max: "max",
	});
	assert.deepEqual(defaults.input, ["text"]);
	assert.equal(defaults.contextWindow, DEFAULT_CONTEXT_WINDOW);
	assert.equal(defaults.maxTokens, DEFAULT_MAX_TOKENS);
});

test("maps 9router and CLIProxyAPI capability metadata", () => {
	const nineRouter = mapModelRecord(
		{
			id: "gpt-5.6-sol",
			capabilities: { vision: true, reasoning: true },
			context_length: 1_000_000,
			max_completion_tokens: 128_000,
		},
		dynamicProvider,
	);
	assert.equal(nineRouter.reasoning, true);
	assert.deepEqual(nineRouter.input, ["text", "image"]);
	assert.equal(nineRouter.contextWindow, 1_000_000);
	assert.equal(nineRouter.maxTokens, 128_000);
	assert.deepEqual(nineRouter.thinkingLevelMap, {
		xhigh: "xhigh",
		max: "max",
	});

	const bareNineRouter = mapModelRecord({ id: "gpt-5.6-sol" }, dynamicProvider);
	assert.equal(bareNineRouter.reasoning, true);
	assert.deepEqual(bareNineRouter.thinkingLevelMap, {
		xhigh: "xhigh",
		max: "max",
	});

	const cliProxyApi = mapModelRecord(
		{
			slug: "gpt-5.6-sol",
			display_name: "GPT-5.6 Sol",
			input_modalities: ["text", "image"],
			context_window: 372_000,
			supported_reasoning_levels: [
				{ effort: "low" },
				{ effort: "medium" },
				{ effort: "high" },
				{ effort: "xhigh" },
				{ effort: "max" },
			],
		},
		dynamicProvider,
	);
	assert.equal(cliProxyApi.id, "gpt-5.6-sol");
	assert.equal(cliProxyApi.name, "GPT-5.6 Sol");
	assert.equal(cliProxyApi.reasoning, true);
	assert.deepEqual(cliProxyApi.input, ["text", "image"]);
	assert.equal(cliProxyApi.contextWindow, 372_000);
	assert.deepEqual(cliProxyApi.thinkingLevelMap, {
		xhigh: "xhigh",
		max: "max",
		low: "low",
		medium: "medium",
		high: "high",
		minimal: "low",
	});
});

test("modelDiscoveryUrl validates schemes and appends /models", () => {
	assert.equal(
		modelDiscoveryUrl("https://example.test/v1/").toString(),
		"https://example.test/v1/models",
	);
	assert.throws(() => modelDiscoveryUrl("file:///tmp/models"), /HTTP\(S\)/);
	assert.throws(() => modelDiscoveryUrl("not a URL"), /valid HTTP/);
});

test("registered providers discover models with Bearer auth and persist them", async () => {
	const originalFetch = globalThis.fetch;
	const originalDirectory = process.env.PI_CODING_AGENT_DIR;
	const { directory } = await writeConfig(`{
		"providers": {
			"local-server": {
				"baseUrl": "http://127.0.0.1:1234/v1",
				"apiKey": "test-key",
			}
		}
	}`);
	process.env.PI_CODING_AGENT_DIR = directory;

	let requestUrl = "";
	let requestHeaders: HeadersInit | undefined;
	globalThis.fetch = async (input, init) => {
		requestUrl = String(input);
		requestHeaders = init?.headers;
		return new Response(
			JSON.stringify({
				data: [
					{
						id: "server-model",
						context_length: 128_000,
						max_completion_tokens: 16_384,
					},
				],
			}),
			{
				status: 200,
				headers: { "content-type": "application/json" },
			},
		);
	};

	const registrations: Registration[] = [];
	const pi = {
		registerProvider(name: string, config: ProviderConfig) {
			registrations.push({ name, config });
		},
	} as unknown as ExtensionAPI;
	const published: Publication[] = [];

	try {
		await openAIModelDiscovery(pi);
		assert.equal(registrations.length, 1);
		assert.equal(registrations[0]?.name, "local-server");

		const refresh = registrations[0]?.config.refreshModels;
		assert.ok(refresh);
		const models = await refresh(
			context({
				credential: { type: "api_key", key: "test-key" },
				publish: async (publication) => {
					published.push(publication);
					return true;
				},
			}),
		);

		assert.equal(requestUrl, "http://127.0.0.1:1234/v1/models");
		assert.equal(
			(requestHeaders as Record<string, string>).Authorization,
			"Bearer test-key",
		);
		assert.deepEqual(
			models.map((model) => model.id),
			["server-model"],
		);
		const persisted = published[0]?.persist;
		assert.ok(persisted);
		assert.equal(persisted.models[0]?.provider, "local-server");
		assert.equal(persisted.models[0]?.id, "server-model");

		const restored = await refresh(
			context({
				allowNetwork: false,
				stored: persisted,
			}),
		);
		assert.deepEqual(
			restored.map((model) => model.id),
			["server-model"],
		);
	} finally {
		globalThis.fetch = originalFetch;
		if (originalDirectory === undefined) delete process.env.PI_CODING_AGENT_DIR;
		else process.env.PI_CODING_AGENT_DIR = originalDirectory;
		await rm(directory, { recursive: true, force: true });
	}
});

test("uses CLIProxyAPI's rich catalog when /v1/models is skeletal", async () => {
	const originalFetch = globalThis.fetch;
	const requestedUrls: string[] = [];
	globalThis.fetch = async (input) => {
		requestedUrls.push(String(input));
		if (requestedUrls.length === 1) {
			return new Response(JSON.stringify({ data: [{ id: "gpt-5.6-sol" }] }), {
				status: 200,
			});
		}
		return new Response(
			JSON.stringify({
				models: [
					{
						slug: "gpt-5.6-sol",
						display_name: "GPT-5.6 Sol",
						context_window: 372_000,
						input_modalities: ["text", "image"],
						supported_reasoning_levels: [
							{ effort: "low" },
							{ effort: "medium" },
							{ effort: "high" },
							{ effort: "xhigh" },
							{ effort: "max" },
						],
					},
				],
			}),
			{ status: 200 },
		);
	};

	try {
		const models = await refreshProvider(
			dynamicProvider,
			context({ publish: async () => true }),
		);
		assert.deepEqual(
			models.map((model) => model.id),
			["gpt-5.6-sol"],
		);
		assert.equal(models[0]?.contextWindow, 372_000);
		assert.deepEqual(models[0]?.input, ["text", "image"]);
		assert.deepEqual(models[0]?.thinkingLevelMap, {
			xhigh: "xhigh",
			max: "max",
			low: "low",
			medium: "medium",
			high: "high",
			minimal: "low",
		});
		assert.equal(
			new URL(requestedUrls[1] ?? "").searchParams.get("client_version"),
			"pi",
		);
	} finally {
		globalThis.fetch = originalFetch;
	}
});

test("failed discovery does not publish a replacement catalog", async () => {
	const originalFetch = globalThis.fetch;
	const { directory, filePath } = await writeConfig(`{
		"providers": {
			"local-server": {"baseUrl": "http://127.0.0.1:1234/v1"}
		}
	}`);
	let published = 0;
	globalThis.fetch = async () =>
		new Response("upstream unavailable", { status: 503 });

	try {
		const [provider] = await readDynamicProviders(filePath);
		assert.ok(provider);
		await assert.rejects(() =>
			refreshProvider(
				provider,
				context({
					publish: async () => {
						published++;
						return true;
					},
				}),
			),
		);
		assert.equal(published, 0);
	} finally {
		globalThis.fetch = originalFetch;
		await rm(directory, { recursive: true, force: true });
	}
});

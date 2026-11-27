import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import type {
	ExtensionAPI,
	ProviderConfig,
} from "@earendil-works/pi-coding-agent";
import openAIModelDiscovery, {
	agentDir,
	applyModelOverride,
	buildDiscoveryHeaders,
	clearConfigValueCache,
	clearPricingMemoryCache,
	compilePatterns,
	DEFAULT_CONTEXT_WINDOW,
	DEFAULT_EXCLUDE_PATTERNS,
	DEFAULT_INCLUDE_PATTERNS,
	DEFAULT_MAX_TOKENS,
	DEFAULT_PRICING_URL,
	fetchPricingCatalog,
	isLoopbackUrl,
	lookupPricingEntry,
	mapModelRecord,
	mergeCompat,
	modelDiscoveryUrl,
	pricingToCost,
	pricingToInput,
	pricingToThinking,
	readDiscoveryConfig,
	readDynamicProviders,
	refreshProvider,
	resolveConfigValue,
	shouldIncludeModel,
	stripJsonComments,
	type DynamicProvider,
	type PricingCatalog,
	type PricingEntry,
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

async function captureStderr(
	fn: () => Promise<void> | void,
): Promise<string> {
	const chunks: string[] = [];
	const originalWrite = process.stderr.write.bind(process.stderr);
	process.stderr.write = ((
		chunk: string | Uint8Array,
		encoding?: BufferEncoding | ((err?: Error | null) => void),
		callback?: (err?: Error | null) => void,
	) => {
		chunks.push(
			typeof chunk === "string" ? chunk : Buffer.from(chunk).toString("utf8"),
		);
		if (typeof encoding === "function") encoding(null);
		else if (typeof callback === "function") callback(null);
		return true;
	}) as typeof process.stderr.write;

	try {
		await fn();
	} finally {
		process.stderr.write = originalWrite;
	}
	return chunks.join("");
}

function createFakePi(): {
	pi: ExtensionAPI;
	registrations: Registration[];
} {
	const registrations: Registration[] = [];
	const pi = {
		registerProvider(name: string, config: ProviderConfig) {
			registrations.push({ name, config });
		},
	} as unknown as ExtensionAPI;
	return { pi, registrations };
}

function headerRecord(
	headers: HeadersInit | undefined,
): Record<string, string> {
	if (!headers) return {};
	if (headers instanceof Headers) {
		return Object.fromEntries(headers.entries());
	}
	if (Array.isArray(headers)) {
		return Object.fromEntries(headers);
	}
	return { ...headers };
}

function findHeaderIgnoreCase(
	headers: HeadersInit | undefined,
	name: string,
): string | undefined {
	const lower = name.toLowerCase();
	for (const [key, value] of Object.entries(headerRecord(headers))) {
		if (key.toLowerCase() === lower) return value;
	}
	return undefined;
}

function jsonResponse(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		status,
		headers: { "content-type": "application/json" },
	});
}

// =============================================================================
// Existing tests (9) — kept intact and 100% offline-safe
// =============================================================================

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
	clearPricingMemoryCache();
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

	clearPricingMemoryCache();
	const capturedRequests: Array<{ url: string; headers?: HeadersInit }> = [];
	globalThis.fetch = async (input, init) => {
		const url = String(input);
		capturedRequests.push({ url, headers: init?.headers });
		if (url === DEFAULT_PRICING_URL) {
			return new Response(JSON.stringify({}), {
				status: 200,
				headers: { "content-type": "application/json" },
			});
		}
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

		capturedRequests.length = 0;
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

		const modelRequest = capturedRequests.find(
			(req) => req.url === "http://127.0.0.1:1234/v1/models",
		);
		assert.ok(modelRequest);
		assert.equal(
			(modelRequest.headers as Record<string, string>).Authorization,
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
		clearPricingMemoryCache();
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

test("cached catalog survives startup and failed or empty refresh", async () => {
	clearPricingMemoryCache();
	const originalFetch = globalThis.fetch;
	const originalDirectory = process.env.PI_CODING_AGENT_DIR;
	const { directory } = await writeConfig(`{
		"providers": {
			"local-server": {"baseUrl": "http://127.0.0.1:1234/v1"}
		}
	}`);
	process.env.PI_CODING_AGENT_DIR = directory;
	const saved = {
		models: [
			{ id: "gpt-5.6-luna", provider: "local-server", contextWindow: 372_000 },
			{ id: "foreign", provider: "other" },
		],
	} as unknown as RefreshContext["stored"];
	await writeFile(
		join(directory, "models-store.json"),
		JSON.stringify({ "local-server": saved }),
	);
	globalThis.fetch = async () =>
		new Response("upstream unavailable", { status: 503 });
	const registrations: Registration[] = [];
	try {
		await captureStderr(async () => {
			await openAIModelDiscovery({
				registerProvider(name: string, config: ProviderConfig) {
					registrations.push({ name, config });
				},
			} as unknown as ExtensionAPI);
		});
		const registered = registrations[0]?.config;
		assert.deepEqual(
			registered?.models?.map((model) => model.id),
			["gpt-5.6-luna"],
		);
		assert.equal(registered?.models?.[0]?.contextWindow, 372_000);
		const refresh = registered?.refreshModels;
		assert.ok(refresh);
		assert.deepEqual(
			(await refresh(context({ allowNetwork: false, stored: saved }))).map(
				(model) => model.id,
			),
			["gpt-5.6-luna"],
		);
		let published = 0;
		const failedContext = context({
			stored: saved,
			publish: async () => {
				published++;
				return true;
			},
		});
		globalThis.fetch = async () =>
			new Response("upstream unavailable", { status: 503 });
		let fallback: Awaited<ReturnType<typeof refresh>> = [];
		await captureStderr(async () => {
			fallback = await refresh(failedContext);
		});
		assert.deepEqual(
			fallback.map((model) => model.id),
			["gpt-5.6-luna"],
		);
		globalThis.fetch = async () => new Response(JSON.stringify({ data: [] }));
		await captureStderr(async () => {
			assert.deepEqual(await refresh(failedContext), fallback);
		});
		assert.equal(published, 0);
	} finally {
		clearPricingMemoryCache();
		globalThis.fetch = originalFetch;
		if (originalDirectory === undefined) delete process.env.PI_CODING_AGENT_DIR;
		else process.env.PI_CODING_AGENT_DIR = originalDirectory;
		await rm(directory, { recursive: true, force: true });
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

// =============================================================================
// Cold-start local API discovery (cliproxyapi) & cache-phase persistence
// =============================================================================

test("cold-start cliproxyapi discovery populates models immediately, survives cache-phase refresh, and persists once", async () => {
	clearPricingMemoryCache();
	const originalFetch = globalThis.fetch;
	const originalDirectory = process.env.PI_CODING_AGENT_DIR;
	const { directory } = await writeConfig(`{
		"providers": {
			"cliproxyapi": {
				"baseUrl": "http://127.0.0.1:8317/v1",
				"apiKey": "cliproxy-secret"
			}
		}
	}`);
	await writeFile(
		join(directory, "pi-openai-model-discovery.json"),
		JSON.stringify({
			pricingUrl: "https://pricing.example.test/prices.json",
		}),
		"utf8",
	);
	process.env.PI_CODING_AGENT_DIR = directory;

	const fetchedRequests: Array<{ url: string; headers: HeadersInit | undefined }> =
		[];
	globalThis.fetch = async (input, init) => {
		const url = String(input);
		fetchedRequests.push({ url, headers: init?.headers });

		if (url === "https://pricing.example.test/prices.json") {
			return jsonResponse({
				"gpt-5.6-sol": {
					max_input_tokens: 200_000,
					max_output_tokens: 64_000,
					input_cost_per_token: 0.0000025,
					output_cost_per_token: 0.000015,
					cache_read_input_token_cost: 0.00000025,
					supports_reasoning: true,
					supports_xhigh_reasoning_effort: true,
					supports_vision: true,
				},
				"claude-sonnet-4": {
					max_input_tokens: 200_000,
					max_output_tokens: 64_000,
					input_cost_per_token: 0.000003,
					output_cost_per_token: 0.000015,
					supports_vision: true,
					supports_reasoning: true,
				},
			});
		}
		if (url === "http://127.0.0.1:8317/v1/models") {
			return jsonResponse({
				data: [{ id: "gpt-5.6-sol" }, { id: "claude-sonnet-4-20250514" }],
			});
		}
		if (url === "http://127.0.0.1:8317/v1/models?client_version=pi") {
			return jsonResponse({
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
						],
					},
					{
						slug: "claude-sonnet-4-20250514",
						display_name: "Claude Sonnet 4",
					},
				],
			});
		}
		throw new Error(`Unexpected URL: ${url}`);
	};

	const { pi, registrations } = createFakePi();

	try {
		await openAIModelDiscovery(pi);

		assert.equal(registrations.length, 1);
		const reg = registrations[0];
		assert.ok(reg);
		assert.equal(reg.name, "cliproxyapi");
		assert.equal(reg.config.apiKey, undefined);

		// Startup immediately fetched /v1/models, /v1/models?client_version=pi, and pricingUrl
		const requestedUrls = fetchedRequests.map((r) => r.url);
		assert.ok(requestedUrls.includes("http://127.0.0.1:8317/v1/models"));
		assert.ok(
			requestedUrls.includes(
				"http://127.0.0.1:8317/v1/models?client_version=pi",
			),
		);
		assert.ok(
			requestedUrls.includes("https://pricing.example.test/prices.json"),
		);

		// Pricing request sent NO headers; discovery requests sent Bearer auth
		const pricingReq = fetchedRequests.find(
			(r) => r.url === "https://pricing.example.test/prices.json",
		);
		assert.equal(pricingReq?.headers, undefined);
		const discoveryReq = fetchedRequests.find(
			(r) => r.url === "http://127.0.0.1:8317/v1/models",
		);
		assert.equal(
			findHeaderIgnoreCase(discoveryReq?.headers, "authorization"),
			"Bearer cliproxy-secret",
		);

		// Cold-start models are immediately registered for CLI modes (--list-models, -p, --model)
		const initialModels = reg.config.models;
		assert.ok(initialModels);
		assert.deepEqual(
			initialModels.map((m) => m.id),
			["gpt-5.6-sol", "claude-sonnet-4-20250514"],
		);
		assert.equal(initialModels[0]?.contextWindow, 372_000);
		assert.equal(initialModels[0]?.maxTokens, 64_000);
		assert.deepEqual(initialModels[0]?.cost, {
			input: 2.5,
			output: 15,
			cacheRead: 0.25,
			cacheWrite: 0,
		});
		assert.equal(initialModels[1]?.contextWindow, 200_000);
		assert.deepEqual(initialModels[1]?.cost, {
			input: 3,
			output: 15,
			cacheRead: 0,
			cacheWrite: 0,
		});

		// Pi's createAgentSessionServices calls refresh({ allowNetwork: false, stored: undefined })
		// right after registerProvider: it must NOT wipe models with [] and MUST persist them
		const refresh = reg.config.refreshModels;
		assert.ok(refresh);
		const publications: Publication[] = [];
		const cachePhaseModels = await refresh(
			context({
				allowNetwork: false,
				stored: undefined,
				publish: async (publication) => {
					publications.push(publication);
					return true;
				},
			}),
		);
		assert.deepEqual(
			cachePhaseModels.map((m) => m.id),
			["gpt-5.6-sol", "claude-sonnet-4-20250514"],
		);
		assert.equal(publications.length, 1);
		const persisted = publications[0]?.persist;
		assert.ok(persisted);
		assert.equal(persisted.models.length, 2);
		assert.equal(persisted.models[0]?.provider, "cliproxyapi");
		assert.equal(persisted.models[0]?.contextWindow, 372_000);

		// Subsequent allowNetwork: false with stored: persisted restores cleanly WITHOUT publishing again
		const restored = await refresh(
			context({
				allowNetwork: false,
				stored: persisted,
				publish: async (publication) => {
					publications.push(publication);
					return true;
				},
			}),
		);
		assert.equal(publications.length, 1);
		assert.deepEqual(
			restored.map((m) => m.id),
			["gpt-5.6-sol", "claude-sonnet-4-20250514"],
		);
		assert.equal(restored[0]?.contextWindow, 372_000);
		assert.deepEqual(restored[0]?.cost, initialModels[0]?.cost);
	} finally {
		clearPricingMemoryCache();
		globalThis.fetch = originalFetch;
		if (originalDirectory === undefined) delete process.env.PI_CODING_AGENT_DIR;
		else process.env.PI_CODING_AGENT_DIR = originalDirectory;
		await rm(directory, { recursive: true, force: true });
	}
});

// =============================================================================
// Keyless loopback server discovery & synthetic "local" key
// =============================================================================

test("keyless loopback servers omit Authorization on startup/refresh and receive synthetic apiKey='local', while non-loopback does not", async () => {
	assert.equal(isLoopbackUrl("http://127.0.0.1:11434/v1"), true);
	assert.equal(isLoopbackUrl("http://127.0.0.42:8080/v1"), true);
	assert.equal(isLoopbackUrl("http://localhost:1234/v1"), true);
	assert.equal(isLoopbackUrl("http://[::1]:8080/v1"), true);
	assert.equal(isLoopbackUrl("https://example.test/v1"), false);
	assert.equal(isLoopbackUrl("http://192.168.1.5:11434/v1"), false);
	assert.equal(isLoopbackUrl("not-a-url"), false);

	clearPricingMemoryCache();
	const originalFetch = globalThis.fetch;
	const originalDirectory = process.env.PI_CODING_AGENT_DIR;
	const { directory } = await writeConfig(`{
		"providers": {
			"ollama": { "baseUrl": "http://127.0.0.1:11434/v1" },
			"lmstudio": { "baseUrl": "http://localhost:1234/v1" },
			"remote": { "baseUrl": "https://example.test/v1" }
		}
	}`);
	await writeFile(
		join(directory, "pi-openai-model-discovery.json"),
		JSON.stringify({ pricingUrl: null }),
		"utf8",
	);
	process.env.PI_CODING_AGENT_DIR = directory;

	const requestsByUrl = new Map<string, Array<HeadersInit | undefined>>();
	globalThis.fetch = async (input, init) => {
		const url = String(input);
		const list = requestsByUrl.get(url) ?? [];
		list.push(init?.headers);
		requestsByUrl.set(url, list);

		if (url === "http://127.0.0.1:11434/v1/models") {
			return jsonResponse({
				data: [{ id: "llama3.3", context_length: 131_072 }],
			});
		}
		if (url === "http://localhost:1234/v1/models") {
			return jsonResponse({
				data: [{ id: "qwen2.5-coder", context_length: 65_536 }],
			});
		}
		if (url === "https://example.test/v1/models") {
			return jsonResponse({
				data: [{ id: "remote-model", context_length: 128_000 }],
			});
		}
		throw new Error(`Unexpected URL: ${url}`);
	};

	const { pi, registrations } = createFakePi();

	try {
		await openAIModelDiscovery(pi);
		assert.equal(registrations.length, 3);

		const ollama = registrations.find((r) => r.name === "ollama");
		const lmstudio = registrations.find((r) => r.name === "lmstudio");
		const remote = registrations.find((r) => r.name === "remote");
		assert.ok(ollama && lmstudio && remote);

		// Loopback servers receive synthetic apiKey: "local"; remote does NOT
		assert.equal(ollama.config.apiKey, "local");
		assert.equal(lmstudio.config.apiKey, "local");
		assert.equal(remote.config.apiKey, undefined);

		// Startup discovery sent NO Authorization header for keyless servers
		const ollamaStartupHeaders = requestsByUrl.get(
			"http://127.0.0.1:11434/v1/models",
		)?.[0];
		const lmstudioStartupHeaders = requestsByUrl.get(
			"http://localhost:1234/v1/models",
		)?.[0];
		assert.equal(
			findHeaderIgnoreCase(ollamaStartupHeaders, "authorization"),
			undefined,
		);
		assert.equal(
			findHeaderIgnoreCase(lmstudioStartupHeaders, "authorization"),
			undefined,
		);

		// Network refresh with synthetic credential { type: "api_key", key: "local" } omits Authorization
		const ollamaRefresh = ollama.config.refreshModels;
		assert.ok(ollamaRefresh);
		await ollamaRefresh(
			context({
				allowNetwork: true,
				credential: { type: "api_key", key: "local" },
			}),
		);
		const ollamaRefreshHeaders = requestsByUrl.get(
			"http://127.0.0.1:11434/v1/models",
		)?.[1];
		assert.equal(
			findHeaderIgnoreCase(ollamaRefreshHeaders, "authorization"),
			undefined,
		);

		// Providing a real (non-synthetic) key still sends Authorization
		await ollamaRefresh(
			context({
				allowNetwork: true,
				credential: { type: "api_key", key: "real-user-key" },
			}),
		);
		const ollamaRealKeyHeaders = requestsByUrl.get(
			"http://127.0.0.1:11434/v1/models",
		)?.[2];
		assert.equal(
			findHeaderIgnoreCase(ollamaRealKeyHeaders, "authorization"),
			"Bearer real-user-key",
		);
	} finally {
		clearPricingMemoryCache();
		globalThis.fetch = originalFetch;
		if (originalDirectory === undefined) delete process.env.PI_CODING_AGENT_DIR;
		else process.env.PI_CODING_AGENT_DIR = originalDirectory;
		await rm(directory, { recursive: true, force: true });
	}
});

// =============================================================================
// Config value resolution, custom headers, and authHeader: false
// =============================================================================

test("resolveConfigValue handles env vars, escapes, unset vars, and cross-platform !command caching", async () => {
	clearConfigValueCache();
	const env = {
		API_TOKEN: "secret-123",
		TENANT: "acme",
	};

	try {
		assert.equal(resolveConfigValue("literal-value", env), "literal-value");
		assert.equal(resolveConfigValue("$API_TOKEN", env), "secret-123");
		assert.equal(resolveConfigValue("${API_TOKEN}", env), "secret-123");
		assert.equal(
			resolveConfigValue("prefix-${TENANT}-$API_TOKEN-suffix", env),
			"prefix-acme-secret-123-suffix",
		);
		assert.equal(resolveConfigValue("$$API_TOKEN", env), "$API_TOKEN");
		assert.equal(resolveConfigValue("$!not-a-command", env), "!not-a-command");
		assert.equal(
			resolveConfigValue("cost=$$5 and $!bang", env),
			"cost=$5 and !bang",
		);
		assert.equal(
			resolveConfigValue("$DEFINITELY_UNSET_VAR_PI_TEST_XYZ", {}),
			undefined,
		);
		assert.equal(
			resolveConfigValue("pre-${DEFINITELY_UNSET_VAR_PI_TEST_XYZ}", {}),
			undefined,
		);

		// Cross-platform !command execution
		assert.equal(
			resolveConfigValue(`!node -e "console.log('cmd-secret')"`),
			"cmd-secret",
		);
		assert.equal(resolveConfigValue("!"), undefined);
		assert.equal(resolveConfigValue("!   "), undefined);
		assert.equal(
			resolveConfigValue(`!node -e "process.exit(1)"`),
			undefined,
		);

		// Command result caching & clearConfigValueCache()
		const tempDir = await mkdtemp(join(tmpdir(), "pi-cmd-cache-"));
		const counterFile = join(tempDir, "counter.txt");
		const scriptFile = join(tempDir, "counter.cjs");
		await writeFile(
			scriptFile,
			`const fs = require("node:fs");
const p = ${JSON.stringify(counterFile)};
const n = (fs.existsSync(p) ? Number(fs.readFileSync(p, "utf8")) : 0) + 1;
fs.writeFileSync(p, String(n));
console.log("run-" + n);
`,
			"utf8",
		);
		const command = `!node ${JSON.stringify(scriptFile)}`;

		try {
			const first = resolveConfigValue(command);
			const second = resolveConfigValue(command);
			assert.equal(first, "run-1");
			assert.equal(second, "run-1");
			assert.equal(await readFile(counterFile, "utf8"), "1");

			clearConfigValueCache();
			const third = resolveConfigValue(command);
			assert.equal(third, "run-2");
			assert.equal(await readFile(counterFile, "utf8"), "2");
		} finally {
			await rm(tempDir, { recursive: true, force: true });
		}
	} finally {
		clearConfigValueCache();
	}
});

test("buildDiscoveryHeaders and discovery pipeline forward custom headers, override Accept case-insensitively, respect explicit Authorization, and honor authHeader: false", async () => {
	clearPricingMemoryCache();
	clearConfigValueCache();

	// Unit checks on buildDiscoveryHeaders
	const customAcceptHeaders = buildDiscoveryHeaders(
		{
			id: "p1",
			baseUrl: "https://api.example.test/v1",
			api: "openai-completions",
			apiKey: "$MY_KEY",
			headers: {
				Accept: "application/vnd.custom+json",
				"X-Tenant": "${MY_TENANT}",
				"X-Unset": "$NO_SUCH_VAR_PI_TEST",
			},
		},
		{ env: { MY_KEY: "k-1", MY_TENANT: "t-1" } },
	);
	assert.deepEqual(customAcceptHeaders, {
		Accept: "application/vnd.custom+json",
		"X-Tenant": "t-1",
		Authorization: "Bearer k-1",
	});
	assert.equal("accept" in customAcceptHeaders, false);

	// Explicit Authorization header is NOT overwritten by apiKey or credentialKey
	const explicitAuthHeaders = buildDiscoveryHeaders(
		{
			id: "p2",
			baseUrl: "https://api.example.test/v1",
			api: "openai-completions",
			apiKey: "ignored-key",
			headers: {
				authorization: "Token explicit-token",
			},
		},
		{ credentialKey: "ignored-cred" },
	);
	assert.equal(explicitAuthHeaders.authorization, "Token explicit-token");
	assert.equal("Authorization" in explicitAuthHeaders, false);

	// authHeader: false suppresses Authorization: Bearer
	const suppressedAuthHeaders = buildDiscoveryHeaders(
		{
			id: "p3",
			baseUrl: "https://api.example.test/v1",
			api: "openai-completions",
			apiKey: "secret-key",
			authHeader: false,
			headers: {
				"X-Api-Key": "$X_KEY",
			},
		},
		{ credentialKey: "cred-key", env: { X_KEY: "header-key" } },
	);
	assert.deepEqual(suppressedAuthHeaders, {
		accept: "application/json",
		"X-Api-Key": "header-key",
	});

	// Integration check: forwarded on BOTH startup and refreshProvider
	const originalFetch = globalThis.fetch;
	const originalDirectory = process.env.PI_CODING_AGENT_DIR;
	const originalCustomEnv = process.env.PI_TEST_CUSTOM_HEADER;
	process.env.PI_TEST_CUSTOM_HEADER = "env-header-value";

	const { directory } = await writeConfig(`{
		"providers": {
			"no-auth-bearer": {
				"baseUrl": "https://gateway.example.test/v1",
				"apiKey": "should-not-be-bearer",
				"authHeader": false,
				"headers": {
					"Accept": "application/vnd.gateway+json",
					"X-Custom-Token": "$PI_TEST_CUSTOM_HEADER"
				}
			},
			"explicit-auth": {
				"baseUrl": "https://explicit.example.test/v1",
				"apiKey": "should-not-overwrite",
				"headers": {
					"Authorization": "ApiKey custom-auth-scheme"
				}
			}
		}
	}`);
	await writeFile(
		join(directory, "pi-openai-model-discovery.json"),
		JSON.stringify({ pricingUrl: null }),
		"utf8",
	);
	process.env.PI_CODING_AGENT_DIR = directory;

	const seenHeaders = new Map<string, Array<Record<string, string>>>();
	globalThis.fetch = async (input, init) => {
		const url = String(input);
		const list = seenHeaders.get(url) ?? [];
		list.push(headerRecord(init?.headers));
		seenHeaders.set(url, list);
		return jsonResponse({
			data: [{ id: "gateway-model", context_length: 128_000 }],
		});
	};

	const { pi, registrations } = createFakePi();

	try {
		await openAIModelDiscovery(pi);
		assert.equal(registrations.length, 2);

		const noAuth = registrations.find((r) => r.name === "no-auth-bearer");
		const explicitAuth = registrations.find((r) => r.name === "explicit-auth");
		assert.ok(noAuth?.config.refreshModels && explicitAuth?.config.refreshModels);

		await noAuth.config.refreshModels(
			context({
				allowNetwork: true,
				credential: { type: "api_key", key: "should-not-be-bearer" },
			}),
		);
		await explicitAuth.config.refreshModels(
			context({
				allowNetwork: true,
				credential: { type: "api_key", key: "should-not-overwrite" },
			}),
		);

		const gatewayCalls = seenHeaders.get(
			"https://gateway.example.test/v1/models",
		);
		assert.equal(gatewayCalls?.length, 2);
		for (const hdrs of gatewayCalls ?? []) {
			assert.equal(hdrs.Accept, "application/vnd.gateway+json");
			assert.equal("accept" in hdrs, false);
			assert.equal(hdrs["X-Custom-Token"], "env-header-value");
			assert.equal(findHeaderIgnoreCase(hdrs, "authorization"), undefined);
		}

		const explicitCalls = seenHeaders.get(
			"https://explicit.example.test/v1/models",
		);
		assert.equal(explicitCalls?.length, 2);
		for (const hdrs of explicitCalls ?? []) {
			assert.equal(hdrs.Authorization, "ApiKey custom-auth-scheme");
		}
	} finally {
		clearPricingMemoryCache();
		clearConfigValueCache();
		globalThis.fetch = originalFetch;
		if (originalDirectory === undefined) delete process.env.PI_CODING_AGENT_DIR;
		else process.env.PI_CODING_AGENT_DIR = originalDirectory;
		if (originalCustomEnv === undefined)
			delete process.env.PI_TEST_CUSTOM_HEADER;
		else process.env.PI_TEST_CUSTOM_HEADER = originalCustomEnv;
		await rm(directory, { recursive: true, force: true });
	}
});

// =============================================================================
// Stale auth.json oauth warning & fallback + stored api_key usage
// =============================================================================

test("stale auth.json oauth warns with /logout guidance and falls back to models.json apiKey; stored api_key is used when models.json omits apiKey", async () => {
	clearPricingMemoryCache();
	const originalFetch = globalThis.fetch;
	const originalDirectory = process.env.PI_CODING_AGENT_DIR;
	const { directory } = await writeConfig(`{
		"providers": {
			"cliproxyapi": {
				"baseUrl": "http://127.0.0.1:8317/v1",
				"apiKey": "models-json-key"
			},
			"my-api": {
				"baseUrl": "http://127.0.0.1:9000/v1"
			}
		}
	}`);
	await writeFile(
		join(directory, "auth.json"),
		JSON.stringify({
			cliproxyapi: { type: "oauth", access: "stale-token" },
			"my-api": { type: "api_key", key: "stored-key" },
		}),
		"utf8",
	);
	await writeFile(
		join(directory, "pi-openai-model-discovery.json"),
		JSON.stringify({ pricingUrl: null }),
		"utf8",
	);
	process.env.PI_CODING_AGENT_DIR = directory;

	const authHeadersByUrl = new Map<string, string | undefined>();
	globalThis.fetch = async (input, init) => {
		const url = String(input);
		authHeadersByUrl.set(
			url,
			findHeaderIgnoreCase(init?.headers, "authorization"),
		);
		return jsonResponse({
			data: [{ id: "model-1", context_length: 128_000 }],
		});
	};

	const { pi, registrations } = createFakePi();

	try {
		const stderr = await captureStderr(async () => {
			await openAIModelDiscovery(pi);
		});

		assert.match(stderr, /auth\.json/);
		assert.match(stderr, /\/logout cliproxyapi/);

		// cliproxyapi uses models.json's apiKey, NOT the stale oauth token
		assert.equal(
			authHeadersByUrl.get("http://127.0.0.1:8317/v1/models"),
			"Bearer models-json-key",
		);

		// my-api uses stored-key from auth.json and does NOT get synthetic apiKey: "local"
		assert.equal(
			authHeadersByUrl.get("http://127.0.0.1:9000/v1/models"),
			"Bearer stored-key",
		);
		const myApiReg = registrations.find((r) => r.name === "my-api");
		assert.ok(myApiReg);
		assert.equal(myApiReg.config.apiKey, undefined);
	} finally {
		clearPricingMemoryCache();
		globalThis.fetch = originalFetch;
		if (originalDirectory === undefined) delete process.env.PI_CODING_AGENT_DIR;
		else process.env.PI_CODING_AGENT_DIR = originalDirectory;
		await rm(directory, { recursive: true, force: true });
	}
});

// =============================================================================
// Non-OpenAI api filtering & providerId scoping
// =============================================================================

test("readDynamicProviders filters non-OpenAI apis and providerId scopes discovery to a single provider", async () => {
	clearPricingMemoryCache();
	const originalFetch = globalThis.fetch;
	const originalDirectory = process.env.PI_CODING_AGENT_DIR;
	const { directory, filePath } = await writeConfig(`{
		"providers": {
			"default-openai": {
				"baseUrl": "https://default.example.test/v1"
			},
			"completions-api": {
				"baseUrl": "https://completions.example.test/v1",
				"api": "openai-completions"
			},
			"responses-api": {
				"baseUrl": "https://responses.example.test/v1",
				"api": "openai-responses"
			},
			"anthropic-api": {
				"baseUrl": "https://anthropic.example.test/v1",
				"api": "anthropic-messages"
			},
			"google-api": {
				"baseUrl": "https://google.example.test/v1",
				"api": "google-generative-ai"
			}
		}
	}`);

	try {
		const providers = await readDynamicProviders(filePath);
		assert.deepEqual(
			providers.map((p) => ({ id: p.id, api: p.api })),
			[
				{ id: "default-openai", api: "openai-completions" },
				{ id: "completions-api", api: "openai-completions" },
				{ id: "responses-api", api: "openai-responses" },
			],
		);

		// providerId scoping via pi-openai-model-discovery.json
		await writeFile(
			join(directory, "pi-openai-model-discovery.json"),
			JSON.stringify({
				providerId: "responses-api",
				pricingUrl: null,
			}),
			"utf8",
		);
		process.env.PI_CODING_AGENT_DIR = directory;

		const requestedUrls: string[] = [];
		globalThis.fetch = async (input) => {
			requestedUrls.push(String(input));
			return jsonResponse({
				data: [{ id: "gpt-5.4", context_length: 200_000 }],
			});
		};

		const { pi, registrations } = createFakePi();
		await openAIModelDiscovery(pi);

		assert.deepEqual(requestedUrls, [
			"https://responses.example.test/v1/models",
		]);
		assert.deepEqual(
			registrations.map((r) => ({ name: r.name, api: r.config.api })),
			[{ name: "responses-api", api: "openai-responses" }],
		);
	} finally {
		clearPricingMemoryCache();
		globalThis.fetch = originalFetch;
		if (originalDirectory === undefined) delete process.env.PI_CODING_AGENT_DIR;
		else process.env.PI_CODING_AGENT_DIR = originalDirectory;
		await rm(directory, { recursive: true, force: true });
	}
});

// =============================================================================
// Pricing & capability enrichment (src/pricing.ts + mapModelRecord)
// =============================================================================

test("lookupPricingEntry supports exact, case-insensitive, prefix, variant, and date-suffix fallback while preferring exact dated entries", () => {
	const exactEntry: PricingEntry = { input_cost_per_token: 0.000001 };
	const gpt55Entry: PricingEntry = { input_cost_per_token: 0.000002 };
	const gpt54Entry: PricingEntry = { input_cost_per_token: 0.000003 };
	const deepseekR1Entry: PricingEntry = { input_cost_per_token: 0.00000055 };
	const claudeBaseEntry: PricingEntry = { input_cost_per_token: 0.000003 };
	const claudeDatedEntry: PricingEntry = { input_cost_per_token: 0.0000035 };
	const gpt4oBaseEntry: PricingEntry = { input_cost_per_token: 0.0000025 };

	const catalog: PricingCatalog = {
		"Exact-Model": exactEntry,
		"gpt-5.5": gpt55Entry,
		"gpt-5.4": gpt54Entry,
		"deepseek-r1": deepseekR1Entry,
		"claude-sonnet-4": claudeBaseEntry,
		"gpt-4o": gpt4oBaseEntry,
	};

	// Exact & case-insensitive
	assert.equal(lookupPricingEntry(catalog, "Exact-Model"), exactEntry);
	assert.equal(lookupPricingEntry(catalog, "exact-model"), exactEntry);
	assert.equal(lookupPricingEntry(catalog, "GPT-5.4"), gpt54Entry);

	// Prefix stripping
	assert.equal(lookupPricingEntry(catalog, "cc/gpt-5.5"), gpt55Entry);
	assert.equal(lookupPricingEntry(catalog, "openai/gpt-5.4"), gpt54Entry);

	// Variant + prefix stripping
	assert.equal(
		lookupPricingEntry(catalog, "deepseek/deepseek-r1:free"),
		deepseekR1Entry,
	);

	// Date suffix fallback (-YYYYMMDD and -YYYY-MM-DD)
	assert.equal(
		lookupPricingEntry(catalog, "claude-sonnet-4-20250514"),
		claudeBaseEntry,
	);
	assert.equal(
		lookupPricingEntry(catalog, "anthropic/claude-sonnet-4-20250514:thinking"),
		claudeBaseEntry,
	);
	assert.equal(lookupPricingEntry(catalog, "gpt-4o-2024-08-06"), gpt4oBaseEntry);

	// Prefers exact dated entry when both dated and undated exist
	const catalogWithDated: PricingCatalog = {
		...catalog,
		"claude-sonnet-4-20250514": claudeDatedEntry,
	};
	assert.equal(
		lookupPricingEntry(catalogWithDated, "claude-sonnet-4-20250514"),
		claudeDatedEntry,
	);
	assert.equal(
		lookupPricingEntry(catalogWithDated, "claude-sonnet-4-20250999"),
		claudeBaseEntry,
	);

	// Unknown / empty
	assert.equal(lookupPricingEntry(catalog, "unknown-model"), undefined);
	assert.equal(lookupPricingEntry(catalog, "   "), undefined);
});

test("pricingToCost converts per-million cleanly and handles tiered_pricing and legacy long-context pricing", () => {
	// Clean per-million conversion without floating-point artifacts
	const clean = pricingToCost({
		input_cost_per_token: 0.00000015,
		output_cost_per_token: 0.0000006,
		cache_read_input_token_cost: 0.0000000375,
		cache_creation_input_token_cost: 0.0000001875,
	});
	assert.deepEqual(clean, {
		input: 0.15,
		output: 0.6,
		cacheRead: 0.0375,
		cacheWrite: 0.1875,
	});

	// tiered_pricing sorting, deduplication, and base-rate fallback
	const tiered = pricingToCost({
		input_cost_per_token: 0.000002,
		output_cost_per_token: 0.000008,
		cache_read_input_token_cost: 0.0000002,
		cache_creation_input_token_cost: 0.0000025,
		tiered_pricing: [
			{
				range: [500_000, Number.POSITIVE_INFINITY],
				input_cost_per_token: 0.000006,
				output_cost_per_token: 0.000024,
			},
			{
				range: [200_000, 500_000],
				input_cost_per_token: 0.000003,
				output_cost_per_token: 0.000012,
			},
			// Duplicate threshold overrides earlier entry for 200_000
			{
				range: [200_000, 500_000],
				input_cost_per_token: 0.000004,
				output_cost_per_token: 0.000016,
				cache_read_input_token_cost: 0.0000004,
			},
			// Invalid/zero threshold is ignored
			{
				range: [0, 200_000],
				input_cost_per_token: 0.000001,
			},
		],
	});
	assert.deepEqual(tiered, {
		input: 2,
		output: 8,
		cacheRead: 0.2,
		cacheWrite: 2.5,
		tiers: [
			{
				inputTokensAbove: 200_000,
				input: 4,
				output: 16,
				cacheRead: 0.4,
				cacheWrite: 2.5,
			},
			{
				inputTokensAbove: 500_000,
				input: 6,
				output: 24,
				cacheRead: 0.2,
				cacheWrite: 2.5,
			},
		],
	});

	// Legacy *_above_272k_tokens (default threshold 272_000)
	const legacyAbove = pricingToCost({
		input_cost_per_token: 0.000002,
		output_cost_per_token: 0.00001,
		cache_read_input_token_cost: 0.0000005,
		cache_creation_input_token_cost: 0.0000025,
		input_cost_per_token_above_272k_tokens: 0.000004,
		output_cost_per_token_above_272k_tokens: 0.000015,
		cache_read_input_token_cost_above_272k_tokens: 0.000001,
		cache_creation_input_token_cost_above_272k_tokens: 0.000005,
	});
	assert.deepEqual(legacyAbove.tiers, [
		{
			inputTokensAbove: 272_000,
			input: 4,
			output: 15,
			cacheRead: 1,
			cacheWrite: 5,
		},
	]);

	// Legacy long_context_*_multiplier + custom long_context_input_token_threshold
	const legacyMultiplier = pricingToCost({
		input_cost_per_token: 0.000003,
		output_cost_per_token: 0.000015,
		cache_read_input_token_cost: 0.0000003,
		cache_creation_input_token_cost: 0.00000375,
		long_context_input_token_threshold: 200_000,
		long_context_input_cost_multiplier: 2,
		long_context_output_cost_multiplier: 1.5,
		long_context_cache_read_cost_multiplier: 2,
		long_context_cache_creation_cost_multiplier: 2,
	});
	assert.deepEqual(legacyMultiplier.tiers, [
		{
			inputTokensAbove: 200_000,
			input: 6,
			output: 22.5,
			cacheRead: 0.6,
			cacheWrite: 7.5,
		},
	]);
});

test("pricingToThinking and pricingToInput map reasoning effort flags and input modalities", () => {
	assert.equal(pricingToThinking(undefined), undefined);

	const nonReasoning = pricingToThinking({ supports_reasoning: false });
	assert.deepEqual(nonReasoning, { reasoning: false });
	assert.equal("thinkingLevelMap" in (nonReasoning ?? {}), false);

	const fullReasoning = pricingToThinking({
		supports_reasoning: true,
		supports_none_reasoning_effort: true,
		supports_minimal_reasoning_effort: true,
		supports_xhigh_reasoning_effort: true,
		supports_max_reasoning_effort: true,
	});
	assert.deepEqual(fullReasoning, {
		reasoning: true,
		thinkingLevelMap: {
			off: "none",
			minimal: "minimal",
			low: "low",
			medium: "medium",
			high: "high",
			xhigh: "xhigh",
			max: "max",
		},
	});

	const partialReasoning = pricingToThinking({
		supports_reasoning: true,
		supports_none_reasoning_effort: false,
		supports_xhigh_reasoning_effort: true,
	});
	assert.deepEqual(partialReasoning, {
		reasoning: true,
		thinkingLevelMap: {
			off: null,
			minimal: null,
			low: "low",
			medium: "medium",
			high: "high",
			xhigh: "xhigh",
			max: null,
		},
	});

	assert.equal(pricingToInput(undefined), undefined);
	assert.deepEqual(
		pricingToInput({ supported_modalities: ["text", "image", "audio"] }),
		["text", "image"],
	);
	assert.deepEqual(
		pricingToInput({ supported_input_modalities: ["TEXT"] }),
		["text"],
	);
	assert.deepEqual(pricingToInput({ supports_vision: true }), [
		"text",
		"image",
	]);
});

test("mapModelRecord enforces priority order: overrides > endpoint > pricing > defaults > built-in defaults", () => {
	const pricing: PricingEntry = {
		max_input_tokens: 200_000,
		max_output_tokens: 32_000,
		input_cost_per_token: 0.000003,
		output_cost_per_token: 0.000015,
		long_context_input_token_threshold: 200_000,
		long_context_input_cost_multiplier: 2,
		long_context_output_cost_multiplier: 1.5,
		supports_vision: true,
		supports_reasoning: true,
		supports_xhigh_reasoning_effort: true,
	};

	const defaults = {
		contextWindow: 64_000,
		maxTokens: 8_192,
		input: ["text"] as Array<"text" | "image">,
		reasoning: false,
	};

	// Endpoint context_window wins over pricing max_input_tokens, while pricing fills cost, tiers, maxTokens, input, reasoning
	const endpointOverPricing = mapModelRecord(
		{
			id: "hybrid-model",
			context_window: 372_000,
		},
		dynamicProvider,
		{ pricing, defaults },
	);
	assert.equal(endpointOverPricing.contextWindow, 372_000);
	assert.equal(endpointOverPricing.maxTokens, 32_000);
	assert.deepEqual(endpointOverPricing.input, ["text", "image"]);
	assert.equal(endpointOverPricing.reasoning, true);
	assert.deepEqual(endpointOverPricing.cost, {
		input: 3,
		output: 15,
		cacheRead: 0,
		cacheWrite: 0,
		tiers: [
			{
				inputTokensAbove: 200_000,
				input: 6,
				output: 22.5,
				cacheRead: 0,
				cacheWrite: 0,
			},
		],
	});

	// Override wins over endpoint and pricing, while preserving unoverridden fields
	const withOverride = mapModelRecord(
		{
			id: "hybrid-model",
			context_window: 372_000,
		},
		dynamicProvider,
		{
			pricing,
			defaults,
			override: {
				name: "Custom Display",
				contextWindow: 500_000,
				cost: { input: 1.25 },
				thinkingLevelMap: { max: "max" },
			},
		},
	);
	assert.equal(withOverride.name, "Custom Display");
	assert.equal(withOverride.contextWindow, 500_000);
	assert.equal(withOverride.maxTokens, 32_000);
	assert.equal(withOverride.cost.input, 1.25);
	assert.equal(withOverride.cost.output, 15);
	assert.equal(withOverride.cost.tiers?.length, 1);
	assert.equal(withOverride.thinkingLevelMap?.max, "max");
	assert.equal(withOverride.thinkingLevelMap?.xhigh, "xhigh");

	// Defaults win over built-in defaults when endpoint and pricing omit them
	const fromDefaults = mapModelRecord({ id: "plain-model" }, dynamicProvider, {
		defaults,
	});
	assert.equal(fromDefaults.contextWindow, 64_000);
	assert.equal(fromDefaults.maxTokens, 8_192);
	assert.equal(fromDefaults.reasoning, false);
	assert.equal("thinkingLevelMap" in fromDefaults, false);
});

test("fetchPricingCatalog sends no headers, caches to disk within TTL, falls back to stale cache on failure, and handles null/empty url", async () => {
	clearPricingMemoryCache();
	const originalFetch = globalThis.fetch;
	const tempDir = await mkdtemp(join(tmpdir(), "pi-pricing-cache-"));
	const cachePath = join(tempDir, "pricing-cache.json");
	const url = "https://pricing.example.test/catalog.json";

	let fetchCalls = 0;
	let receivedInit: RequestInit | undefined;

	globalThis.fetch = async (_input, init) => {
		fetchCalls += 1;
		receivedInit = init;
		return jsonResponse({
			sample_spec: { max_input_tokens: 999 },
			$schema: { title: "ignored" },
			"gpt-5.4": { input_cost_per_token: 0.0000025 },
		});
	};

	try {
		// null or empty URL returns {} cleanly without calling fetch
		assert.deepEqual(await fetchPricingCatalog({ url: null }), {});
		assert.deepEqual(await fetchPricingCatalog({ url: "" }), {});
		assert.deepEqual(await fetchPricingCatalog({ url: "   " }), {});
		assert.equal(fetchCalls, 0);

		// First live fetch populates disk + memory cache and sends NO headers
		const first = await fetchPricingCatalog({
			url,
			cachePath,
			cacheTtlMs: 60_000,
		});
		assert.equal(fetchCalls, 1);
		assert.equal(receivedInit?.headers, undefined);
		assert.equal(first["gpt-5.4"]?.input_cost_per_token, 0.0000025);
		assert.equal("sample_spec" in first, false);
		assert.equal("$schema" in first, false);

		// Clear memory cache -> should read from disk cache within TTL without calling fetch
		clearPricingMemoryCache();
		const fromDisk = await fetchPricingCatalog({
			url,
			cachePath,
			cacheTtlMs: 60_000,
		});
		assert.equal(fetchCalls, 1);
		assert.deepEqual(fromDisk, first);

		// Backdate disk cache timestamp so it is stale, then simulate network failure -> stale fallback
		const rawCache = JSON.parse(await readFile(cachePath, "utf8")) as {
			url: string;
			fetchedAt: number;
			catalog: PricingCatalog;
		};
		rawCache.fetchedAt = Date.now() - 120_000;
		await writeFile(cachePath, JSON.stringify(rawCache), "utf8");
		clearPricingMemoryCache();

		globalThis.fetch = async () => {
			fetchCalls += 1;
			throw new Error("simulated offline");
		};
		let staleNetworkFallback: PricingCatalog = {};
		const netWarn = await captureStderr(async () => {
			staleNetworkFallback = await fetchPricingCatalog({
				url,
				cachePath,
				cacheTtlMs: 60_000,
			});
		});
		assert.equal(fetchCalls, 2);
		assert.match(netWarn, /Pricing catalog warning/);
		assert.deepEqual(staleNetworkFallback, first);

		// Simulate HTTP 503 failure with forceRefresh -> still falls back to stale disk cache
		clearPricingMemoryCache();
		globalThis.fetch = async () => {
			fetchCalls += 1;
			return new Response("service unavailable", { status: 503 });
		};
		let staleHttpFallback: PricingCatalog = {};
		const httpWarn = await captureStderr(async () => {
			staleHttpFallback = await fetchPricingCatalog({
				url,
				cachePath,
				cacheTtlMs: 60_000,
				forceRefresh: true,
			});
		});
		assert.equal(fetchCalls, 3);
		assert.match(httpWarn, /HTTP 503/);
		assert.deepEqual(staleHttpFallback, first);
	} finally {
		clearPricingMemoryCache();
		globalThis.fetch = originalFetch;
		await rm(tempDir, { recursive: true, force: true });
	}
});

// =============================================================================
// Include / exclude regex filtering & deduplication
// =============================================================================

test("default exclude patterns filter non-chat models while keeping multimodal chat models, exclude: [] disables defaults, invalid regex warns safely, and duplicates keep first", async () => {
	const defaultInclude = compilePatterns(undefined, DEFAULT_INCLUDE_PATTERNS);
	const defaultExclude = compilePatterns(undefined, DEFAULT_EXCLUDE_PATTERNS);

	const excludedIds = [
		"whisper-1",
		"tts-1",
		"dall-e-3",
		"text-embedding-3-large",
		"omni-moderation-latest",
		"gpt-4o-realtime-preview",
		"auto-review",
		"sora-2",
	];
	for (const id of excludedIds) {
		assert.equal(
			shouldIncludeModel(id, defaultInclude, defaultExclude),
			false,
			`Expected ${id} to be excluded by default patterns`,
		);
	}

	const keptIds = [
		"gemini-2.5-flash-image",
		"gemini-2.5-flash-native-audio-latest",
		"gpt-5.4",
		"claude-sonnet-4",
	];
	for (const id of keptIds) {
		assert.equal(
			shouldIncludeModel(id, defaultInclude, defaultExclude),
			true,
			`Expected ${id} to be kept by default patterns`,
		);
	}

	// exclude: [] disables default exclusions
	const emptyExclude = compilePatterns([], DEFAULT_EXCLUDE_PATTERNS);
	assert.equal(emptyExclude.length, 0);
	assert.equal(
		shouldIncludeModel("whisper-1", defaultInclude, emptyExclude),
		true,
	);
	assert.equal(
		shouldIncludeModel("auto-review", defaultInclude, emptyExclude),
		true,
	);

	// Invalid user regex warns on stderr without crashing
	let compiledWithInvalid: RegExp[] = [];
	const regexStderr = await captureStderr(() => {
		compiledWithInvalid = compilePatterns(
			["[unclosed-group", "^gpt-"],
			DEFAULT_INCLUDE_PATTERNS,
		);
	});
	assert.match(regexStderr, /ignoring invalid regex pattern/);
	assert.equal(compiledWithInvalid.length, 1);
	assert.equal(compiledWithInvalid[0]?.test("gpt-5.4"), true);

	// End-to-end refreshProvider: filtering + duplicate deduplication (first wins)
	const originalFetch = globalThis.fetch;
	globalThis.fetch = async () =>
		jsonResponse({
			data: [
				{ id: "whisper-1", context_length: 1000 },
				{ id: "tts-1", context_length: 1000 },
				{ id: "dall-e-3", context_length: 1000 },
				{ id: "text-embedding-3-large", context_length: 8192 },
				{ id: "omni-moderation-latest", context_length: 4096 },
				{ id: "gpt-4o-realtime-preview", context_length: 128_000 },
				{ id: "auto-review", context_length: 128_000 },
				{ id: "sora-2", context_length: 1000 },
				{
					id: "gemini-2.5-flash-image",
					name: "First Flash Image",
					context_length: 1_000_000,
				},
				{
					id: "gemini-2.5-flash-image",
					name: "Duplicate Flash Image",
					context_length: 12_345,
				},
				{
					id: "gemini-2.5-flash-native-audio-latest",
					context_length: 1_000_000,
				},
			],
		});

	try {
		const filtered = await refreshProvider(dynamicProvider, context(), {
			pricingUrl: null,
			include: defaultInclude,
			exclude: defaultExclude,
		});
		assert.deepEqual(
			filtered.map((m) => m.id),
			["gemini-2.5-flash-image", "gemini-2.5-flash-native-audio-latest"],
		);
		// First duplicate wins
		assert.equal(filtered[0]?.name, "First Flash Image");
		assert.equal(filtered[0]?.contextWindow, 1_000_000);

		// With exclude: [] all non-duplicate models pass through
		const unfiltered = await refreshProvider(dynamicProvider, context(), {
			pricingUrl: null,
			include: defaultInclude,
			exclude: emptyExclude,
		});
		assert.equal(unfiltered.length, 10);
		assert.ok(unfiltered.some((m) => m.id === "whisper-1"));
		assert.ok(unfiltered.some((m) => m.id === "auto-review"));
	} finally {
		globalThis.fetch = originalFetch;
	}
});

// =============================================================================
// JSONC pi-openai-model-discovery.json, deep merging, & PI_CODING_AGENT_DIR ~
// =============================================================================

test("readDiscoveryConfig parses JSONC defaults/overrides with deep compat/thinkingLevelMap/cost merging and agentDir expands ~", async () => {
	const originalDirectory = process.env.PI_CODING_AGENT_DIR;
	const tempDir = await mkdtemp(join(tmpdir(), "pi-discovery-cfg-"));
	const configFile = join(tempDir, "pi-openai-model-discovery.json");

	await writeFile(
		configFile,
		`\uFEFF{
		// Optional JSONC config
		"providerId": "cliproxyapi",
		"pricingUrl": "https://pricing.example.test/prices.json",
		"pricingCacheTtlMs": 30000,
		"requestTimeoutMs": 5000,
		"include": ["^gpt-", "^claude-"],
		"exclude": [],
		/* Defaults for discovered models */
		"defaults": {
			"contextWindow": 256000,
			"maxTokens": 32768,
			"input": ["text", "image"],
			"reasoning": true,
			"thinkingLevelMap": {
				"low": "low",
				"high": "high",
				"max": null,
			},
			"compat": {
				"supportsDeveloperRole": false,
				"openRouterRouting": {
					"order": ["openai"],
					"allow_fallbacks": false,
				},
				"chatTemplateKwargs": {
					"enable_thinking": true,
				},
			},
		},
		"overrides": {
			"gpt-5.4": {
				"name": "GPT 5.4 Override",
				"maxTokens": 65536,
				"thinkingLevelMap": {
					"max": "max",
					"xhigh": "xhigh",
				},
				"cost": {
					"input": 2.0,
				},
				"compat": {
					"openRouterRouting": {
						"allow_fallbacks": true,
					},
					"chatTemplateKwargs": {
						"temperature_override": 0.2,
					},
				},
			},
		},
	}`,
		"utf8",
	);

	try {
		const config = await readDiscoveryConfig(configFile);
		assert.equal(config.providerId, "cliproxyapi");
		assert.equal(config.pricingUrl, "https://pricing.example.test/prices.json");
		assert.equal(config.pricingCacheTtlMs, 30_000);
		assert.equal(config.requestTimeoutMs, 5_000);
		assert.deepEqual(config.include, ["^gpt-", "^claude-"]);
		assert.deepEqual(config.exclude, []);
		assert.equal(config.defaults?.contextWindow, 256_000);
		assert.equal(config.defaults?.maxTokens, 32_768);
		assert.deepEqual(config.defaults?.input, ["text", "image"]);
		assert.equal(config.defaults?.reasoning, true);
		assert.deepEqual(config.defaults?.thinkingLevelMap, {
			low: "low",
			high: "high",
			max: null,
		});

		// Verify deep merge via mapModelRecord + applyModelOverride + mergeCompat
		const mapped = mapModelRecord(
			{
				id: "gpt-5.4",
				cost: { input: 5, output: 15, cacheRead: 0.5, cacheWrite: 1 },
			},
			dynamicProvider,
			{
				defaults: config.defaults,
				override: config.overrides?.["gpt-5.4"],
			},
		);

		assert.equal(mapped.name, "GPT 5.4 Override");
		assert.equal(mapped.contextWindow, 256_000);
		assert.equal(mapped.maxTokens, 65_536);
		assert.deepEqual(mapped.input, ["text", "image"]);
		assert.deepEqual(mapped.thinkingLevelMap, {
			low: "low",
			high: "high",
			max: "max",
			xhigh: "xhigh",
		});
		assert.deepEqual(mapped.cost, {
			input: 2.0,
			output: 15,
			cacheRead: 0.5,
			cacheWrite: 1,
		});
		assert.deepEqual(mapped.compat, {
			supportsDeveloperRole: false,
			openRouterRouting: {
				order: ["openai"],
				allow_fallbacks: true,
			},
			chatTemplateKwargs: {
				enable_thinking: true,
				temperature_override: 0.2,
			},
		});

		// Direct mergeCompat + applyModelOverride check
		assert.deepEqual(
			mergeCompat(
				{ a: 1, openRouterRouting: { order: ["a"], only: ["x"] } },
				{ b: 2, openRouterRouting: { only: ["y"] } },
			),
			{ a: 1, b: 2, openRouterRouting: { order: ["a"], only: ["y"] } },
		);
		const overridden = applyModelOverride(mapped, { reasoning: false });
		assert.equal(overridden.reasoning, false);

		// agentDir() tilde expansion
		process.env.PI_CODING_AGENT_DIR = "~";
		assert.equal(agentDir(), homedir());

		process.env.PI_CODING_AGENT_DIR = "~/my-pi-agent";
		assert.equal(agentDir(), join(homedir(), "my-pi-agent"));

		process.env.PI_CODING_AGENT_DIR = "~\\win-pi-agent";
		assert.equal(agentDir(), join(homedir(), "win-pi-agent"));

		delete process.env.PI_CODING_AGENT_DIR;
		assert.equal(agentDir(), join(homedir(), ".pi", "agent"));
	} finally {
		if (originalDirectory === undefined) delete process.env.PI_CODING_AGENT_DIR;
		else process.env.PI_CODING_AGENT_DIR = originalDirectory;
		await rm(tempDir, { recursive: true, force: true });
	}
});

// =============================================================================
// Stored catalog round-trip preservation
// =============================================================================

test("persisted models preserve reasoning: false, custom thinkingLevelMap, cost.tiers, and compat through models-store.json round-trip", async () => {
	clearPricingMemoryCache();
	const originalFetch = globalThis.fetch;
	const originalDirectory = process.env.PI_CODING_AGENT_DIR;
	const { directory } = await writeConfig(`{
		"providers": {
			"local-server": {
				"baseUrl": "http://127.0.0.1:1234/v1",
				"apiKey": "roundtrip-key"
			}
		}
	}`);
	await writeFile(
		join(directory, "pi-openai-model-discovery.json"),
		JSON.stringify({
			pricingUrl: null,
			defaults: {
				compat: {
					supportsDeveloperRole: false,
					openRouterRouting: { order: ["anthropic"] },
				},
			},
			overrides: {
				"fast-chat": {
					reasoning: false,
				},
			},
		}),
		"utf8",
	);
	process.env.PI_CODING_AGENT_DIR = directory;

	const pricingCatalog: PricingCatalog = {
		"fast-chat": {
			max_input_tokens: 128_000,
			max_output_tokens: 8_192,
			input_cost_per_token: 0.00000015,
			output_cost_per_token: 0.0000006,
			supports_reasoning: false,
			supported_input_modalities: ["text"],
		},
		"deep-reasoner": {
			max_input_tokens: 1_000_000,
			max_output_tokens: 64_000,
			input_cost_per_token: 0.000003,
			output_cost_per_token: 0.000015,
			cache_read_input_token_cost: 0.0000003,
			cache_creation_input_token_cost: 0.00000375,
			long_context_input_token_threshold: 200_000,
			long_context_input_cost_multiplier: 2,
			long_context_output_cost_multiplier: 1.5,
			long_context_cache_read_cost_multiplier: 2,
			long_context_cache_creation_cost_multiplier: 2,
			supports_reasoning: true,
			supports_none_reasoning_effort: true,
			supports_minimal_reasoning_effort: false,
			supports_xhigh_reasoning_effort: false,
			supports_max_reasoning_effort: false,
			supports_vision: true,
		},
	};

	globalThis.fetch = async () =>
		jsonResponse({
			data: [{ id: "fast-chat" }, { id: "deep-reasoner" }],
		});

	try {
		const config = await readDiscoveryConfig(
			join(directory, "pi-openai-model-discovery.json"),
		);
		const published: Publication[] = [];
		const discovered = await refreshProvider(
			dynamicProvider,
			context({
				allowNetwork: true,
				publish: async (pub) => {
					published.push(pub);
					return true;
				},
			}),
			{
				config,
				pricingCatalog,
				pricingUrl: null,
			},
		);

		assert.equal(published.length, 1);
		const persisted = published[0]?.persist;
		assert.ok(persisted);

		// Serialize to models-store.json on disk
		const storePath = join(directory, "models-store.json");
		await writeFile(
			storePath,
			JSON.stringify({ "local-server": persisted }),
			"utf8",
		);
		const reloadedStore = JSON.parse(await readFile(storePath, "utf8")) as {
			"local-server": RefreshContext["stored"];
		};

		// 1. Round-trip via refreshProvider({ allowNetwork: false, stored })
		const fromRefresh = await refreshProvider(
			dynamicProvider,
			context({
				allowNetwork: false,
				stored: reloadedStore["local-server"],
			}),
		);
		assert.deepEqual(fromRefresh, discovered);

		// 2. Round-trip via cold openAIModelDiscovery reading models-store.json when network is down
		globalThis.fetch = async () => new Response("offline", { status: 503 });
		const { pi, registrations } = createFakePi();
		await captureStderr(async () => {
			await openAIModelDiscovery(pi);
		});

		const registeredModels = registrations[0]?.config.models;
		assert.ok(registeredModels);
		assert.deepEqual(registeredModels, discovered);

		const fastChat = registeredModels.find((m) => m.id === "fast-chat");
		const deepReasoner = registeredModels.find((m) => m.id === "deep-reasoner");
		assert.ok(fastChat && deepReasoner);

		assert.equal(fastChat.reasoning, false);
		assert.equal("thinkingLevelMap" in fastChat, false);
		assert.deepEqual(fastChat.compat, {
			supportsDeveloperRole: false,
			openRouterRouting: { order: ["anthropic"] },
		});

		assert.equal(deepReasoner.reasoning, true);
		assert.deepEqual(deepReasoner.thinkingLevelMap, {
			off: "none",
			minimal: null,
			low: "low",
			medium: "medium",
			high: "high",
			xhigh: null,
			max: null,
		});
		assert.deepEqual(deepReasoner.cost.tiers, [
			{
				inputTokensAbove: 200_000,
				input: 6,
				output: 22.5,
				cacheRead: 0.6,
				cacheWrite: 7.5,
			},
		]);
		assert.deepEqual(deepReasoner.compat, {
			supportsDeveloperRole: false,
			openRouterRouting: { order: ["anthropic"] },
		});
	} finally {
		clearPricingMemoryCache();
		globalThis.fetch = originalFetch;
		if (originalDirectory === undefined) delete process.env.PI_CODING_AGENT_DIR;
		else process.env.PI_CODING_AGENT_DIR = originalDirectory;
		await rm(directory, { recursive: true, force: true });
	}
});

test("pricingToCost supports 128k and 200k long-context thresholds", () => {
	const claudeAndGemini = pricingToCost({
		input_cost_per_token: 0.000003,
		output_cost_per_token: 0.000015,
		cache_read_input_token_cost: 0.0000003,
		input_cost_per_token_above_128k_tokens: 0.0000045,
		input_cost_per_token_above_200k_tokens: 0.000006,
		output_cost_per_token_above_200k_tokens: 0.0000225,
	});
	assert.deepEqual(claudeAndGemini.tiers, [
		{
			inputTokensAbove: 128_000,
			input: 4.5,
			output: 15,
			cacheRead: 0.3,
			cacheWrite: 0,
		},
		{
			inputTokensAbove: 200_000,
			input: 6,
			output: 22.5,
			cacheRead: 0.3,
			cacheWrite: 0,
		},
	]);
});

test("provider.compat, override reasoning round-trip, and prototype-key safety", async () => {
	const providerWithCompat: DynamicProvider = {
		...dynamicProvider,
		compat: {
			supportsDeveloperRole: false,
			openRouterRouting: { order: ["openai"] },
		},
	};

	const mapped = mapModelRecord({ id: "gpt-4o" }, providerWithCompat, {
		pricing: { supports_reasoning: false },
		defaults: {
			compat: { maxTokensField: "max_tokens" },
		},
		override: {
			reasoning: true,
			cost: { input: 5, output: undefined },
		},
	});
	assert.deepEqual(mapped.compat, {
		maxTokensField: "max_tokens",
		supportsDeveloperRole: false,
		openRouterRouting: { order: ["openai"] },
	});
	assert.equal(mapped.reasoning, true);
	assert.deepEqual(mapped.thinkingLevelMap, { xhigh: "xhigh", max: "max" });
	assert.equal(mapped.cost.input, 5);
	assert.equal(mapped.cost.output, 0);

	// Round-tripping through mapModelRecord preserves identical reasoning/thinkingLevelMap
	const roundTripped = mapModelRecord(mapped, providerWithCompat);
	assert.deepEqual(roundTripped, mapped);

	// Overriding reasoning: false drops thinkingLevelMap cleanly
	const disabled = mapModelRecord(
		{ id: "o3" },
		dynamicProvider,
		{
			pricing: { supports_reasoning: true },
			override: { reasoning: false },
		},
	);
	assert.equal(disabled.reasoning, false);
	assert.equal("thinkingLevelMap" in disabled, false);

	// Prototype keys in upstream /models and overrides do not corrupt model names
	const originalFetch = globalThis.fetch;
	globalThis.fetch = async () =>
		new Response(
			JSON.stringify({
				data: [{ id: "constructor" }, { id: "toString" }],
			}),
			{ status: 200, headers: { "content-type": "application/json" } },
		);
	try {
		const models = await refreshProvider(
			dynamicProvider,
			context(),
			{
				config: {
					overrides: {
						"real-model": { name: "Real" },
					},
				},
				pricingUrl: null,
			},
		);
		assert.deepEqual(
			models.map((m) => ({ id: m.id, name: m.name })),
			[
				{ id: "constructor", name: "constructor" },
				{ id: "toString", name: "toString" },
			],
		);
	} finally {
		globalThis.fetch = originalFetch;
	}
});

test("auth.json env interpolation on startup, PI_OFFLINE, header-only auth, and /reload cleanup", async () => {
	const originalFetch = globalThis.fetch;
	const originalDirectory = process.env.PI_CODING_AGENT_DIR;
	const originalOffline = process.env.PI_OFFLINE;
	const originalEnvSecret = process.env.PI_TEST_AUTH_JSON_KEY;
	clearPricingMemoryCache();

	const { directory } = await writeConfig(`{
		"providers": {
			"auth-env-proxy": {
				"baseUrl": "http://127.0.0.1:8318/v1",
				"headers": { "X-Tenant": "$TENANT_ID" }
			},
			"header-only-remote": {
				"baseUrl": "https://remote.example.test/v1",
				"authHeader": false,
				"headers": { "X-Api-Key": "secret-header-key" }
			}
		}
	}`);
	await writeFile(
		join(directory, "pi-openai-model-discovery.json"),
		JSON.stringify({ pricingUrl: null }),
		"utf8",
	);
	await writeFile(
		join(directory, "auth.json"),
		JSON.stringify({
			"auth-env-proxy": {
				type: "api_key",
				key: "$PI_TEST_AUTH_JSON_KEY",
				env: { TENANT_ID: "tenant-42" },
			},
		}),
		"utf8",
	);

	process.env.PI_CODING_AGENT_DIR = directory;
	process.env.PI_TEST_AUTH_JSON_KEY = "resolved-from-env";

	const seenHeaders = new Map<string, Record<string, string>>();
	globalThis.fetch = async (input, init) => {
		const url = String(input);
		seenHeaders.set(url, (init?.headers ?? {}) as Record<string, string>);
		return new Response(
			JSON.stringify({ data: [{ id: "proxy-model", context_length: 64_000 }] }),
			{ status: 200, headers: { "content-type": "application/json" } },
		);
	};

	const registrations: Registration[] = [];
	const unregistered: string[] = [];
	const handlers = new Map<string, Array<(event: { reason?: string }) => void>>();
	const pi = {
		registerProvider(name: string, config: ProviderConfig) {
			registrations.push({ name, config });
		},
		unregisterProvider(name: string) {
			unregistered.push(name);
		},
		on(event: string, handler: (event: { reason?: string }) => void) {
			const list = handlers.get(event) ?? [];
			list.push(handler);
			handlers.set(event, list);
		},
	} as unknown as ExtensionAPI;

	try {
		await openAIModelDiscovery(pi);

		assert.equal(
			seenHeaders.get("http://127.0.0.1:8318/v1/models")?.Authorization,
			"Bearer resolved-from-env",
		);
		assert.equal(
			seenHeaders.get("http://127.0.0.1:8318/v1/models")?.["X-Tenant"],
			"tenant-42",
		);

		const remoteHeaders = seenHeaders.get("https://remote.example.test/v1/models");
		assert.equal(remoteHeaders?.["X-Api-Key"], "secret-header-key");
		assert.equal(remoteHeaders?.Authorization, undefined);
		const remoteReg = registrations.find((r) => r.name === "header-only-remote");
		assert.equal(remoteReg?.config.apiKey, "header-auth");

		// /reload triggers session_shutdown(reason: "reload") and unregisters all dynamic providers
		for (const handler of handlers.get("session_shutdown") ?? []) {
			handler({ reason: "reload" });
		}
		assert.deepEqual(unregistered.sort(), [
			"auth-env-proxy",
			"header-only-remote",
		]);

		// PI_OFFLINE=1 skips startup network calls completely and restores cached models
		await writeFile(
			join(directory, "models-store.json"),
			JSON.stringify({
				"auth-env-proxy": {
					models: [
						{ id: "offline-cached", provider: "auth-env-proxy", contextWindow: 64_000 },
					],
				},
			}),
			"utf8",
		);
		process.env.PI_OFFLINE = "1";
		let offlineFetches = 0;
		globalThis.fetch = async () => {
			offlineFetches++;
			throw new Error("should not fetch while PI_OFFLINE=1");
		};
		const offlineRegs: Registration[] = [];
		await openAIModelDiscovery({
			registerProvider(name: string, config: ProviderConfig) {
				offlineRegs.push({ name, config });
			},
		} as unknown as ExtensionAPI);
		assert.equal(offlineFetches, 0);
		assert.deepEqual(
			offlineRegs
				.find((r) => r.name === "auth-env-proxy")
				?.config.models?.map((m) => m.id),
			["offline-cached"],
		);
	} finally {
		clearPricingMemoryCache();
		globalThis.fetch = originalFetch;
		if (originalDirectory === undefined) delete process.env.PI_CODING_AGENT_DIR;
		else process.env.PI_CODING_AGENT_DIR = originalDirectory;
		if (originalOffline === undefined) delete process.env.PI_OFFLINE;
		else process.env.PI_OFFLINE = originalOffline;
		if (originalEnvSecret === undefined) delete process.env.PI_TEST_AUTH_JSON_KEY;
		else process.env.PI_TEST_AUTH_JSON_KEY = originalEnvSecret;
		await rm(directory, { recursive: true, force: true });
	}
});

test("fetchPricingCatalog negative-caches failures briefly unless forceRefresh is set", async () => {
	const originalFetch = globalThis.fetch;
	clearPricingMemoryCache();
	let calls = 0;
	globalThis.fetch = async () => {
		calls++;
		if (calls === 1) return new Response("down", { status: 503 });
		return new Response(
			JSON.stringify({ "gpt-5.4": { max_input_tokens: 400_000 } }),
			{ status: 200, headers: { "content-type": "application/json" } },
		);
	};

	try {
		await captureStderr(async () => {
			const first = await fetchPricingCatalog({
				url: "https://pricing.example.test/neg.json",
			});
			assert.deepEqual(first, {});
		});
		assert.equal(calls, 1);

		// Second call within negative-cache TTL skips network
		const second = await fetchPricingCatalog({
			url: "https://pricing.example.test/neg.json",
		});
		assert.deepEqual(second, {});
		assert.equal(calls, 1);

		// forceRefresh bypasses negative cache and succeeds
		const forced = await fetchPricingCatalog({
			url: "https://pricing.example.test/neg.json",
			forceRefresh: true,
		});
		assert.equal(forced["gpt-5.4"]?.max_input_tokens, 400_000);
		assert.equal(calls, 2);
	} finally {
		clearPricingMemoryCache();
		globalThis.fetch = originalFetch;
	}
});

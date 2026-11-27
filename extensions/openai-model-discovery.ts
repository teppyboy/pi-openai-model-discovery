import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type {
	ExtensionAPI,
	ProviderConfig,
} from "@earendil-works/pi-coding-agent";
import {
	agentDir,
	asObject,
	buildDiscoveryHeaders,
	clearConfigValueCache,
	compilePatterns,
	DEFAULT_API,
	DEFAULT_CONTEXT_WINDOW,
	DEFAULT_EXCLUDE_PATTERNS,
	DEFAULT_INCLUDE_PATTERNS,
	DEFAULT_MAX_TOKENS,
	DEFAULT_PRICING_CACHE_TTL_MS,
	DEFAULT_PRICING_URL,
	DEFAULT_REQUEST_TIMEOUT_MS,
	isLoopbackUrl,
	pricingCachePath,
	readAuthStore,
	readDiscoveryConfig,
	readDynamicProviders,
	resolveConfigValue,
	resolveHeaders,
	shouldIncludeModel,
	stripJsonComments,
	type DiscoveryConfig,
	type DiscoveryDefaults,
	type DynamicProvider,
	type JsonObject,
	type ModelCost,
	type ModelCostTier,
	type ModelOverride,
	type ProviderApi,
	type StoredAuthEntry,
	type ThinkingLevel,
	type ThinkingLevelMap,
} from "../src/config.ts";
import {
	applyModelOverride,
	buildLowerPricingIndex,
	clearPricingMemoryCache,
	fetchPricingCatalog,
	lookupPricingEntry,
	mergeCompat,
	pricingToCost,
	pricingToInput,
	pricingToThinking,
	type FetchPricingOptions,
	type PricingCatalog,
	type PricingEntry,
	type PricingTierEntry,
} from "../src/pricing.ts";

export {
	DEFAULT_API,
	DEFAULT_CONTEXT_WINDOW,
	DEFAULT_EXCLUDE_PATTERNS,
	DEFAULT_INCLUDE_PATTERNS,
	DEFAULT_MAX_TOKENS,
	DEFAULT_PRICING_CACHE_TTL_MS,
	DEFAULT_PRICING_URL,
	DEFAULT_REQUEST_TIMEOUT_MS,
	agentDir,
	applyModelOverride,
	buildDiscoveryHeaders,
	buildLowerPricingIndex,
	clearConfigValueCache,
	clearPricingMemoryCache,
	compilePatterns,
	fetchPricingCatalog,
	isLoopbackUrl,
	lookupPricingEntry,
	mergeCompat,
	pricingToCost,
	pricingToInput,
	pricingToThinking,
	readAuthStore,
	readDiscoveryConfig,
	readDynamicProviders,
	resolveConfigValue,
	resolveHeaders,
	shouldIncludeModel,
	stripJsonComments,
	type DiscoveryConfig,
	type DiscoveryDefaults,
	type DynamicProvider,
	type FetchPricingOptions,
	type JsonObject,
	type ModelCost,
	type ModelCostTier,
	type ModelOverride,
	type PricingCatalog,
	type PricingEntry,
	type PricingTierEntry,
	type ProviderApi,
	type StoredAuthEntry,
	type ThinkingLevel,
	type ThinkingLevelMap,
};

type RefreshContext = Parameters<
	NonNullable<ProviderConfig["refreshModels"]>
>[0];

const DEFAULT_COST: ModelCost = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
};

const FORCED_THINKING_LEVELS: ThinkingLevelMap = {
	xhigh: "xhigh",
	max: "max",
};

const THINKING_LEVEL_KEYS = new Set<ThinkingLevel>([
	"off",
	"minimal",
	"low",
	"medium",
	"high",
	"xhigh",
	"max",
]);

export type ProviderModelDefinition = {
	id: string;
	name: string;
	api: ProviderApi;
	baseUrl: string;
	reasoning: boolean;
	thinkingLevelMap?: ThinkingLevelMap;
	input: Array<"text" | "image">;
	cost: ModelCost;
	contextWindow: number;
	maxTokens: number;
	compat?: JsonObject;
};

export type MapModelOptions = {
	pricing?: PricingEntry;
	defaults?: DiscoveryDefaults;
	override?: ModelOverride;
};

export type RefreshProviderOptions = {
	config?: DiscoveryConfig;
	pricingCatalog?: PricingCatalog;
	pricingUrl?: string | null;
	pricingCachePath?: string;
	pricingCacheTtlMs?: number;
	timeoutMs?: number;
	include?: readonly RegExp[];
	exclude?: readonly RegExp[];
	syntheticLocalKey?: boolean;
	fallbackModels?: ProviderModelDefinition[];
	persistOnCachePhase?: boolean;
	onDiscovered?: (
		models: ProviderModelDefinition[],
		persisted: boolean,
	) => void;
};

function firstBoolean(...values: unknown[]): boolean | undefined {
	return values.find((value): value is boolean => typeof value === "boolean");
}

function positiveNumber(...values: unknown[]): number | undefined {
	for (const value of values) {
		if (typeof value === "number" && Number.isFinite(value) && value > 0)
			return value;
	}
	return undefined;
}

function numberOrZero(value: unknown): number {
	return typeof value === "number" && Number.isFinite(value) && value >= 0
		? value
		: 0;
}

function validInput(value: unknown): Array<"text" | "image"> | undefined {
	let values: unknown[];
	if (Array.isArray(value)) values = value;
	else if (typeof value === "string") values = [value];
	else values = [];

	const input = [
		...new Set(
			values.flatMap((item) => {
				if (typeof item !== "string") return [];
				const normalized = item.trim().toLowerCase();
				return normalized === "text" || normalized === "image"
					? [normalized as "text" | "image"]
					: [];
			}),
		),
	];
	return input.length > 0 ? input : undefined;
}

function firstString(...values: unknown[]): string | undefined {
	for (const value of values) {
		if (typeof value === "string" && value.trim()) return value.trim();
	}
	return undefined;
}

function reasoningLevels(value: unknown): string[] {
	if (!Array.isArray(value)) return [];
	return value.flatMap((entry) => {
		const raw = typeof entry === "string" ? entry : asObject(entry)?.effort;
		return typeof raw === "string" && raw.trim()
			? [raw.trim().toLowerCase()]
			: [];
	});
}

function reasoningFromLevels(value: unknown): boolean | undefined {
	const levels = reasoningLevels(value);
	return levels.length > 0
		? levels.some((level) => level !== "none" && level !== "off")
		: undefined;
}

function thinkingLevelMapFromLevels(
	value: unknown,
): ThinkingLevelMap | undefined {
	const levels = reasoningLevels(value);
	if (levels.length === 0) return undefined;

	const map: ThinkingLevelMap = { xhigh: null, max: null };
	for (const level of levels) {
		switch (level) {
			case "none":
			case "off":
				map.off = level;
				break;
			case "minimal":
			case "low":
			case "medium":
			case "high":
			case "xhigh":
			case "max":
				map[level] = level;
				break;
		}
	}
	if (levels.includes("low") && !levels.includes("minimal")) {
		map.minimal = "low";
	}
	return map;
}

function parseExplicitThinkingLevelMap(
	value: unknown,
): ThinkingLevelMap | undefined {
	const obj = asObject(value);
	if (!obj) return undefined;
	const map: ThinkingLevelMap = {};
	let hasAny = false;
	for (const [key, entry] of Object.entries(obj)) {
		if (!THINKING_LEVEL_KEYS.has(key as ThinkingLevel)) continue;
		if (entry === null) {
			map[key as ThinkingLevel] = null;
			hasAny = true;
		} else if (typeof entry === "string" && entry.trim()) {
			map[key as ThinkingLevel] = entry.trim();
			hasAny = true;
		}
	}
	return hasAny ? map : undefined;
}

function parseCostTiers(value: unknown): ModelCostTier[] | undefined {
	if (!Array.isArray(value)) return undefined;
	const tiers: ModelCostTier[] = [];
	for (const entry of value) {
		const tier = asObject(entry);
		if (!tier) continue;
		const inputTokensAbove = positiveNumber(tier.inputTokensAbove);
		if (inputTokensAbove === undefined) continue;
		tiers.push({
			inputTokensAbove,
			input: numberOrZero(tier.input),
			output: numberOrZero(tier.output),
			cacheRead: numberOrZero(tier.cacheRead ?? tier.cache_read),
			cacheWrite: numberOrZero(tier.cacheWrite ?? tier.cache_write),
		});
	}
	if (tiers.length === 0) return undefined;
	return tiers.sort((a, b) => a.inputTokensAbove - b.inputTokensAbove);
}

function parseEndpointCost(cost: JsonObject | undefined): ModelCost | undefined {
	if (!cost) return undefined;
	const tiers = parseCostTiers(cost.tiers);
	return {
		input: numberOrZero(cost.input),
		output: numberOrZero(cost.output),
		cacheRead: numberOrZero(cost.cacheRead ?? cost.cache_read),
		cacheWrite: numberOrZero(cost.cacheWrite ?? cost.cache_write),
		...(tiers ? { tiers } : {}),
	};
}

function explicitInputFromModel(
	model: JsonObject | undefined,
	capabilities: JsonObject | undefined,
): Array<"text" | "image"> | undefined {
	const candidates = [
		model?.input,
		model?.input_types,
		model?.input_modalities,
		model?.supportedInputModalities,
		model?.supported_input_modalities,
		capabilities?.input_modalities,
		capabilities?.inputModalities,
	];
	const explicit = candidates
		.map(validInput)
		.find((value): value is Array<"text" | "image"> => value !== undefined);
	if (model?.vision === true || capabilities?.vision === true) {
		const augmented: Array<"text" | "image"> = explicit
			? [...explicit]
			: ["text"];
		if (!augmented.includes("image")) augmented.push("image");
		return augmented;
	}
	return explicit;
}

function extractRecordId(record: unknown): string {
	const model = asObject(record);
	return (
		firstString(model?.id, model?.slug, model?.model, model?.name) ?? ""
	);
}

export function mapModelRecord(
	record: unknown,
	provider: DynamicProvider,
	options?: MapModelOptions,
): ProviderModelDefinition {
	const model = asObject(record);
	const id = extractRecordId(model);
	if (!id)
		throw new Error(
			`Model discovery failed for ${provider.id}: every model needs a non-empty id`,
		);

	const capabilities = asObject(model?.capabilities);
	const pricing = options?.pricing;
	const defaults = options?.defaults;

	const endpointCost = parseEndpointCost(asObject(model?.cost));
	const cost: ModelCost =
		endpointCost ?? (pricing ? pricingToCost(pricing) : { ...DEFAULT_COST });

	const endpointInput = explicitInputFromModel(model, capabilities);
	const input: Array<"text" | "image"> =
		endpointInput ??
		pricingToInput(pricing) ??
		(defaults?.input ? [...defaults.input] : ["text"]);

	const storedThinkingMap = parseExplicitThinkingLevelMap(
		model?.thinkingLevelMap,
	);
	const levels =
		model?.supported_reasoning_levels ??
		model?.reasoning_levels ??
		capabilities?.supported_reasoning_levels;
	const levelMap = thinkingLevelMapFromLevels(levels);
	const explicitReasoning = firstBoolean(
		model?.reasoning,
		model?.supports_reasoning,
		capabilities?.reasoning,
	);
	const pricingThinking = pricingToThinking(pricing);

	let reasoning: boolean;
	let thinkingLevelMap: ThinkingLevelMap | undefined;

	if (storedThinkingMap) {
		reasoning = explicitReasoning ?? true;
		thinkingLevelMap = reasoning ? storedThinkingMap : undefined;
	} else if (levelMap) {
		reasoning = explicitReasoning ?? reasoningFromLevels(levels) ?? true;
		thinkingLevelMap = reasoning ? levelMap : undefined;
	} else if (explicitReasoning !== undefined) {
		reasoning = explicitReasoning;
		if (reasoning) {
			thinkingLevelMap = pricingThinking?.thinkingLevelMap
				? { ...pricingThinking.thinkingLevelMap }
				: defaults?.thinkingLevelMap
					? { ...defaults.thinkingLevelMap }
					: { ...FORCED_THINKING_LEVELS };
		}
	} else if (pricingThinking) {
		reasoning = pricingThinking.reasoning;
		thinkingLevelMap = pricingThinking.thinkingLevelMap
			? { ...pricingThinking.thinkingLevelMap }
			: undefined;
	} else {
		reasoning = defaults?.reasoning ?? true;
		if (reasoning) {
			thinkingLevelMap = defaults?.thinkingLevelMap
				? { ...defaults.thinkingLevelMap }
				: { ...FORCED_THINKING_LEVELS };
		}
	}

	const compat = mergeCompat(
		mergeCompat(defaults?.compat, provider.compat),
		asObject(model?.compat),
	);

	const baseModel: ProviderModelDefinition = {
		id,
		name:
			firstString(model?.name, model?.display_name, model?.displayName) ?? id,
		api: provider.api,
		baseUrl: provider.baseUrl,
		reasoning,
		...(thinkingLevelMap ? { thinkingLevelMap } : {}),
		input,
		cost,
		contextWindow:
			positiveNumber(
				model?.contextWindow,
				model?.context_window,
				model?.context_length,
				model?.max_context_window,
				model?.max_model_len,
				capabilities?.contextWindow,
				capabilities?.context_window,
				capabilities?.contextLength,
				capabilities?.context_length,
				pricing?.max_input_tokens,
				defaults?.contextWindow,
			) ?? DEFAULT_CONTEXT_WINDOW,
		maxTokens:
			positiveNumber(
				model?.maxTokens,
				model?.max_tokens,
				model?.max_output_tokens,
				model?.max_completion_tokens,
				model?.max_output,
				capabilities?.maxOutput,
				capabilities?.max_output,
				capabilities?.maxTokens,
				capabilities?.max_tokens,
				pricing?.max_output_tokens,
				pricing?.max_tokens,
				defaults?.maxTokens,
			) ?? DEFAULT_MAX_TOKENS,
		...(compat ? { compat } : {}),
	};

	const overridden = applyModelOverride(baseModel, options?.override);
	if (
		options?.override?.reasoning === true &&
		overridden.reasoning &&
		overridden.thinkingLevelMap === undefined
	) {
		overridden.thinkingLevelMap = defaults?.thinkingLevelMap
			? { ...defaults.thinkingLevelMap }
			: { ...FORCED_THINKING_LEVELS };
	}
	return overridden;
}

function storedModels(
	stored: unknown,
	provider: DynamicProvider,
): ProviderModelDefinition[] {
	const models = asObject(stored)?.models;
	return Array.isArray(models)
		? models
				.filter((model) => asObject(model)?.provider === provider.id)
				.map((model) => mapModelRecord(model, provider))
		: [];
}

export function modelDiscoveryUrl(baseUrl: string): URL {
	let url: URL;
	try {
		url = new URL(baseUrl);
	} catch {
		throw new Error(
			`Model discovery requires a valid HTTP(S) base URL, got ${baseUrl}`,
		);
	}
	if (url.protocol !== "http:" && url.protocol !== "https:") {
		throw new Error(
			`Model discovery requires an HTTP(S) base URL, got ${url.protocol}`,
		);
	}
	if (!url.hostname || url.username || url.password) {
		throw new Error(
			"Model discovery base URL must have a hostname and no embedded credentials",
		);
	}
	url.pathname = `${url.pathname.replace(/\/+$/, "")}/models`;
	url.search = "";
	url.hash = "";
	return url;
}

function extractModelRecords(payload: unknown): unknown[] | undefined {
	if (Array.isArray(payload)) return payload;
	const root = asObject(payload);
	if (!root) return undefined;
	for (const key of ["data", "models", "results"]) {
		if (Array.isArray(root[key])) return root[key];
	}
	return undefined;
}

function hasModelMetadata(record: unknown): boolean {
	const model = asObject(record);
	if (!model) return false;
	return (
		[
			"slug",
			"name",
			"display_name",
			"context_window",
			"context_length",
			"max_context_window",
			"max_completion_tokens",
			"max_output_tokens",
			"input_modalities",
			"supportedInputModalities",
			"supported_reasoning_levels",
			"reasoning",
			"supports_reasoning",
		].some((key) => key in model) || asObject(model.capabilities) !== undefined
	);
}

function withTimeoutSignal(
	timeoutMs: number,
	signal?: AbortSignal,
): AbortSignal {
	const timeoutSignal = AbortSignal.timeout(timeoutMs);
	return signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;
}

async function fetchCatalog(
	url: URL,
	headers: Record<string, string>,
	signal: AbortSignal,
): Promise<unknown> {
	const response = await fetch(url, { signal, headers });
	if (!response.ok) {
		throw new Error(`Model discovery failed: HTTP ${response.status}`);
	}
	return response.json();
}

async function discoverModelRecords(
	discoveryUrl: URL,
	headers: Record<string, string>,
	signal: AbortSignal,
): Promise<unknown[]> {
	const payload = await fetchCatalog(discoveryUrl, headers, signal);
	const records = extractModelRecords(payload);
	if (!records) {
		throw new Error("Model discovery failed: response has no model list");
	}
	if (records.length === 0 || records.some(hasModelMetadata)) return records;

	// CLIProxyAPI exposes its richer Codex catalog behind this query parameter;
	// ordinary OpenAI-compatible servers generally ignore the extra parameter.
	let metadataUrl: URL;
	try {
		metadataUrl = new URL(discoveryUrl.toString());
	} catch {
		return records;
	}
	metadataUrl.searchParams.set("client_version", "pi");
	try {
		const metadataPayload = await fetchCatalog(metadataUrl, headers, signal);
		const metadataRecords = extractModelRecords(metadataPayload);
		if (metadataRecords && metadataRecords.length > 0) return metadataRecords;
	} catch {
		// Keep the standard catalog when the optional metadata request is unsupported.
	}
	return records;
}

function mapAndFilterRecords(
	records: unknown[],
	provider: DynamicProvider,
	options: {
		pricingCatalog?: PricingCatalog;
		lowerPricingIndex?: Map<string, PricingEntry>;
		defaults?: DiscoveryDefaults;
		overrides?: Record<string, ModelOverride>;
		include?: readonly RegExp[];
		exclude?: readonly RegExp[];
	},
): ProviderModelDefinition[] {
	const include = options.include ?? [];
	const exclude = options.exclude ?? [];
	const seen = new Set<string>();
	const definitions: ProviderModelDefinition[] = [];

	for (const record of records) {
		const id = extractRecordId(record);
		if (!id || seen.has(id)) continue;
		if (!shouldIncludeModel(id, include, exclude)) continue;
		seen.add(id);

		const pricing = options.pricingCatalog
			? lookupPricingEntry(
					options.pricingCatalog,
					id,
					options.lowerPricingIndex,
				)
			: undefined;
		const override =
			options.overrides && Object.hasOwn(options.overrides, id)
				? options.overrides[id]
				: undefined;
		definitions.push(
			mapModelRecord(record, provider, {
				pricing,
				defaults: options.defaults,
				override,
			}),
		);
	}

	return definitions;
}

function extractContextCredentialKey(
	credential: RefreshContext["credential"],
	syntheticLocalKey?: boolean,
): string | undefined {
	if (!credential) return undefined;
	if (credential.type === "api_key" && credential.key) {
		if (
			syntheticLocalKey &&
			(credential.key === "local" || credential.key === "header-auth")
		) {
			return undefined;
		}
		return credential.key;
	}
	if (credential.type === "oauth") {
		const access = (credential as { access?: unknown }).access;
		if (typeof access === "string" && access.length > 0) return access;
	}
	return undefined;
}

function extractContextCredentialEnv(
	credential: RefreshContext["credential"],
): Record<string, string> | undefined {
	if (!credential) return undefined;
	const raw = asObject((credential as { env?: unknown }).env);
	if (!raw) return undefined;
	const env: Record<string, string> = {};
	for (const [key, value] of Object.entries(raw)) {
		if (typeof value === "string") env[key] = value;
	}
	return Object.keys(env).length > 0 ? env : undefined;
}

function toPersistedModels(
	provider: DynamicProvider,
	models: ProviderModelDefinition[],
) {
	return {
		checkedAt: Date.now(),
		models: models.map((model) => ({
			...model,
			provider: provider.id,
		})),
	};
}

export async function refreshProvider(
	provider: DynamicProvider,
	context: RefreshContext,
	options?: RefreshProviderOptions,
): Promise<ProviderModelDefinition[]> {
	const cached = storedModels(context.stored, provider);
	const fallbackModels = options?.fallbackModels ?? [];
	const fallback =
		options?.persistOnCachePhase && fallbackModels.length > 0
			? fallbackModels
			: cached.length > 0
				? cached
				: fallbackModels;

	if (!context.allowNetwork) {
		if (
			options?.persistOnCachePhase &&
			fallbackModels.length > 0
		) {
			try {
				const persisted = await context.publish({
					persist: toPersistedModels(provider, fallbackModels),
				});
				options.onDiscovered?.(fallbackModels, persisted);
			} catch {
				// Best-effort persistence of startup-discovered catalog; still return models.
			}
			return fallbackModels;
		}
		return fallback;
	}

	try {
		const timeoutMs =
			options?.timeoutMs ??
			options?.config?.requestTimeoutMs ??
			DEFAULT_REQUEST_TIMEOUT_MS;
		const requestSignal = withTimeoutSignal(timeoutMs, context.signal);
		const headers = buildDiscoveryHeaders(provider, {
			credentialKey: extractContextCredentialKey(
				context.credential,
				options?.syntheticLocalKey,
			),
			env: extractContextCredentialEnv(context.credential),
		});

		const shouldFetchPricing =
			options?.pricingCatalog === undefined &&
			options?.pricingUrl !== undefined &&
			options.pricingUrl !== null;

		const [records, pricingCatalog] = await Promise.all([
			discoverModelRecords(
				modelDiscoveryUrl(provider.baseUrl),
				headers,
				requestSignal,
			),
			shouldFetchPricing
				? fetchPricingCatalog({
						url: options.pricingUrl,
						timeoutMs,
						cachePath: options.pricingCachePath,
						cacheTtlMs: options.pricingCacheTtlMs,
						signal: context.signal,
						forceRefresh: context.force,
					})
				: Promise.resolve(options?.pricingCatalog),
		]);

		const lowerPricingIndex = pricingCatalog
			? buildLowerPricingIndex(pricingCatalog)
			: undefined;

		const definitions = mapAndFilterRecords(records, provider, {
			pricingCatalog,
			lowerPricingIndex,
			defaults: options?.config?.defaults,
			overrides: options?.config?.overrides,
			include: options?.include,
			exclude: options?.exclude,
		});

		if (definitions.length === 0) {
			throw new Error("Model discovery returned no models");
		}

		const persisted = await context.publish({
			persist: toPersistedModels(provider, definitions),
		});
		options?.onDiscovered?.(definitions, persisted);
		return definitions;
	} catch (error) {
		if (context.signal.aborted || fallback.length === 0) throw error;
		process.stderr.write(
			`OpenAI model discovery failed for ${provider.id}; using cached models: ${String(error)}\n`,
		);
		return fallback;
	}
}

const STARTUP_PRICING_MAX_TIMEOUT_MS = 5_000;

function isOfflineMode(): boolean {
	const raw = process.env.PI_OFFLINE?.trim().toLowerCase();
	return raw === "1" || raw === "true";
}

function ownProperty<T>(
	record: Record<string, T> | undefined,
	key: string,
): T | undefined {
	return record && Object.hasOwn(record, key) ? record[key] : undefined;
}

function startupCredentialKey(
	provider: DynamicProvider,
	authEntry: StoredAuthEntry | undefined,
): string | undefined {
	if (authEntry?.type === "api_key" && authEntry.key) {
		return resolveConfigValue(authEntry.key, authEntry.env);
	}
	// When auth.json has a stale oauth entry AND models.json configures apiKey,
	// let buildDiscoveryHeaders resolve provider.apiKey instead of the stale oauth token.
	if (provider.apiKey !== undefined) {
		return undefined;
	}
	if (authEntry?.type === "oauth" && authEntry.access) {
		return authEntry.access;
	}
	return undefined;
}

export default async function openAIModelDiscovery(
	pi: ExtensionAPI,
): Promise<void> {
	const config = await readDiscoveryConfig();
	const providers = await readDynamicProviders(undefined, config);
	if (providers.length === 0) return;

	const currentAgentDir = agentDir();
	let cache: JsonObject = {};
	try {
		cache =
			asObject(
				JSON.parse(
					await readFile(join(currentAgentDir, "models-store.json"), "utf8"),
				),
			) ?? {};
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code !== "ENOENT") {
			process.stderr.write(
				`OpenAI model discovery could not read cached models: ${String(error)}\n`,
			);
		}
	}

	const authStore = await readAuthStore(join(currentAgentDir, "auth.json"));
	const timeoutMs = config.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS;
	const pricingUrl =
		config.pricingUrl === undefined ? DEFAULT_PRICING_URL : config.pricingUrl;
	const resolvedPricingCachePath = pricingCachePath();
	const pricingCacheTtlMs =
		config.pricingCacheTtlMs ?? DEFAULT_PRICING_CACHE_TTL_MS;
	const include = compilePatterns(config.include, DEFAULT_INCLUDE_PATTERNS);
	const exclude = compilePatterns(config.exclude, DEFAULT_EXCLUDE_PATTERNS);
	const offline = isOfflineMode();

	for (const provider of providers) {
		const authEntry = ownProperty(authStore, provider.id);
		if (authEntry?.type === "oauth" && provider.apiKey !== undefined) {
			process.stderr.write(
				`OpenAI model discovery warning for ${provider.id}: auth.json contains a stored "oauth" credential that shadows models.json apiKey in Pi and can hide models or block requests. Run "/logout ${provider.id}" or remove "${provider.id}" from auth.json to use models.json apiKey.\n`,
			);
		}
	}

	const pricingPromise =
		!offline && pricingUrl !== null
			? fetchPricingCatalog({
					url: pricingUrl,
					timeoutMs: Math.min(timeoutMs, STARTUP_PRICING_MAX_TIMEOUT_MS),
					cachePath: resolvedPricingCachePath,
					cacheTtlMs: pricingCacheTtlMs,
				})
			: Promise.resolve<PricingCatalog>({});

	const rawStartupResults = await Promise.all(
		providers.map(async (provider) => {
			const cached = storedModels(ownProperty(cache, provider.id), provider);
			const authEntry = ownProperty(authStore, provider.id);
			if (offline) {
				return { provider, cached, authEntry, records: undefined };
			}
			const headers = buildDiscoveryHeaders(provider, {
				credentialKey: startupCredentialKey(provider, authEntry),
				env: authEntry?.env,
			});
			try {
				const records = await discoverModelRecords(
					modelDiscoveryUrl(provider.baseUrl),
					headers,
					AbortSignal.timeout(timeoutMs),
				);
				return { provider, cached, authEntry, records };
			} catch (error) {
				if (cached.length > 0) {
					process.stderr.write(
						`OpenAI model discovery failed on startup for ${provider.id}; using cached models: ${String(error)}\n`,
					);
				} else {
					process.stderr.write(
						`OpenAI model discovery failed on startup for ${provider.id}: ${String(error)}\n`,
					);
				}
				return { provider, cached, authEntry, records: undefined };
			}
		}),
	);

	const pricingCatalog = await pricingPromise;
	const lowerPricingIndex = buildLowerPricingIndex(pricingCatalog);
	const registeredIds: string[] = [];

	for (const { provider, cached, authEntry, records } of rawStartupResults) {
		let discovered: ProviderModelDefinition[] = [];
		if (records && records.length > 0) {
			discovered = mapAndFilterRecords(records, provider, {
				pricingCatalog,
				lowerPricingIndex,
				defaults: config.defaults,
				overrides: config.overrides,
				include,
				exclude,
			});
		}

		const startupSucceeded = discovered.length > 0;
		const initialModels = startupSucceeded ? discovered : cached;
		const hasHeaderOnlyAuth =
			!provider.apiKey &&
			!authEntry &&
			provider.authHeader === false &&
			Object.keys(resolveHeaders(provider.headers)).length > 0;
		const syntheticLoopback =
			!provider.apiKey && !authEntry && isLoopbackUrl(provider.baseUrl);
		const syntheticKey = syntheticLoopback
			? "local"
			: hasHeaderOnlyAuth
				? "header-auth"
				: undefined;
		const syntheticLocalKey = syntheticKey !== undefined;

		const state = {
			inMemoryModels: initialModels,
			needsPersist: startupSucceeded,
		};

		registeredIds.push(provider.id);
		pi.registerProvider(provider.id, {
			baseUrl: provider.baseUrl,
			api: provider.api,
			...(provider.name ? { name: provider.name } : {}),
			...(syntheticKey !== undefined ? { apiKey: syntheticKey } : {}),
			...(initialModels.length > 0 ? { models: initialModels } : {}),
			refreshModels: (context: RefreshContext) =>
				refreshProvider(provider, context, {
					config,
					pricingUrl,
					pricingCachePath: resolvedPricingCachePath,
					pricingCacheTtlMs,
					timeoutMs,
					include,
					exclude,
					syntheticLocalKey,
					fallbackModels: state.inMemoryModels,
					persistOnCachePhase: state.needsPersist,
					onDiscovered(models, persisted) {
						state.inMemoryModels = models;
						if (persisted) {
							state.needsPersist = false;
						}
					},
				}),
		});
	}

	if (
		registeredIds.length > 0 &&
		typeof pi.on === "function" &&
		typeof pi.unregisterProvider === "function"
	) {
		pi.on("session_shutdown", (event) => {
			if (event.reason === "reload") {
				for (const id of registeredIds) {
					pi.unregisterProvider(id);
				}
			}
		});
	}
}

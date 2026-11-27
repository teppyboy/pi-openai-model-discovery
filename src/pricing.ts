import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type {
	JsonObject,
	ModelCost,
	ModelCostTier,
	ModelOverride,
	ThinkingLevelMap,
} from "./config.ts";

export type PricingTierEntry = {
	range?: unknown;
	input_cost_per_token?: number;
	output_cost_per_token?: number;
	cache_read_input_token_cost?: number;
	cache_creation_input_token_cost?: number;
};

export type PricingEntry = {
	max_input_tokens?: number;
	max_output_tokens?: number;
	max_tokens?: number;
	input_cost_per_token?: number;
	output_cost_per_token?: number;
	cache_read_input_token_cost?: number;
	cache_creation_input_token_cost?: number;
	input_cost_per_token_above_128k_tokens?: number;
	output_cost_per_token_above_128k_tokens?: number;
	cache_read_input_token_cost_above_128k_tokens?: number;
	cache_creation_input_token_cost_above_128k_tokens?: number;
	input_cost_per_token_above_200k_tokens?: number;
	output_cost_per_token_above_200k_tokens?: number;
	cache_read_input_token_cost_above_200k_tokens?: number;
	cache_creation_input_token_cost_above_200k_tokens?: number;
	input_cost_per_token_above_272k_tokens?: number;
	output_cost_per_token_above_272k_tokens?: number;
	cache_read_input_token_cost_above_272k_tokens?: number;
	cache_creation_input_token_cost_above_272k_tokens?: number;
	long_context_input_token_threshold?: number;
	long_context_input_cost_multiplier?: number;
	long_context_output_cost_multiplier?: number;
	long_context_cache_read_cost_multiplier?: number;
	long_context_cache_creation_cost_multiplier?: number;
	tiered_pricing?: PricingTierEntry[];
	supported_modalities?: string[];
	supported_input_modalities?: string[];
	supports_vision?: boolean;
	supports_reasoning?: boolean;
	supports_none_reasoning_effort?: boolean;
	supports_minimal_reasoning_effort?: boolean;
	supports_xhigh_reasoning_effort?: boolean;
	supports_max_reasoning_effort?: boolean;
};

export type PricingCatalog = Record<string, PricingEntry>;

export type FetchPricingOptions = {
	url?: string | null;
	timeoutMs?: number;
	cachePath?: string;
	cacheTtlMs?: number;
	signal?: AbortSignal;
	forceRefresh?: boolean;
};

type CachedPricingRecord = {
	fetchedAt: number;
	catalog: PricingCatalog;
};

const DEFAULT_TIMEOUT_MS = 15_000;
const DEFAULT_CACHE_TTL_MS = 6 * 60 * 60 * 1000;
const NEGATIVE_CACHE_TTL_MS = 60 * 1000;
const DEFAULT_LONG_CONTEXT_THRESHOLD = 272_000;
const DATE_SUFFIX_RE = /(?:-\d{8}|-\d{4}-\d{2}-\d{2})$/;
const NESTED_COMPAT_KEYS = [
	"openRouterRouting",
	"vercelGatewayRouting",
	"chatTemplateKwargs",
	"chatTemplateArgs",
] as const;
const IGNORED_CATALOG_KEYS = new Set([
	"sample_spec",
	"$schema",
	"__proto__",
	"constructor",
	"prototype",
]);

const memoryCache = new Map<string, CachedPricingRecord>();
const failedFetchCache = new Map<string, CachedPricingRecord>();

function asObject(value: unknown): JsonObject | undefined {
	return typeof value === "object" && value !== null && !Array.isArray(value)
		? (value as JsonObject)
		: undefined;
}

function isNonNegativeFinite(value: unknown): value is number {
	return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function isPositiveFinite(value: unknown): value is number {
	return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function perMillion(perToken: unknown): number {
	return isNonNegativeFinite(perToken)
		? Number((perToken * 1_000_000).toPrecision(12))
		: 0;
}

function warnPricing(message: string): void {
	const singleLine = message.replace(/\r?\n/g, " ").trim();
	process.stderr.write(`Pricing catalog warning: ${singleLine}\n`);
}

function stripPrefix(id: string): string {
	const slashIndex = id.lastIndexOf("/");
	return slashIndex !== -1 ? id.slice(slashIndex + 1) : "";
}

function stripVariant(id: string): string[] {
	const slashIndex = id.lastIndexOf("/");
	const lastColon = id.lastIndexOf(":");
	if (lastColon <= slashIndex) return [];
	const firstColon = id.indexOf(":", slashIndex + 1);
	const candidates: string[] = [];
	const fromLast = id.slice(0, lastColon);
	if (fromLast) candidates.push(fromLast);
	if (firstColon > slashIndex && firstColon !== lastColon) {
		const fromFirst = id.slice(0, firstColon);
		if (fromFirst) candidates.push(fromFirst);
	}
	return candidates;
}

function stripDateSuffix(id: string): string {
	const stripped = id.replace(DATE_SUFFIX_RE, "");
	return stripped !== id ? stripped : "";
}

export function buildLowerPricingIndex(
	catalog: PricingCatalog,
): Map<string, PricingEntry> {
	const index = new Map<string, PricingEntry>();
	for (const [key, entry] of Object.entries(catalog)) {
		if (!key || entry === undefined) continue;
		const lower = key.toLowerCase();
		if (!index.has(lower)) {
			index.set(lower, entry);
		}
	}
	return index;
}

export function lookupPricingEntry(
	catalog: PricingCatalog,
	modelId: string,
	lowerIndex?: Map<string, PricingEntry>,
): PricingEntry | undefined {
	const id = modelId.trim();
	if (!id) return undefined;

	let resolvedLowerIndex = lowerIndex;
	const getLowerIndex = (): Map<string, PricingEntry> => {
		resolvedLowerIndex ??= buildLowerPricingIndex(catalog);
		return resolvedLowerIndex;
	};

	const tryMatch = (candidate: string): PricingEntry | undefined => {
		if (!candidate) return undefined;
		if (Object.hasOwn(catalog, candidate) && catalog[candidate] !== undefined) {
			return catalog[candidate];
		}
		return getLowerIndex().get(candidate.toLowerCase());
	};

	// 1. Exact match & 2. Lowercase match
	const direct = tryMatch(id);
	if (direct) return direct;

	// 3. Strip provider/namespace prefix after last "/"
	const withoutPrefix = stripPrefix(id);
	if (withoutPrefix) {
		const hit = tryMatch(withoutPrefix);
		if (hit) return hit;
	}

	// 4. Strip ":variant" suffix (full ID, then prefix+variant stripped)
	const withoutVariantList = stripVariant(id);
	const withoutPrefixAndVariantList = withoutPrefix
		? stripVariant(withoutPrefix)
		: [];
	for (const candidate of [
		...withoutVariantList,
		...withoutPrefixAndVariantList,
	]) {
		const hit = tryMatch(candidate);
		if (hit) return hit;
	}

	// 5. Strip trailing date suffix (-YYYYMMDD or -YYYY-MM-DD) as fallback
	const dateBases = [
		id,
		...withoutVariantList,
		...(withoutPrefix ? [withoutPrefix] : []),
		...withoutPrefixAndVariantList,
	];
	const seen = new Set<string>();
	for (const base of dateBases) {
		const stripped = stripDateSuffix(base);
		if (!stripped || seen.has(stripped)) continue;
		seen.add(stripped);
		const hit = tryMatch(stripped);
		if (hit) return hit;
	}

	return undefined;
}

function resolveLegacyTierRate(
	explicitAbove: unknown,
	basePerToken: unknown,
	multiplier: unknown,
	basePerMillion: number,
): number {
	if (isNonNegativeFinite(explicitAbove)) {
		return perMillion(explicitAbove);
	}
	if (isPositiveFinite(multiplier) && isNonNegativeFinite(basePerToken)) {
		return perMillion(basePerToken * multiplier);
	}
	return basePerMillion;
}

export function pricingToCost(entry: PricingEntry | undefined): ModelCost {
	const base: ModelCost = {
		input: perMillion(entry?.input_cost_per_token),
		output: perMillion(entry?.output_cost_per_token),
		cacheRead: perMillion(entry?.cache_read_input_token_cost),
		cacheWrite: perMillion(entry?.cache_creation_input_token_cost),
	};

	if (!entry) return base;

	let tiers: ModelCostTier[] = [];

	if (Array.isArray(entry.tiered_pricing)) {
		const byThreshold = new Map<number, ModelCostTier>();
		for (const rawItem of entry.tiered_pricing) {
			const item = asObject(rawItem);
			if (!item || !Array.isArray(item.range)) continue;
			const threshold = item.range[0];
			if (!isPositiveFinite(threshold)) continue;

			byThreshold.set(threshold, {
				inputTokensAbove: threshold,
				input: isNonNegativeFinite(item.input_cost_per_token)
					? perMillion(item.input_cost_per_token)
					: base.input,
				output: isNonNegativeFinite(item.output_cost_per_token)
					? perMillion(item.output_cost_per_token)
					: base.output,
				cacheRead: isNonNegativeFinite(item.cache_read_input_token_cost)
					? perMillion(item.cache_read_input_token_cost)
					: base.cacheRead,
				cacheWrite: isNonNegativeFinite(item.cache_creation_input_token_cost)
					? perMillion(item.cache_creation_input_token_cost)
					: base.cacheWrite,
			});
		}
		tiers = [...byThreshold.values()].sort(
			(a, b) => a.inputTokensAbove - b.inputTokensAbove,
		);
	}

	if (tiers.length === 0) {
		const byThreshold = new Map<number, ModelCostTier>();

		const namedThresholds: Array<{
			threshold: number;
			input: unknown;
			output: unknown;
			cacheRead: unknown;
			cacheWrite: unknown;
		}> = [
			{
				threshold: 128_000,
				input: entry.input_cost_per_token_above_128k_tokens,
				output: entry.output_cost_per_token_above_128k_tokens,
				cacheRead: entry.cache_read_input_token_cost_above_128k_tokens,
				cacheWrite: entry.cache_creation_input_token_cost_above_128k_tokens,
			},
			{
				threshold: 200_000,
				input: entry.input_cost_per_token_above_200k_tokens,
				output: entry.output_cost_per_token_above_200k_tokens,
				cacheRead: entry.cache_read_input_token_cost_above_200k_tokens,
				cacheWrite: entry.cache_creation_input_token_cost_above_200k_tokens,
			},
		];

		for (const item of namedThresholds) {
			if (
				[item.input, item.output, item.cacheRead, item.cacheWrite].some(
					isNonNegativeFinite,
				)
			) {
				byThreshold.set(item.threshold, {
					inputTokensAbove: item.threshold,
					input: isNonNegativeFinite(item.input)
						? perMillion(item.input)
						: base.input,
					output: isNonNegativeFinite(item.output)
						? perMillion(item.output)
						: base.output,
					cacheRead: isNonNegativeFinite(item.cacheRead)
						? perMillion(item.cacheRead)
						: base.cacheRead,
					cacheWrite: isNonNegativeFinite(item.cacheWrite)
						? perMillion(item.cacheWrite)
						: base.cacheWrite,
				});
			}
		}

		const hasLegacyTier =
			[
				entry.input_cost_per_token_above_272k_tokens,
				entry.output_cost_per_token_above_272k_tokens,
				entry.cache_read_input_token_cost_above_272k_tokens,
				entry.cache_creation_input_token_cost_above_272k_tokens,
			].some(isNonNegativeFinite) ||
			[
				entry.long_context_input_cost_multiplier,
				entry.long_context_output_cost_multiplier,
				entry.long_context_cache_read_cost_multiplier,
				entry.long_context_cache_creation_cost_multiplier,
			].some(isPositiveFinite);

		if (hasLegacyTier) {
			const inputTokensAbove = isPositiveFinite(
				entry.long_context_input_token_threshold,
			)
				? entry.long_context_input_token_threshold
				: DEFAULT_LONG_CONTEXT_THRESHOLD;

			byThreshold.set(inputTokensAbove, {
				inputTokensAbove,
				input: resolveLegacyTierRate(
					entry.input_cost_per_token_above_272k_tokens,
					entry.input_cost_per_token,
					entry.long_context_input_cost_multiplier,
					base.input,
				),
				output: resolveLegacyTierRate(
					entry.output_cost_per_token_above_272k_tokens,
					entry.output_cost_per_token,
					entry.long_context_output_cost_multiplier,
					base.output,
				),
				cacheRead: resolveLegacyTierRate(
					entry.cache_read_input_token_cost_above_272k_tokens,
					entry.cache_read_input_token_cost,
					entry.long_context_cache_read_cost_multiplier,
					base.cacheRead,
				),
				cacheWrite: resolveLegacyTierRate(
					entry.cache_creation_input_token_cost_above_272k_tokens,
					entry.cache_creation_input_token_cost,
					entry.long_context_cache_creation_cost_multiplier,
					base.cacheWrite,
				),
			});
		}

		tiers = [...byThreshold.values()].sort(
			(a, b) => a.inputTokensAbove - b.inputTokensAbove,
		);
	}

	return tiers.length > 0 ? { ...base, tiers } : base;
}

export function pricingToThinking(
	entry: PricingEntry | undefined,
): { reasoning: boolean; thinkingLevelMap?: ThinkingLevelMap } | undefined {
	if (!entry) return undefined;
	if (entry.supports_reasoning !== true) return { reasoning: false };
	return {
		reasoning: true,
		thinkingLevelMap: {
			off: entry.supports_none_reasoning_effort ? "none" : null,
			minimal: entry.supports_minimal_reasoning_effort ? "minimal" : null,
			low: "low",
			medium: "medium",
			high: "high",
			xhigh: entry.supports_xhigh_reasoning_effort ? "xhigh" : null,
			max: entry.supports_max_reasoning_effort ? "max" : null,
		},
	};
}

export function pricingToInput(
	entry: PricingEntry | undefined,
): Array<"text" | "image"> | undefined {
	if (!entry) return undefined;

	const modalities = new Set<string>();
	for (const list of [
		entry.supported_modalities,
		entry.supported_input_modalities,
	]) {
		if (!Array.isArray(list)) continue;
		for (const item of list) {
			if (typeof item === "string") {
				const normalized = item.trim().toLowerCase();
				if (normalized) modalities.add(normalized);
			}
		}
	}

	if (modalities.has("image") || entry.supports_vision === true) {
		return ["text", "image"];
	}
	if (modalities.has("text")) {
		return ["text"];
	}
	return undefined;
}

export function mergeCompat(
	base: JsonObject | undefined,
	override: JsonObject | undefined,
): JsonObject | undefined {
	if (!base && !override) return undefined;
	if (!base) return { ...override };
	if (!override) return { ...base };

	const merged: JsonObject = { ...base, ...override };
	for (const key of NESTED_COMPAT_KEYS) {
		const baseNested = asObject(base[key]);
		const overrideNested = asObject(override[key]);
		if (baseNested && overrideNested) {
			merged[key] = { ...baseNested, ...overrideNested };
		} else if (overrideNested) {
			merged[key] = { ...overrideNested };
		} else if (baseNested && override[key] === undefined) {
			merged[key] = { ...baseNested };
		}
	}
	return merged;
}

export function applyModelOverride<
	T extends {
		id: string;
		name: string;
		reasoning: boolean;
		thinkingLevelMap?: ThinkingLevelMap;
		input: Array<"text" | "image">;
		cost: ModelCost;
		contextWindow: number;
		maxTokens: number;
		compat?: JsonObject;
	},
>(model: T, override: ModelOverride | undefined): T {
	if (!override) return model;

	const reasoning = override.reasoning ?? model.reasoning;
	const mergedThinkingLevelMap = !reasoning
		? undefined
		: override.thinkingLevelMap
			? { ...model.thinkingLevelMap, ...override.thinkingLevelMap }
			: model.thinkingLevelMap;
	const mergedTiers = override.cost?.tiers ?? model.cost.tiers;
	const mergedCost: ModelCost = override.cost
		? {
				input: override.cost.input ?? model.cost.input,
				output: override.cost.output ?? model.cost.output,
				cacheRead: override.cost.cacheRead ?? model.cost.cacheRead,
				cacheWrite: override.cost.cacheWrite ?? model.cost.cacheWrite,
				...(mergedTiers !== undefined ? { tiers: mergedTiers } : {}),
			}
		: model.cost;
	const mergedCompat = mergeCompat(model.compat, override.compat);

	const result: T = {
		...model,
		id: model.id,
		name: override.name ?? model.name,
		reasoning,
		input: override.input ? [...override.input] : model.input,
		cost: mergedCost,
		contextWindow: override.contextWindow ?? model.contextWindow,
		maxTokens: override.maxTokens ?? model.maxTokens,
	};

	if (mergedThinkingLevelMap !== undefined) {
		result.thinkingLevelMap = mergedThinkingLevelMap;
	} else {
		delete result.thinkingLevelMap;
	}

	if (mergedCompat !== undefined) {
		result.compat = mergedCompat;
	} else {
		delete result.compat;
	}

	return result;
}

function sanitizePricingCatalog(raw: JsonObject): PricingCatalog {
	const catalog: PricingCatalog = {};
	for (const [key, value] of Object.entries(raw)) {
		if (!key || IGNORED_CATALOG_KEYS.has(key.toLowerCase())) continue;
		const entry = asObject(value);
		if (!entry) continue;
		catalog[key] = entry as PricingEntry;
	}
	return catalog;
}

async function readDiskPricingCache(
	cachePath: string,
	rawUrl: string,
	normalizedUrl: string,
): Promise<CachedPricingRecord | undefined> {
	try {
		const content = await readFile(cachePath, "utf8");
		const parsed = asObject(JSON.parse(content));
		if (!parsed) return undefined;
		if (
			typeof parsed.url !== "string" ||
			(parsed.url !== rawUrl && parsed.url !== normalizedUrl)
		) {
			return undefined;
		}
		if (
			typeof parsed.fetchedAt !== "number" ||
			!Number.isFinite(parsed.fetchedAt)
		) {
			return undefined;
		}
		const rawCatalog = asObject(parsed.catalog);
		if (!rawCatalog) return undefined;
		return {
			fetchedAt: parsed.fetchedAt,
			catalog: sanitizePricingCatalog(rawCatalog),
		};
	} catch {
		return undefined;
	}
}

export function clearPricingMemoryCache(): void {
	memoryCache.clear();
	failedFetchCache.clear();
}

export async function fetchPricingCatalog(
	options: FetchPricingOptions = {},
): Promise<PricingCatalog> {
	const rawUrl = typeof options.url === "string" ? options.url.trim() : "";
	if (!rawUrl) return {};

	let parsedUrl: URL;
	try {
		parsedUrl = new URL(rawUrl);
	} catch {
		warnPricing(`invalid pricing URL: ${rawUrl}`);
		return {};
	}
	if (parsedUrl.protocol !== "http:" && parsedUrl.protocol !== "https:") {
		warnPricing(
			`pricing URL must use http or https protocol, got ${parsedUrl.protocol}`,
		);
		return {};
	}
	if (!parsedUrl.hostname || parsedUrl.username || parsedUrl.password) {
		warnPricing(
			"pricing URL must have a hostname and no embedded credentials",
		);
		return {};
	}

	const timeoutMs = isPositiveFinite(options.timeoutMs)
		? options.timeoutMs
		: DEFAULT_TIMEOUT_MS;
	const cacheTtlMs = isNonNegativeFinite(options.cacheTtlMs)
		? options.cacheTtlMs
		: DEFAULT_CACHE_TTL_MS;
	const cacheKey = `${options.cachePath ?? ""}::${parsedUrl.href}`;

	let staleDiskCatalog: PricingCatalog | undefined;

	if (!options.forceRefresh) {
		const cached = memoryCache.get(cacheKey);
		if (
			cached &&
			cacheTtlMs > 0 &&
			Date.now() - cached.fetchedAt < cacheTtlMs
		) {
			return cached.catalog;
		}

		if (options.cachePath) {
			const disk = await readDiskPricingCache(
				options.cachePath,
				rawUrl,
				parsedUrl.href,
			);
			if (disk) {
				staleDiskCatalog = disk.catalog;
				if (cacheTtlMs > 0 && Date.now() - disk.fetchedAt < cacheTtlMs) {
					memoryCache.set(cacheKey, disk);
					failedFetchCache.delete(cacheKey);
					return disk.catalog;
				}
			}
		}

		const failed = failedFetchCache.get(cacheKey);
		if (
			failed &&
			cacheTtlMs > 0 &&
			Date.now() - failed.fetchedAt < NEGATIVE_CACHE_TTL_MS
		) {
			return staleDiskCatalog ?? failed.catalog;
		}
	}

	try {
		const timeoutSignal = AbortSignal.timeout(timeoutMs);
		const combinedSignal = options.signal
			? AbortSignal.any([options.signal, timeoutSignal])
			: timeoutSignal;

		const response = await fetch(parsedUrl, { signal: combinedSignal });
		if (!response.ok) {
			throw new Error(`HTTP ${response.status}`);
		}

		const payload = await response.json();
		const root = asObject(payload);
		if (!root) {
			throw new Error("expected a JSON object");
		}

		const catalog = sanitizePricingCatalog(root);
		const fetchedAt = Date.now();
		memoryCache.set(cacheKey, { fetchedAt, catalog });
		failedFetchCache.delete(cacheKey);

		if (options.cachePath) {
			try {
				await mkdir(dirname(options.cachePath), { recursive: true });
				await writeFile(
					options.cachePath,
					JSON.stringify({ url: rawUrl, fetchedAt, catalog }),
					"utf8",
				);
			} catch {
				// Best-effort disk cache write; ignore filesystem errors.
			}
		}

		return catalog;
	} catch (error) {
		if (options.forceRefresh && options.cachePath && !staleDiskCatalog) {
			const disk = await readDiskPricingCache(
				options.cachePath,
				rawUrl,
				parsedUrl.href,
			);
			staleDiskCatalog = disk?.catalog;
		}
		const fallbackCatalog = staleDiskCatalog ?? {};
		if (!options.signal?.aborted) {
			if (cacheTtlMs > 0) {
				failedFetchCache.set(cacheKey, {
					fetchedAt: Date.now(),
					catalog: fallbackCatalog,
				});
			}
			warnPricing(`failed to fetch ${rawUrl}: ${String(error)}`);
		}
		return fallbackCatalog;
	}
}

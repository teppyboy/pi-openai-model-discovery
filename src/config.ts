import { execSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import type { ProviderConfig } from "@earendil-works/pi-coding-agent";

export const DEFAULT_API = "openai-completions";
export const DEFAULT_CONTEXT_WINDOW = 128_000;
export const DEFAULT_MAX_TOKENS = 16_384;
export const DEFAULT_REQUEST_TIMEOUT_MS = 15_000;
export const DEFAULT_PRICING_URL =
	"https://raw.githubusercontent.com/Wei-Shaw/model-price-repo/main/model_prices_and_context_window.json";
export const DEFAULT_PRICING_CACHE_TTL_MS = 6 * 60 * 60 * 1000;
export const DEFAULT_INCLUDE_PATTERNS: readonly string[] = [".*"];
export const DEFAULT_EXCLUDE_PATTERNS: readonly string[] = [
	"(?:^|[-_/])(?:embedding|embeddings|tts|whisper|transcribe|transcription|moderation|realtime|rerank|auto-review)(?:$|[-_/])",
	"(?:^|[-_/])(?:dall-e|imagen|flux|sora|veo|gpt-image)(?:$|[-_/])",
];

export type ProviderApi = NonNullable<ProviderConfig["api"]>;
export type JsonObject = Record<string, unknown>;
export type ThinkingLevel =
	| "off"
	| "minimal"
	| "low"
	| "medium"
	| "high"
	| "xhigh"
	| "max";
export type ThinkingLevelMap = Partial<Record<ThinkingLevel, string | null>>;
export type ModelCostTier = {
	inputTokensAbove: number;
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
};
export type ModelCost = {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	tiers?: ModelCostTier[];
};
export type DynamicProvider = {
	id: string;
	baseUrl: string;
	api: ProviderApi;
	name?: string;
	apiKey?: string;
	headers?: Record<string, string>;
	authHeader?: boolean;
	compat?: JsonObject;
};
export type DiscoveryDefaults = {
	contextWindow?: number;
	maxTokens?: number;
	input?: Array<"text" | "image">;
	reasoning?: boolean;
	thinkingLevelMap?: ThinkingLevelMap;
	compat?: JsonObject;
};
export type ModelOverride = {
	name?: string;
	contextWindow?: number;
	maxTokens?: number;
	input?: Array<"text" | "image">;
	reasoning?: boolean;
	thinkingLevelMap?: ThinkingLevelMap;
	cost?: Partial<ModelCost>;
	compat?: JsonObject;
};
export type DiscoveryConfig = {
	providerId?: string;
	pricingUrl?: string | null;
	pricingCacheTtlMs?: number;
	requestTimeoutMs?: number;
	include?: string[];
	exclude?: string[];
	defaults?: DiscoveryDefaults;
	overrides?: Record<string, ModelOverride>;
};
export type StoredAuthEntry =
	| { type: "api_key"; key: string; env?: Record<string, string> }
	| {
			type: "oauth";
			access?: string;
			refresh?: string;
			expires?: number;
			env?: Record<string, string>;
	  };

const IGNORED_MAP_KEYS = new Set(["__proto__", "constructor", "prototype"]);

const SUPPORTED_DEFAULT_APIS = new Set([
	"openai-completions",
	"openai-responses",
]);
const THINKING_LEVELS = new Set<ThinkingLevel>([
	"off",
	"minimal",
	"low",
	"medium",
	"high",
	"xhigh",
	"max",
]);

const commandValueCache = new Map<string, string | undefined>();

export function asObject(value: unknown): JsonObject | undefined {
	return typeof value === "object" && value !== null && !Array.isArray(value)
		? (value as JsonObject)
		: undefined;
}

/**
 * Strip leading UTF-8 BOM, // and /* *\/ comments, and trailing commas
 * outside quoted strings without modifying URLs or string literals.
 */
export function stripJsonComments(input: string): string {
	return input
		.replace(/^\uFEFF/, "")
		.replace(
			/"(?:\\.|[^"\\])*"|\/\/[^\r\n]*|\/\*[\s\S]*?\*\//g,
			(match) => (match[0] === '"' ? match : ""),
		)
		.replace(
			/"(?:\\.|[^"\\])*"|,(\s*[}\]])/g,
			(match, tail: string | undefined) => (tail !== undefined ? tail : match),
		);
}

export function agentDir(): string {
	const configured = process.env.PI_CODING_AGENT_DIR;
	if (configured && configured.length > 0) {
		if (configured === "~") return homedir();
		if (configured.startsWith("~/") || configured.startsWith("~\\")) {
			return join(homedir(), configured.slice(2));
		}
		return configured;
	}
	return join(homedir(), ".pi", "agent");
}

export function modelsConfigPath(): string {
	return join(agentDir(), "models.json");
}

export function discoveryConfigPath(): string {
	return join(agentDir(), "pi-openai-model-discovery.json");
}

export function authConfigPath(): string {
	return join(agentDir(), "auth.json");
}

export function pricingCachePath(): string {
	return join(agentDir(), "openai-model-discovery-pricing-cache.json");
}

export function isLoopbackUrl(baseUrl: string): boolean {
	let url: URL;
	try {
		url = new URL(baseUrl);
	} catch {
		return false;
	}
	const host = url.hostname.toLowerCase();
	return (
		host === "localhost" ||
		host === "::1" ||
		host === "[::1]" ||
		/^127(?:\.\d{1,3}){3}$/.test(host)
	);
}

export function clearConfigValueCache(): void {
	commandValueCache.clear();
}

export function resolveConfigValue(
	value: string,
	env?: Record<string, string | undefined>,
): string | undefined {
	if (value.startsWith("!")) {
		const command = value.slice(1);
		if (commandValueCache.has(command)) {
			return commandValueCache.get(command);
		}
		if (!command.trim()) {
			commandValueCache.set(command, undefined);
			return undefined;
		}
		try {
			const output = execSync(command, {
				encoding: "utf8",
				timeout: 10_000,
				stdio: ["ignore", "pipe", "ignore"],
				windowsHide: true,
			});
			const resolved = output.trim() || undefined;
			commandValueCache.set(command, resolved);
			return resolved;
		} catch {
			commandValueCache.set(command, undefined);
			return undefined;
		}
	}

	let result = "";
	let i = 0;
	while (i < value.length) {
		if (value[i] !== "$") {
			result += value[i];
			i += 1;
			continue;
		}

		const next = value[i + 1];
		if (next === "$") {
			result += "$";
			i += 2;
			continue;
		}
		if (next === "!") {
			result += "!";
			i += 2;
			continue;
		}
		if (next === "{") {
			const match = /^([A-Za-z_][A-Za-z0-9_]*)\}/.exec(value.slice(i + 2));
			if (match) {
				const name = match[1];
				const resolved = env?.[name] || process.env[name] || undefined;
				if (resolved === undefined) return undefined;
				result += resolved;
				i += 2 + match[0].length;
				continue;
			}
			result += "$";
			i += 1;
			continue;
		}

		const varMatch = /^([A-Za-z_][A-Za-z0-9_]*)/.exec(value.slice(i + 1));
		if (varMatch) {
			const name = varMatch[1];
			const resolved = env?.[name] || process.env[name] || undefined;
			if (resolved === undefined) return undefined;
			result += resolved;
			i += 1 + name.length;
			continue;
		}

		result += "$";
		i += 1;
	}
	return result;
}

export function resolveHeaders(
	headers: Record<string, string> | undefined,
	env?: Record<string, string | undefined>,
): Record<string, string> {
	if (!headers) return {};
	const resolved: Record<string, string> = {};
	for (const [key, value] of Object.entries(headers)) {
		if (typeof value !== "string") continue;
		const resolvedValue = resolveConfigValue(value, env);
		if (resolvedValue !== undefined && resolvedValue.length > 0) {
			resolved[key] = resolvedValue;
		}
	}
	return resolved;
}

export function buildDiscoveryHeaders(
	provider: DynamicProvider,
	options?: {
		credentialKey?: string;
		env?: Record<string, string | undefined>;
	},
): Record<string, string> {
	const headers: Record<string, string> = { accept: "application/json" };
	const resolvedProviderHeaders = resolveHeaders(
		provider.headers,
		options?.env,
	);

	for (const [key, value] of Object.entries(resolvedProviderHeaders)) {
		const lowerKey = key.toLowerCase();
		for (const existingKey of Object.keys(headers)) {
			if (existingKey.toLowerCase() === lowerKey) {
				delete headers[existingKey];
			}
		}
		headers[key] = value;
	}

	const hasAuthorization = Object.keys(headers).some(
		(key) => key.toLowerCase() === "authorization",
	);
	const credentialKey =
		typeof options?.credentialKey === "string" &&
		options.credentialKey.length > 0
			? options.credentialKey
			: undefined;
	const apiKey =
		credentialKey ??
		(provider.apiKey !== undefined
			? resolveConfigValue(provider.apiKey, options?.env)
			: undefined);

	if (
		apiKey !== undefined &&
		apiKey.length > 0 &&
		provider.authHeader !== false &&
		!hasAuthorization
	) {
		headers.Authorization = `Bearer ${apiKey}`;
	}

	return headers;
}

function positiveFiniteNumber(value: unknown): number | undefined {
	return typeof value === "number" && Number.isFinite(value) && value > 0
		? value
		: undefined;
}

function nonNegativeFiniteNumber(value: unknown): number | undefined {
	return typeof value === "number" && Number.isFinite(value) && value >= 0
		? value
		: undefined;
}

function parseInputModalities(
	value: unknown,
): Array<"text" | "image"> | undefined {
	if (!Array.isArray(value)) return undefined;
	const items = [
		...new Set(
			value.flatMap((entry) => {
				if (typeof entry !== "string") return [];
				const normalized = entry.trim().toLowerCase();
				return normalized === "text" || normalized === "image"
					? [normalized as "text" | "image"]
					: [];
			}),
		),
	];
	return items.length > 0 ? items : undefined;
}

function parseThinkingLevelMap(value: unknown): ThinkingLevelMap | undefined {
	const obj = asObject(value);
	if (!obj) return undefined;
	const map: ThinkingLevelMap = {};
	let hasAny = false;
	for (const [key, entry] of Object.entries(obj)) {
		if (!THINKING_LEVELS.has(key as ThinkingLevel)) continue;
		if (entry === null) {
			map[key as ThinkingLevel] = null;
			hasAny = true;
		} else if (typeof entry === "string" && entry.trim().length > 0) {
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
		const inputTokensAbove = positiveFiniteNumber(tier.inputTokensAbove);
		const input = nonNegativeFiniteNumber(tier.input);
		const output = nonNegativeFiniteNumber(tier.output);
		if (
			inputTokensAbove === undefined ||
			input === undefined ||
			output === undefined
		) {
			continue;
		}
		const cacheRead =
			nonNegativeFiniteNumber(tier.cacheRead ?? tier.cache_read) ?? 0;
		const cacheWrite =
			nonNegativeFiniteNumber(tier.cacheWrite ?? tier.cache_write) ?? 0;
		tiers.push({
			inputTokensAbove,
			input,
			output,
			cacheRead,
			cacheWrite,
		});
	}
	if (tiers.length === 0) return undefined;
	return tiers.sort((a, b) => a.inputTokensAbove - b.inputTokensAbove);
}

function parsePartialModelCost(value: unknown): Partial<ModelCost> | undefined {
	const obj = asObject(value);
	if (!obj) return undefined;
	const cost: Partial<ModelCost> = {};
	const input = nonNegativeFiniteNumber(obj.input);
	if (input !== undefined) cost.input = input;
	const output = nonNegativeFiniteNumber(obj.output);
	if (output !== undefined) cost.output = output;
	const cacheRead = nonNegativeFiniteNumber(obj.cacheRead ?? obj.cache_read);
	if (cacheRead !== undefined) cost.cacheRead = cacheRead;
	const cacheWrite = nonNegativeFiniteNumber(obj.cacheWrite ?? obj.cache_write);
	if (cacheWrite !== undefined) cost.cacheWrite = cacheWrite;
	const tiers = parseCostTiers(obj.tiers);
	if (tiers !== undefined) cost.tiers = tiers;
	return Object.keys(cost).length > 0 ? cost : undefined;
}

function parseDiscoveryDefaults(
	value: unknown,
): DiscoveryDefaults | undefined {
	const obj = asObject(value);
	if (!obj) return undefined;
	const defaults: DiscoveryDefaults = {};
	const contextWindow = positiveFiniteNumber(obj.contextWindow);
	if (contextWindow !== undefined) defaults.contextWindow = contextWindow;
	const maxTokens = positiveFiniteNumber(obj.maxTokens);
	if (maxTokens !== undefined) defaults.maxTokens = maxTokens;
	const input = parseInputModalities(obj.input);
	if (input !== undefined) defaults.input = input;
	if (typeof obj.reasoning === "boolean") defaults.reasoning = obj.reasoning;
	const thinkingLevelMap = parseThinkingLevelMap(obj.thinkingLevelMap);
	if (thinkingLevelMap !== undefined) {
		defaults.thinkingLevelMap = thinkingLevelMap;
	}
	const compat = asObject(obj.compat);
	if (compat !== undefined) defaults.compat = { ...compat };
	return Object.keys(defaults).length > 0 ? defaults : undefined;
}

function parseModelOverride(value: unknown): ModelOverride | undefined {
	const obj = asObject(value);
	if (!obj) return undefined;
	const override: ModelOverride = {};
	if (typeof obj.name === "string" && obj.name.trim().length > 0) {
		override.name = obj.name.trim();
	}
	const contextWindow = positiveFiniteNumber(obj.contextWindow);
	if (contextWindow !== undefined) override.contextWindow = contextWindow;
	const maxTokens = positiveFiniteNumber(obj.maxTokens);
	if (maxTokens !== undefined) override.maxTokens = maxTokens;
	const input = parseInputModalities(obj.input);
	if (input !== undefined) override.input = input;
	if (typeof obj.reasoning === "boolean") override.reasoning = obj.reasoning;
	const thinkingLevelMap = parseThinkingLevelMap(obj.thinkingLevelMap);
	if (thinkingLevelMap !== undefined) {
		override.thinkingLevelMap = thinkingLevelMap;
	}
	const cost = parsePartialModelCost(obj.cost);
	if (cost !== undefined) override.cost = cost;
	const compat = asObject(obj.compat);
	if (compat !== undefined) override.compat = { ...compat };
	return Object.keys(override).length > 0 ? override : undefined;
}

export async function readDiscoveryConfig(
	filePath = discoveryConfigPath(),
): Promise<DiscoveryConfig> {
	let source: string;
	try {
		source = await readFile(filePath, "utf8");
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === "ENOENT") return {};
		process.stderr.write(
			`OpenAI model discovery could not read ${filePath}: ${String(error)}\n`,
		);
		return {};
	}

	let raw: JsonObject | undefined;
	try {
		raw = asObject(JSON.parse(stripJsonComments(source)));
	} catch (error) {
		process.stderr.write(
			`OpenAI model discovery could not parse ${filePath}: ${String(error)}\n`,
		);
		return {};
	}

	if (!raw) return {};

	const config: DiscoveryConfig = {};

	if (typeof raw.providerId === "string" && raw.providerId.trim().length > 0) {
		config.providerId = raw.providerId.trim();
	}

	if (raw.pricingUrl === null) {
		config.pricingUrl = null;
	} else if (typeof raw.pricingUrl === "string") {
		const trimmed = raw.pricingUrl.trim();
		config.pricingUrl = trimmed.length > 0 ? trimmed : null;
	}

	const pricingCacheTtlMs = nonNegativeFiniteNumber(raw.pricingCacheTtlMs);
	if (pricingCacheTtlMs !== undefined) {
		config.pricingCacheTtlMs = pricingCacheTtlMs;
	}

	const requestTimeoutMs = positiveFiniteNumber(raw.requestTimeoutMs);
	if (requestTimeoutMs !== undefined) {
		config.requestTimeoutMs = requestTimeoutMs;
	}

	if (Array.isArray(raw.include)) {
		const include = raw.include.filter(
			(item): item is string => typeof item === "string" && item.length > 0,
		);
		if (include.length > 0) {
			config.include = include;
		}
	}

	if (Array.isArray(raw.exclude)) {
		const exclude = raw.exclude.filter(
			(item): item is string => typeof item === "string" && item.length > 0,
		);
		config.exclude = exclude;
	}

	const defaults = parseDiscoveryDefaults(raw.defaults);
	if (defaults !== undefined) {
		config.defaults = defaults;
	}

	const rawOverrides = asObject(raw.overrides);
	if (rawOverrides) {
		const overrides = Object.create(null) as Record<string, ModelOverride>;
		for (const [modelId, value] of Object.entries(rawOverrides)) {
			if (!modelId.trim() || IGNORED_MAP_KEYS.has(modelId)) continue;
			const parsed = parseModelOverride(value);
			if (parsed !== undefined) {
				overrides[modelId] = parsed;
			}
		}
		if (Object.keys(overrides).length > 0) {
			config.overrides = overrides;
		}
	}

	return config;
}

function parseStringRecord(
	value: unknown,
): Record<string, string> | undefined {
	const obj = asObject(value);
	if (!obj) return undefined;
	const result: Record<string, string> = {};
	for (const [key, item] of Object.entries(obj)) {
		if (IGNORED_MAP_KEYS.has(key)) continue;
		if (typeof item === "string") {
			result[key] = item;
		}
	}
	return Object.keys(result).length > 0 ? result : undefined;
}

export async function readAuthStore(
	filePath = authConfigPath(),
): Promise<Record<string, StoredAuthEntry>> {
	let source: string;
	try {
		source = await readFile(filePath, "utf8");
	} catch {
		return Object.create(null) as Record<string, StoredAuthEntry>;
	}

	let root: JsonObject | undefined;
	try {
		root = asObject(JSON.parse(stripJsonComments(source)));
	} catch {
		return Object.create(null) as Record<string, StoredAuthEntry>;
	}

	if (!root) return Object.create(null) as Record<string, StoredAuthEntry>;

	const store = Object.create(null) as Record<string, StoredAuthEntry>;
	for (const [providerId, value] of Object.entries(root)) {
		if (IGNORED_MAP_KEYS.has(providerId)) continue;
		const entry = asObject(value);
		if (!entry) continue;
		const env = parseStringRecord(entry.env);
		if (entry.type === "api_key") {
			if (typeof entry.key === "string" && entry.key.length > 0) {
				store[providerId] = {
					type: "api_key",
					key: entry.key,
					...(env ? { env } : {}),
				};
			}
		} else if (entry.type === "oauth") {
			const oauth: Extract<StoredAuthEntry, { type: "oauth" }> = {
				type: "oauth",
				...(env ? { env } : {}),
			};
			if (typeof entry.access === "string" && entry.access.length > 0) {
				oauth.access = entry.access;
			}
			if (typeof entry.refresh === "string" && entry.refresh.length > 0) {
				oauth.refresh = entry.refresh;
			}
			if (typeof entry.expires === "number" && Number.isFinite(entry.expires)) {
				oauth.expires = entry.expires;
			}
			store[providerId] = oauth;
		}
	}
	return store;
}

export async function readDynamicProviders(
	filePath = modelsConfigPath(),
	config?: DiscoveryConfig,
): Promise<DynamicProvider[]> {
	let source: string;
	try {
		source = await readFile(filePath, "utf8");
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
		process.stderr.write(
			`OpenAI model discovery could not read ${filePath}: ${String(error)}\n`,
		);
		return [];
	}

	let parsedConfig: JsonObject;
	try {
		const parsed = JSON.parse(stripJsonComments(source));
		parsedConfig = asObject(parsed) ?? {};
	} catch (error) {
		process.stderr.write(
			`OpenAI model discovery could not parse ${filePath}: ${String(error)}\n`,
		);
		return [];
	}

	const providers = asObject(parsedConfig.providers);
	if (!providers) return [];

	return Object.entries(providers).flatMap(([id, value]) => {
		if (config?.providerId !== undefined && id !== config.providerId) {
			return [];
		}

		const provider = asObject(value);
		if (!provider) return [];

		const baseUrl =
			typeof provider.baseUrl === "string" ? provider.baseUrl.trim() : "";
		const models = provider.models;
		if (
			!baseUrl ||
			(models !== undefined && (!Array.isArray(models) || models.length > 0))
		) {
			return [];
		}

		let api: ProviderApi;
		if (typeof provider.api === "string") {
			const explicitApi = provider.api.trim();
			if (!explicitApi) {
				api = DEFAULT_API;
			} else if (
				SUPPORTED_DEFAULT_APIS.has(explicitApi) ||
				config?.providerId === id
			) {
				api = explicitApi as ProviderApi;
			} else {
				return [];
			}
		} else {
			api = DEFAULT_API;
		}

		const result: DynamicProvider = {
			id,
			baseUrl,
			api,
		};

		if (typeof provider.name === "string" && provider.name.trim().length > 0) {
			result.name = provider.name.trim();
		}
		if (typeof provider.apiKey === "string" && provider.apiKey.length > 0) {
			result.apiKey = provider.apiKey;
		}
		const rawHeaders = asObject(provider.headers);
		if (rawHeaders) {
			const headers: Record<string, string> = {};
			for (const [headerKey, headerValue] of Object.entries(rawHeaders)) {
				if (typeof headerValue === "string") {
					headers[headerKey] = headerValue;
				}
			}
			if (Object.keys(headers).length > 0) {
				result.headers = headers;
			}
		}
		if (typeof provider.authHeader === "boolean") {
			result.authHeader = provider.authHeader;
		}
		const compat = asObject(provider.compat);
		if (compat !== undefined) {
			result.compat = { ...compat };
		}

		return [result];
	});
}

export function compilePatterns(
	patterns: readonly string[] | undefined,
	fallback: readonly string[],
): RegExp[] {
	const source = patterns === undefined ? fallback : patterns;
	const compiled: RegExp[] = [];
	for (const pattern of source) {
		try {
			compiled.push(new RegExp(pattern, "i"));
		} catch (error) {
			process.stderr.write(
				`OpenAI model discovery ignoring invalid regex pattern ${JSON.stringify(pattern)}: ${String(error)}\n`,
			);
		}
	}
	return compiled;
}

export function shouldIncludeModel(
	id: string,
	include: readonly RegExp[],
	exclude: readonly RegExp[],
): boolean {
	return (
		(include.length === 0 || include.some((re) => re.test(id))) &&
		!exclude.some((re) => re.test(id))
	);
}

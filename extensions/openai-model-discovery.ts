import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import type {
	ExtensionAPI,
	ProviderConfig,
} from "@earendil-works/pi-coding-agent";

export const DEFAULT_API = "openai-completions";
export const DEFAULT_CONTEXT_WINDOW = 128_000;
export const DEFAULT_MAX_TOKENS = 16_384;

type ProviderApi = NonNullable<ProviderConfig["api"]>;
type RefreshContext = Parameters<
	NonNullable<ProviderConfig["refreshModels"]>
>[0];

const DEFAULT_COST = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
};

export type DynamicProvider = {
	id: string;
	baseUrl: string;
	api: ProviderApi;
};

type ThinkingLevel =
	| "off"
	| "minimal"
	| "low"
	| "medium"
	| "high"
	| "xhigh"
	| "max";
type ThinkingLevelMap = Partial<Record<ThinkingLevel, string | null>>;

export type ProviderModelDefinition = {
	id: string;
	name: string;
	api: ProviderApi;
	baseUrl: string;
	reasoning: boolean;
	thinkingLevelMap?: ThinkingLevelMap;
	input: Array<"text" | "image">;
	cost: typeof DEFAULT_COST;
	contextWindow: number;
	maxTokens: number;
};

type JsonObject = Record<string, unknown>;

function asObject(value: unknown): JsonObject | undefined {
	return typeof value === "object" && value !== null && !Array.isArray(value)
		? (value as JsonObject)
		: undefined;
}

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
				const normalized = item.toLowerCase();
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
				map[level] = level;
				break;
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

function isGptReasoningModel(id: string): boolean {
	// ponytail: bare 9router catalogs expose IDs only; use the conservative GPT-5
	// floor until the gateway provides per-model reasoning metadata.
	return /(?:^|\/)gpt-5(?:[.-]|$)/i.test(id);
}

function inputFromModel(
	model: JsonObject | undefined,
	capabilities: JsonObject | undefined,
): Array<"text" | "image"> {
	const candidates = [
		model?.input,
		model?.input_types,
		model?.input_modalities,
		model?.supportedInputModalities,
		model?.supported_input_modalities,
		capabilities?.input_modalities,
		capabilities?.inputModalities,
	];
	const input = candidates
		.map(validInput)
		.find((value): value is Array<"text" | "image"> => value !== undefined) ?? [
		"text",
	];
	if (model?.vision === true || capabilities?.vision === true) {
		const augmented: Array<"text" | "image"> = [...input];
		if (!augmented.includes("image")) augmented.push("image");
		return augmented;
	}
	return input;
}

/** Strip // comments and trailing commas without changing quoted strings. */
export function stripJsonComments(input: string): string {
	return input
		.replace(/"(?:\\.|[^"\\])*"|\/\/[^\n]*/g, (match) =>
			match[0] === '"' ? match : "",
		)
		.replace(
			/"(?:\\.|[^"\\])*"|,(\s*[}\]])/g,
			(match, tail: string) => tail ?? (match[0] === '"' ? match : ""),
		);
}

function modelsConfigPath(): string {
	return join(
		process.env.PI_CODING_AGENT_DIR || join(homedir(), ".pi", "agent"),
		"models.json",
	);
}

export async function readDynamicProviders(
	filePath = modelsConfigPath(),
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

	let config: JsonObject;
	try {
		const parsed = JSON.parse(stripJsonComments(source));
		config = asObject(parsed) ?? {};
	} catch (error) {
		process.stderr.write(
			`OpenAI model discovery could not parse ${filePath}: ${String(error)}\n`,
		);
		return [];
	}

	const providers = asObject(config.providers);
	if (!providers) return [];

	return Object.entries(providers).flatMap(([id, value]) => {
		const provider = asObject(value);
		const baseUrl =
			typeof provider?.baseUrl === "string" ? provider.baseUrl.trim() : "";
		const models = provider?.models;
		if (
			!baseUrl ||
			(models !== undefined && (!Array.isArray(models) || models.length > 0))
		)
			return [];
		return [
			{
				id,
				baseUrl,
				api: (typeof provider?.api === "string"
					? provider.api
					: DEFAULT_API) as ProviderApi,
			},
		];
	});
}

export function mapModelRecord(
	record: unknown,
	provider: DynamicProvider,
): ProviderModelDefinition {
	const model = asObject(record);
	const id =
		firstString(model?.id, model?.slug, model?.model, model?.name) ?? "";
	if (!id)
		throw new Error(
			`Model discovery failed for ${provider.id}: every model needs a non-empty id`,
		);

	const capabilities = asObject(model?.capabilities);
	const cost = asObject(model?.cost);
	const levels =
		model?.supported_reasoning_levels ??
		model?.reasoning_levels ??
		capabilities?.supported_reasoning_levels;
	const gptModel = isGptReasoningModel(id);
	const explicitReasoning = firstBoolean(
		model?.reasoning,
		model?.supports_reasoning,
		capabilities?.reasoning,
		reasoningFromLevels(levels),
	);
	const reasoning = explicitReasoning ?? gptModel;
	const thinkingLevelMap =
		thinkingLevelMapFromLevels(levels) ??
		(reasoning && gptModel
			? { minimal: "low", xhigh: null, max: null }
			: undefined);
	return {
		id,
		name: firstString(model?.name, model?.display_name, model?.displayName) ?? id,
		api: provider.api,
		baseUrl: provider.baseUrl,
		reasoning,
		...(thinkingLevelMap ? { thinkingLevelMap } : {}),
		input: inputFromModel(model, capabilities),
		cost: {
			input: numberOrZero(cost?.input),
			output: numberOrZero(cost?.output),
			cacheRead: numberOrZero(cost?.cacheRead ?? cost?.cache_read),
			cacheWrite: numberOrZero(cost?.cacheWrite ?? cost?.cache_write),
		},
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
			) ?? DEFAULT_MAX_TOKENS,
	};
}

function toProviderModelDefinition(
	model: unknown,
	provider: DynamicProvider,
): ProviderModelDefinition {
	return mapModelRecord(model, provider);
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

export async function refreshProvider(
	provider: DynamicProvider,
	context: RefreshContext,
): Promise<ProviderModelDefinition[]> {
	if (!context.allowNetwork) {
		return (Array.isArray(context.stored?.models) ? context.stored.models : [])
			.filter((model: unknown) => asObject(model)?.provider === provider.id)
			.map((model: unknown) => toProviderModelDefinition(model, provider));
	}

	const headers: Record<string, string> = { accept: "application/json" };
	if (context.credential?.type === "api_key" && context.credential.key) {
		headers.Authorization = `Bearer ${context.credential.key}`;
	}

	const discoveryUrl = modelDiscoveryUrl(provider.baseUrl);
	const records = await discoverModelRecords(
		discoveryUrl,
		headers,
		context.signal,
	);

	const definitions = records.map((record: unknown) =>
		mapModelRecord(record, provider),
	);
	await context.publish({
		persist: {
			checkedAt: Date.now(),
			models: definitions.map((model: ProviderModelDefinition) => ({
				...model,
				provider: provider.id,
			})),
		},
	});
	return definitions;
}

export default async function openAIModelDiscovery(
	pi: ExtensionAPI,
): Promise<void> {
	for (const provider of await readDynamicProviders()) {
		pi.registerProvider(provider.id, {
			baseUrl: provider.baseUrl,
			api: provider.api,
			refreshModels: (context: RefreshContext) =>
				refreshProvider(provider, context),
		});
	}
}

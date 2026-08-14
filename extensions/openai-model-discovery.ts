import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ProviderConfig } from "@earendil-works/pi-coding-agent";

export const DEFAULT_API = "openai-completions";
export const DEFAULT_CONTEXT_WINDOW = 128_000;
export const DEFAULT_MAX_TOKENS = 16_384;

type ProviderApi = NonNullable<ProviderConfig["api"]>;
type RefreshContext = Parameters<NonNullable<ProviderConfig["refreshModels"]>>[0];

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

export type ProviderModelDefinition = {
	id: string;
	name: string;
	api: ProviderApi;
	baseUrl: string;
	reasoning: boolean;
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
		if (typeof value === "number" && Number.isFinite(value) && value > 0) return value;
	}
	return undefined;
}

function numberOrZero(value: unknown): number {
	return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : 0;
}

function validInput(value: unknown): Array<"text" | "image"> | undefined {
	let values: unknown[];
	if (Array.isArray(value)) values = value;
	else if (typeof value === "string") values = [value];
	else values = [];

	const input = [...new Set(values.filter((item): item is "text" | "image" => item === "text" || item === "image"))];
	return input.length > 0 ? input : undefined;
}

/** Strip // comments and trailing commas without changing quoted strings. */
export function stripJsonComments(input: string): string {
	return input
		.replace(/"(?:\\.|[^"\\])*"|\/\/[^\n]*/g, (match) => (match[0] === '"' ? match : ""))
		.replace(/"(?:\\.|[^"\\])*"|,(\s*[}\]])/g, (match, tail: string) =>
			tail ?? (match[0] === '"' ? match : ""),
		);
}

function modelsConfigPath(): string {
	return join(process.env.PI_CODING_AGENT_DIR || join(homedir(), ".pi", "agent"), "models.json");
}

export async function readDynamicProviders(filePath = modelsConfigPath()): Promise<DynamicProvider[]> {
	let source: string;
	try {
		source = await readFile(filePath, "utf8");
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
		process.stderr.write(`OpenAI model discovery could not read ${filePath}: ${String(error)}\n`);
		return [];
	}

	let config: JsonObject;
	try {
		const parsed = JSON.parse(stripJsonComments(source));
		config = asObject(parsed) ?? {};
	} catch (error) {
		process.stderr.write(`OpenAI model discovery could not parse ${filePath}: ${String(error)}\n`);
		return [];
	}

	const providers = asObject(config.providers);
	if (!providers) return [];

	return Object.entries(providers).flatMap(([id, value]) => {
		const provider = asObject(value);
		const baseUrl = typeof provider?.baseUrl === "string" ? provider.baseUrl.trim() : "";
		const models = provider?.models;
		if (!baseUrl || (models !== undefined && (!Array.isArray(models) || models.length > 0))) return [];
		return [{
			id,
			baseUrl,
			api: (typeof provider?.api === "string" ? provider.api : DEFAULT_API) as ProviderApi,
		}];
	});
}

export function mapModelRecord(record: unknown, provider: DynamicProvider): ProviderModelDefinition {
	const model = asObject(record);
	const id = typeof model?.id === "string" ? model.id.trim() : "";
	if (!id) throw new Error(`Model discovery failed for ${provider.id}: every model needs a non-empty id`);

	const capabilities = asObject(model?.capabilities);
	const cost = asObject(model?.cost);
	return {
		id,
		name: typeof model?.name === "string" && model.name.trim() ? model.name : id,
		api: provider.api,
		baseUrl: provider.baseUrl,
		reasoning: firstBoolean(model?.reasoning, model?.supports_reasoning, capabilities?.reasoning) ?? false,
		input: validInput(model?.input ?? model?.input_types) ?? ["text"],
		cost: {
			input: numberOrZero(cost?.input),
			output: numberOrZero(cost?.output),
			cacheRead: numberOrZero(cost?.cacheRead ?? cost?.cache_read),
			cacheWrite: numberOrZero(cost?.cacheWrite ?? cost?.cache_write),
		},
		contextWindow:
			positiveNumber(model?.contextWindow, model?.context_window, model?.context_length, model?.max_model_len) ??
			DEFAULT_CONTEXT_WINDOW,
		maxTokens:
			positiveNumber(model?.maxTokens, model?.max_tokens, model?.max_output_tokens) ?? DEFAULT_MAX_TOKENS,
	};
}

function toProviderModelDefinition(model: unknown, provider: DynamicProvider): ProviderModelDefinition {
	return mapModelRecord(model, provider);
}

export function modelDiscoveryUrl(baseUrl: string): URL {
	let url: URL;
	try {
		url = new URL(baseUrl);
	} catch {
		throw new Error(`Model discovery requires a valid HTTP(S) base URL, got ${baseUrl}`);
	}
	if (url.protocol !== "http:" && url.protocol !== "https:") {
		throw new Error(`Model discovery requires an HTTP(S) base URL, got ${url.protocol}`);
	}
	if (!url.hostname || url.username || url.password) {
		throw new Error("Model discovery base URL must have a hostname and no embedded credentials");
	}
	url.pathname = `${url.pathname.replace(/\/+$/, "")}/models`;
	url.search = "";
	url.hash = "";
	return url;
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
	const response = await fetch(discoveryUrl, {
		signal: context.signal,
		headers,
	});
	if (!response.ok) {
		throw new Error(`Model discovery failed for ${provider.id}: HTTP ${response.status}`);
	}

	const payload = await response.json();
	if (!Array.isArray(payload?.data)) {
		throw new Error(`Model discovery failed for ${provider.id}: response.data is not an array`);
	}

	const definitions = payload.data.map((record: unknown) => mapModelRecord(record, provider));
	await context.publish({
		persist: {
			checkedAt: Date.now(),
			models: definitions.map((model: ProviderModelDefinition) => ({ ...model, provider: provider.id })),
		},
	});
	return definitions;
}

export default async function openAIModelDiscovery(pi: ExtensionAPI): Promise<void> {
	for (const provider of await readDynamicProviders()) {
		pi.registerProvider(provider.id, {
			baseUrl: provider.baseUrl,
			api: provider.api,
			refreshModels: (context: RefreshContext) => refreshProvider(provider, context),
		});
	}
}

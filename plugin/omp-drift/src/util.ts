export function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function describeError(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

/** Non-negative integer check for typed protocol fields and command arguments. */
export function isNonNegativeInteger(value: unknown): value is number {
	return typeof value === "number" && Number.isInteger(value) && value >= 0;
}

/** Parse a decimal non-negative integer argument; anything else is free text and is rejected. */
export function parseIntegerArgument(text: string): number | undefined {
	if (!/^\d{1,9}$/.test(text)) return undefined;
	return Number.parseInt(text, 10);
}

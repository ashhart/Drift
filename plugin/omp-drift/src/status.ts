import { isEnumName, isHexDigest, isPhase, type Phase } from "./protocol";
import { isNonNegativeInteger, isRecord } from "./util";

// Parse only the typed status fields that may reach the plugin UI.
export interface MemberStatus {
	index: number;
	writer: number;
	epoch: number;
	sequence: number;
	sourcePosition: number;
	localTokens: number;
	mailSent: number;
	foreignTokens: number;
	mailForeignTokens: number;
	/** layer → gate state enum name (e.g. "open"). */
	gate: Record<string, string>;
	/** layer → mean injected mass. */
	massMean: Record<string, number>;
}

export interface ServiceStatus {
	phase: Phase;
	epoch: number;
	poisoned: boolean;
	members: MemberStatus[];
	/** mailbox state enum name → count. */
	mailboxCounts: Record<string, number>;
	detectors: {
		nonfinite: boolean;
		normDrift: Record<string, number>;
		massOscillation: Record<string, number>;
	};
	manifestSha256: string;
	lastCheckpointSha256?: string;
}

const LAYER_KEY = /^\d{1,4}$/;

function integerField(record: Record<string, unknown>, key: string): number | undefined {
	const value = record[key];
	return isNonNegativeInteger(value) ? value : undefined;
}

function numberMap(value: unknown): Record<string, number> | undefined {
	if (!isRecord(value)) return undefined;
	const out: Record<string, number> = {};
	for (const key of Object.keys(value).sort()) {
		const item = value[key];
		if (!LAYER_KEY.test(key) || typeof item !== "number" || !Number.isFinite(item)) return undefined;
		out[key] = item;
	}
	return out;
}

function enumMap(value: unknown): Record<string, string> | undefined {
	if (!isRecord(value)) return undefined;
	const out: Record<string, string> = {};
	for (const key of Object.keys(value).sort()) {
		const item = value[key];
		if (!LAYER_KEY.test(key) || !isEnumName(item)) return undefined;
		out[key] = item;
	}
	return out;
}

function parseMember(value: unknown): MemberStatus | undefined {
	if (!isRecord(value)) return undefined;
	const fields = {
		index: integerField(value, "index"),
		writer: integerField(value, "writer"),
		epoch: integerField(value, "epoch"),
		sequence: integerField(value, "sequence"),
		sourcePosition: integerField(value, "source_position"),
		localTokens: integerField(value, "local_tokens"),
		mailSent: integerField(value, "mail_sent"),
		foreignTokens: integerField(value, "foreign_tokens"),
		mailForeignTokens: integerField(value, "mail_foreign_tokens"),
	};
	const gate = enumMap(value.gate ?? {});
	const massMean = numberMap(value.mass_mean ?? {});
	if (!gate || !massMean) return undefined;
	for (const item of Object.values(fields)) if (item === undefined) return undefined;
	return { ...(fields as { [K in keyof typeof fields]: number }), gate, massMean };
}

function parseMailboxCounts(value: unknown): Record<string, number> | undefined {
	if (!isRecord(value) || !isRecord(value.counts)) return undefined;
	const out: Record<string, number> = {};
	for (const key of Object.keys(value.counts)) {
		const count = value.counts[key];
		if (!isEnumName(key) || !isNonNegativeInteger(count)) return undefined;
		out[key] = count;
	}
	return out;
}

function parseDetectors(value: unknown): ServiceStatus["detectors"] | undefined {
	if (!isRecord(value) || typeof value.nonfinite !== "boolean") return undefined;
	const normDrift = numberMap(value.norm_drift ?? {});
	const massOscillation = numberMap(value.mass_oscillation ?? {});
	if (!normDrift || !massOscillation) return undefined;
	return { nonfinite: value.nonfinite, normDrift, massOscillation };
}

/** Strictly parse a status object; anything off-schema yields undefined (treated as MALFORMED). */
export function parseStatus(value: unknown): ServiceStatus | undefined {
	if (!isRecord(value)) return undefined;
	if (!isPhase(value.phase) || !isNonNegativeInteger(value.epoch) || typeof value.poisoned !== "boolean") {
		return undefined;
	}
	if (!Array.isArray(value.members) || !isHexDigest(value.manifest_sha256)) return undefined;
	const members: MemberStatus[] = [];
	for (const item of value.members) {
		const member = parseMember(item);
		if (!member) return undefined;
		members.push(member);
	}
	const mailboxCounts = parseMailboxCounts(value.mailbox);
	const detectors = parseDetectors(value.detectors);
	if (!mailboxCounts || !detectors) return undefined;
	const checkpoint = value.last_checkpoint_sha256;
	if (checkpoint !== undefined && checkpoint !== null && checkpoint !== "" && !isHexDigest(checkpoint)) {
		return undefined;
	}
	return {
		phase: value.phase,
		epoch: value.epoch,
		poisoned: value.poisoned,
		members,
		mailboxCounts,
		detectors,
		manifestSha256: value.manifest_sha256.toLowerCase(),
		lastCheckpointSha256: isHexDigest(checkpoint) ? checkpoint.toLowerCase() : undefined,
	};
}

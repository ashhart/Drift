import { createHmac } from "node:crypto";
import { describe, expect, test } from "bun:test";
import { canonicalJson, computeAuth, signMessage, verifyAuth } from "../src/protocol";
import { renderNumber } from "../src/board";
import { parseStatus } from "../src/status";

describe("protocol", () => {
	test("canonical JSON sorts keys at every level, drops whitespace and escapes non-ASCII like Python", () => {
		const message = { op: 1, id: 7, task: "héllo \"w\"\n  ✓ \u{1F600} \x7f", assignments: { "1": "b", "0": "a" }, nested: { z: [1, 2.5, false, null], a: "x" } };
		// Reference output of json.dumps(sample, sort_keys=True, separators=(",", ":")).
		expect(canonicalJson(message)).toBe(
			'{"assignments":{"0":"a","1":"b"},"id":7,"nested":{"a":"x","z":[1,2.5,false,null]},"op":1,"task":"h\\u00e9llo \\"w\\"\\n\\u2028 \\u2713 \\ud83d\\ude00 \\u007f"}',
		);
	});

	test("auth is HMAC-SHA256 over the canonical body without the auth field", () => {
		const secret = "k";
		const signed = signMessage({ op: 4, id: 3 }, secret);
		const expected = createHmac("sha256", secret).update('{"id":3,"op":4}').digest("hex");
		expect(signed.auth).toBe(expected);
		expect(computeAuth({ id: 3, op: 4, auth: "ignored" }, secret)).toBe(expected);
		expect(verifyAuth(signed, secret)).toBe(true);
		expect(verifyAuth(signed, "other")).toBe(false);
		expect(verifyAuth({ ...signed, id: 4 }, secret)).toBe(false);
		expect(verifyAuth({ id: 3, op: 4 }, secret)).toBe(false);
		expect(verifyAuth({ id: 3, op: 4, auth: "not-hex" }, secret)).toBe(false);
	});

	test("status parsing rejects any string that is not an enum name or hex digest", () => {
		const good = {
			phase: "STRICT",
			epoch: 1,
			poisoned: false,
			members: [{
				index: 0, writer: 1, epoch: 0, sequence: 1, source_position: 4, local_tokens: 4, mail_sent: 0,
				foreign_tokens: 2, mail_foreign_tokens: 0, gate: { "3": "open" }, mass_mean: { "3": 0.1 },
			}],
			mailbox: { counts: { pending: 0 } },
			detectors: { nonfinite: false, norm_drift: {}, mass_oscillation: {} },
			manifest_sha256: "a".repeat(64),
			last_checkpoint_sha256: null,
		};
		expect(parseStatus(good)?.members[0].gate).toEqual({ "3": "open" });
		expect(parseStatus({ ...good, phase: "OPEN" })).toBeUndefined();
		expect(parseStatus({ ...good, manifest_sha256: "/audit/run-1/final.txt" })).toBeUndefined();
		expect(parseStatus({ ...good, last_checkpoint_sha256: "latest" })).toBeUndefined();
		expect(parseStatus({ ...good, members: [{ ...good.members[0], gate: { "3": "open sesame" } }] })).toBeUndefined();
		expect(parseStatus({ ...good, mailbox: { counts: { "pending items": 0 } } })).toBeUndefined();
		expect(parseStatus({ ...good, epoch: -1 })).toBeUndefined();
	});

	test("a diverged run's status keeps parsing: the service sends NaN and infinity as null, shown as ?", () => {
		const diverged = {
			phase: "STRICT", epoch: 2, poisoned: false,
			members: [{
				index: 0, writer: 1, epoch: 1, sequence: 2, source_position: 4, local_tokens: 4, mail_sent: 0,
				foreign_tokens: 2, mail_foreign_tokens: 0, gate: { "3": "open" }, mass_mean: { "3": null, "7": 0.2 },
			}],
			mailbox: { counts: { pending: 0 } },
			detectors: { nonfinite: true, norm_drift: { "3": null }, mass_oscillation: {} },
			manifest_sha256: "a".repeat(64), last_checkpoint_sha256: null,
		};
		const status = parseStatus(diverged);
		expect(status?.detectors.nonfinite).toBe(true);
		expect(Number.isNaN(status?.members[0].massMean["3"])).toBe(true);
		expect(renderNumber(status?.members[0].massMean["3"])).toBe("?");
		expect(parseStatus({ ...diverged, detectors: { ...diverged.detectors, norm_drift: { "3": "NaN" } } })).toBeUndefined();
	});
});

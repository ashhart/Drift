import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { Op } from "../src/protocol";
import { FakeService, fakeManifestSha256 } from "./fake-service";
import { createHarness, manifestPath, secret, setEnv } from "./harness";

describe("drift behavior", () => {
	let service: FakeService;

	beforeEach(async () => {
		service = await FakeService.start({ secret });
		setEnv(service.port);
	});

	afterEach(async () => {
		await service.close();
		setEnv(undefined, null);
	});

	test("live startup is blocked before connection or assignment delivery", async () => {
		const h = createHarness();
		await h.runCommand("assign 0 private setup must stay local");
		await h.runCommand("start " + manifestPath + " --task connect the real pair");
		expect(h.errors()).toHaveLength(1);
		expect(h.errors()[0].message).toContain("BLOCKED");
		expect(service.connectionsOpened).toBe(0);
		expect(service.requests).toHaveLength(0);
		expect(h.status.get("drift")).toBeUndefined();
	});

	test("missing environment variables produce a clear error and no connection", async () => {
		const h = createHarness();
		setEnv(undefined, null);
		await h.runCommand("start --reference " + manifestPath);
		expect(h.errors()).toHaveLength(1);
		expect(h.errors()[0].message).toContain("DRIFT_SERVICE_PORT");
		expect(h.errors()[0].message).toContain("DRIFT_SERVICE_SECRET");
		expect(h.errors()[0].message).not.toContain(secret);
		expect(service.connectionsOpened).toBe(0);
		expect(h.status.get("drift")).toBeUndefined();
	});

	test("a missing secret alone is named without revealing anything", async () => {
		const h = createHarness();
		setEnv(service.port, null);
		await h.runCommand("start --reference " + manifestPath);
		expect(h.errors()[0].message).toContain("DRIFT_SERVICE_SECRET");
		expect(h.errors()[0].message).not.toContain("DRIFT_SERVICE_PORT");
	});

	test("start --reference runs SETUP then START, persists the run and shows the board", async () => {
		const h = createHarness();
		await h.runCommand("start --reference " + manifestPath + " --task write the abstract --checkpoint-every 5");
		expect(h.errors()).toHaveLength(0);
		expect(service.requests.map(request => request.op)).toEqual([Op.SETUP, Op.START, Op.STATUS]);
		expect(service.manifestPath).toBe(manifestPath);
		expect(service.task).toBe("write the abstract");
		expect(service.requestsWithOp(Op.START)[0].checkpoint_every).toBe(5);
		expect(h.status.get("drift")).toBe("drift: STRICT e0");
		expect(h.board()).toContain("STRICT");
		expect(h.board()).toContain("backend reference · live workers BLOCKED");
		expect(h.board()).toContain("manifest " + fakeManifestSha256.slice(0, 12));
		const started = h.roomMessages("started");
		expect(started).toHaveLength(1);
		expect(started[0].message.content).toContain("[Drift] Run started");
		const entries = h.runStateEntries();
		expect(entries).toHaveLength(1);
		expect(entries[0].data.phase).toBe("STRICT");
		expect(entries[0].data.manifestSha256).toBe(fakeManifestSha256);
		expect(JSON.stringify(entries[0].data)).not.toContain(secret);
		expect(JSON.stringify(entries[0].data)).not.toContain(String(service.port));
	});

	test("assignments staged before start travel with SETUP and are refused during STRICT", async () => {
		const h = createHarness();
		await h.runCommand("assign 0 take the lemma");
		await h.runCommand("assign 1 take the proof");
		expect(h.errors()).toHaveLength(0);
		await h.runCommand("start --reference " + manifestPath);
		expect(service.assignments).toEqual({ "0": "take the lemma", "1": "take the proof" });
		const before = service.requests.length;
		await h.runCommand("assign 0 change of plan");
		expect(h.errors()).toHaveLength(1);
		expect(h.errors()[0].message).toContain("STRICT");
		expect(service.requests).toHaveLength(before);
		expect(service.assignments["0"]).toBe("take the lemma");
	});

	test("start --reference is refused during STRICT because it carries free text", async () => {
		const h = createHarness();
		await h.runCommand("start --reference " + manifestPath);
		await h.runCommand("start --reference /elsewhere/run.yaml --task something else");
		expect(h.errors()).toHaveLength(1);
		expect(h.errors()[0].message).toContain("STRICT");
		expect(service.requestsWithOp(Op.SETUP)).toHaveLength(1);
	});

	test("typed commands refuse free text arguments", async () => {
		const h = createHarness();
		await h.runCommand("start --reference " + manifestPath);
		const before = service.requests.length;
		await h.runCommand("tick three");
		await h.runCommand("pause now");
		await h.runCommand("mail zero");
		expect(h.errors()).toHaveLength(3);
		for (const error of h.errors()) expect(error.message).toMatch(/integer|free text/);
		expect(service.requests).toHaveLength(before);
	});

	test("tick advances the epoch and renders the board and status line", async () => {
		const h = createHarness();
		await h.runCommand("start --reference " + manifestPath);
		await h.runCommand("tick 3");
		expect(h.errors()).toHaveLength(0);
		expect(service.requestsWithOp(Op.TICK)[0].epochs).toBe(3);
		expect(service.epoch).toBe(3);
		expect(h.status.get("drift")).toBe("drift: STRICT e3");
		const board = h.widgets.get("drift-board") ?? [];
		expect(board[0]).toContain("drift · STRICT · epoch 3");
		expect(board.join("\n")).toContain("m0 ▸ e2 seq 3 pos 12 writer 1 · local 12 foreign 6 · mail sent 0 mail-foreign 0 · gates 3:open 7:open · mass 3:0.120");
		expect(board.join("\n")).toContain("mailbox ▸ attended_not_proven 0 · cancelled 0 · causally_incorporated 0 · expired_without_incorporation 0 · pending 0 · rejected 0 · visible 0");
		expect(board.join("\n")).toContain("detectors ▸ nonfinite no · norm drift 3:0.010 · mass oscillation 3:0.200");
		expect(board.join("\n")).toContain("poisoned no · checkpoint —");
		// Nothing but enum names, hex, numbers and the plugin's own labels.
		expect(board.join("\n")).not.toContain(manifestPath);
		expect(board.join("\n")).not.toContain(secret);
		await h.runCommand("tick");
		expect(service.epoch).toBe(4);
		expect(h.status.get("drift")).toBe("drift: STRICT e4");
	});

});

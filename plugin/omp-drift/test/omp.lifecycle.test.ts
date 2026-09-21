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

	test("status refreshes the board without changing the run", async () => {
		const h = createHarness();
		await h.runCommand("start --reference " + manifestPath);
		await h.runCommand("tick 2");
		const before = service.epoch;
		await h.runCommand("status");
		expect(service.epoch).toBe(before);
		expect(service.requests.at(-1)?.op).toBe(Op.STATUS);
		expect(h.notifications.at(-1)?.message).toContain("STRICT at epoch 2");
	});

	test("pause sends the typed op and keeps the run STRICT", async () => {
		const h = createHarness();
		await h.runCommand("start --reference " + manifestPath);
		await h.runCommand("pause");
		expect(h.errors()).toHaveLength(0);
		expect(service.paused).toBe(true);
		expect(service.requestsWithOp(Op.PAUSE)).toHaveLength(1);
		expect(h.status.get("drift")).toBe("drift: STRICT e0");
	});

	test("mail sends the sender index and slot count and never any text", async () => {
		const h = createHarness();
		await h.runCommand("start --reference " + manifestPath);
		await h.runCommand("mail 0 2");
		expect(h.errors()).toHaveLength(0);
		const mail = service.requestsWithOp(Op.MAIL);
		expect(mail).toHaveLength(1);
		expect(mail[0].sender).toBe(0);
		expect(mail[0].slots).toBe(2);
		expect(Object.keys(mail[0]).sort()).toEqual(["auth", "id", "op", "sender", "slots"]);
		expect(h.board()).toContain("mail sent 1");
		await h.runCommand("mail 1");
		expect(service.requestsWithOp(Op.MAIL)[1].slots).toBe(1);
	});

	test("checkpoint reports the digest as a chat line", async () => {
		const h = createHarness();
		await h.runCommand("start --reference " + manifestPath);
		await h.runCommand("checkpoint");
		expect(h.errors()).toHaveLength(0);
		const lines = h.roomMessages("checkpoint");
		expect(lines).toHaveLength(1);
		expect(lines[0].message.content).toContain("[Drift] Checkpoint " + service.lastCheckpointSha256!.slice(0, 12));
		expect(lines[0].message.details?.sha256).toBe(service.lastCheckpointSha256);
		expect(h.board()).toContain("checkpoint " + service.lastCheckpointSha256!.slice(0, 12));
	});

	test("abort poisons the run, announces it and persists the phase", async () => {
		const h = createHarness();
		await h.runCommand("start --reference " + manifestPath);
		await h.runCommand("tick 2");
		await h.runCommand("abort");
		expect(service.requestsWithOp(Op.ABORT)).toHaveLength(1);
		expect(service.poisoned).toBe(true);
		expect(h.roomMessages("poisoned")).toHaveLength(1);
		expect(h.roomMessages("poisoned")[0].message.content).toContain("poisoned");
		expect(h.roomMessages("aborted")).toHaveLength(1);
		expect(h.roomMessages("phase")[0].message.content).toContain("STRICT → CLOSED");
		expect(h.board()).toContain("poisoned yes");
		expect(h.status.get("drift")).toBe("drift: CLOSED e2");
		const entries = h.runStateEntries();
		expect(entries.at(-1)?.data.phase).toBe("CLOSED");
		expect(entries.at(-1)?.data.manifestSha256).toBe(fakeManifestSha256);
	});

	test("complete closes the run and announces completion without outputs", async () => {
		const h = createHarness();
		await h.runCommand("start --reference " + manifestPath);
		await h.runCommand("tick 1");
		await h.runCommand("complete");
		expect(h.errors()).toHaveLength(0);
		expect(service.phase).toBe("CLOSED");
		expect(h.roomMessages("completed")).toHaveLength(1);
		expect(h.roomMessages("phase").at(-1)?.message.content).toContain("STRICT → CLOSED");
		expect(h.status.get("drift")).toBe("drift: CLOSED e1");
		expect(h.runStateEntries().at(-1)?.data.phase).toBe("CLOSED");
		// STRICT-only commands are now refused locally.
		const before = service.requests.length;
		await h.runCommand("tick");
		expect(h.errors()).toHaveLength(1);
		expect(service.requests).toHaveLength(before);
	});

	test("stop aborts a STRICT run and closes the socket", async () => {
		const h = createHarness();
		await h.runCommand("start --reference " + manifestPath);
		expect(service.openConnections).toBe(1);
		await h.runCommand("stop");
		expect(service.requests.at(-2)?.op).toBe(Op.ABORT);
		await service.waitForAllClosed();
		expect(service.openConnections).toBe(0);
		expect(h.roomMessages("stopped")).toHaveLength(1);
		expect(h.status.get("drift")).toBeUndefined();
		expect(h.widgets.get("drift-board")).toBeUndefined();
	});

	test("stop after complete closes the socket without a second ABORT", async () => {
		const h = createHarness();
		await h.runCommand("start --reference " + manifestPath);
		await h.runCommand("complete");
		await h.runCommand("stop");
		expect(service.requestsWithOp(Op.ABORT)).toHaveLength(0);
		await service.waitForAllClosed();
		expect(service.openConnections).toBe(0);
	});

});

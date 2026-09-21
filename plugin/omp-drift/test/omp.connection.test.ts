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

	test("a wrong secret surfaces an AUTH error and closes the connection", async () => {
		const h = createHarness();
		setEnv(service.port, "not-the-secret");
		await h.runCommand("start --reference " + manifestPath);
		expect(h.errors()).toHaveLength(1);
		expect(h.errors()[0].message).toContain("AUTH");
		expect(h.errors()[0].message).not.toContain("not-the-secret");
		expect(service.rejectedLines).toHaveLength(1);
		await service.waitForAllClosed();
		expect(service.openConnections).toBe(0);
		expect(h.board()).toContain("disconnected");
		expect(h.runStateEntries()).toHaveLength(0);
	});

	test("an unauthenticated response closes the connection with an error", async () => {
		await service.close();
		service = await FakeService.start({ secret, corruptResponseAuth: true });
		setEnv(service.port);
		const h = createHarness();
		await h.runCommand("start --reference " + manifestPath);
		expect(h.errors()).toHaveLength(1);
		expect(h.errors()[0].message).toContain("[AUTH]");
		await service.waitForAllClosed();
		expect(service.openConnections).toBe(0);
		expect(h.roomMessages("disconnected")[0].message.details?.code).toBe("AUTH");
	});

	test("a malformed response closes the connection with an error", async () => {
		await service.close();
		service = await FakeService.start({ secret, malformedResponse: true });
		setEnv(service.port);
		const h = createHarness();
		await h.runCommand("start --reference " + manifestPath);
		expect(h.errors()).toHaveLength(1);
		expect(h.errors()[0].message).toContain("[MALFORMED]");
		await service.waitForAllClosed();
		expect(service.openConnections).toBe(0);
	});

	test("a refused connection is reported without a run state", async () => {
		const closedPort = service.port;
		await service.close();
		service = await FakeService.start({ secret });
		setEnv(closedPort);
		const h = createHarness();
		await h.runCommand("start --reference " + manifestPath);
		expect(h.errors()).toHaveLength(1);
		expect(h.errors()[0].message).toContain("could not connect");
		expect(h.status.get("drift")).toBeUndefined();
	});

	test("the turn-start layer names only the phase and epoch", async () => {
		const h = createHarness();
		expect(await h.emitBeforeAgentStart(["base"])).toBeUndefined();
		await h.runCommand("start --reference " + manifestPath);
		await h.runCommand("tick 2");
		await h.runCommand("checkpoint");
		const result = await h.emitBeforeAgentStart(["base prompt"]);
		expect(result?.systemPrompt[0]).toBe("base prompt");
		expect(result?.systemPrompt).toHaveLength(2);
		const layer = result?.systemPrompt[1] ?? "";
		expect(layer).toContain("phase STRICT");
		expect(layer).toContain("reference run");
		expect(layer).toContain("integration is BLOCKED");
		expect(layer).toContain("epoch 2");
		expect(layer).not.toContain(fakeManifestSha256.slice(0, 8));
		expect(layer).not.toContain(service.lastCheckpointSha256!.slice(0, 8));
		expect(layer).not.toContain(manifestPath);
		expect(layer).not.toContain("mail");
		expect(layer).not.toContain("gate");
	});

	test("a resumed session shows the last known phase without reconnecting", async () => {
		const h = createHarness();
		h.sessionManager.branch.push({
			type: "custom",
			customType: "drift-run-state",
			data: { manifestSha256: fakeManifestSha256, phase: "STRICT", epoch: 7 },
		});
		await h.fireSessionEvent("session_start");
		expect(service.connectionsOpened).toBe(0);
		expect(h.status.get("drift")).toBe("drift: STRICT e7 (resumed)");
		expect(h.board()).toContain("resumed, not connected");
		expect(h.roomMessages("resumed")).toHaveLength(1);
		const before = service.requests.length;
		await h.runCommand("tick");
		expect(h.errors()[0].message).toContain("not connected");
		expect(service.requests).toHaveLength(before);
		await h.runCommand("stop");
		expect(h.status.get("drift")).toBeUndefined();
	});

	test("session shutdown closes the socket without aborting the run", async () => {
		const h = createHarness();
		await h.runCommand("start --reference " + manifestPath);
		await h.fireSessionEvent("session_shutdown");
		await service.waitForAllClosed();
		expect(service.openConnections).toBe(0);
		expect(service.requestsWithOp(Op.ABORT)).toHaveLength(0);
		expect(service.phase).toBe("STRICT");
		expect(h.status.get("drift")).toBeUndefined();
	});

	test("argument completions list the subcommands", () => {
		const h = createHarness();
		expect(h.completions("")?.map(item => item.value)).toEqual([
			"start", "assign", "tick", "pause", "status", "checkpoint", "mail", "complete", "abort", "stop",
		]);
		expect(h.completions("st")?.map(item => item.value)).toEqual(["start", "status", "stop"]);
		expect(h.completions("tick 3")).toBeNull();
		expect(h.completions("zzz")).toBeNull();
	});
});

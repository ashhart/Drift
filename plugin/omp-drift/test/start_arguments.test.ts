import { describe, expect, test } from "bun:test";
import { parseStartArguments } from "../src/start_arguments";

describe("reference-only startup", () => {
	test("live requests cannot be enabled by a flag inside task text", () => {
		for (const input of ["run.json", "run.json --task --reference", "--backend live run.json", "--reference-live run.json"]) {
			expect(parseStartArguments(input)).toContain("BLOCKED");
		}
	});

	test("reference opt-in preserves the setup and bounded checkpoint interval", () => {
		expect(parseStartArguments("--reference run.json --task implement the parser --checkpoint-every 4")).toEqual({
			manifest: "run.json", task: "implement the parser", checkpointEvery: 4,
		});
		expect(parseStartArguments("--reference")).toContain("Usage");
		expect(parseStartArguments("--reference run.json --checkpoint-every 0")).toContain("positive integer");
	});
});

import { test, expect, spyOn } from "bun:test";
import * as fs from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { writeBoundary } from "../../../scripts/omp/project_boundary_files.mjs";

test("a ready record appears only after its complete write and sync", () => {
  const root = fs.realpathSync(fs.mkdtempSync(join(tmpdir(), "boundary-publish-"))), path = join(root, "ready.json");
  const originalWrite = fs.writeSync, originalSync = fs.fsyncSync; let written = false, synced = false;
  const writer = spyOn(fs, "writeSync").mockImplementation(((...args: any[]) => {
    expect(fs.existsSync(path)).toBe(false); written = true; return (originalWrite as Function)(...args);
  }) as typeof fs.writeSync);
  const sync = spyOn(fs, "fsyncSync").mockImplementation(fd => {
    expect(written).toBe(true); expect(fs.existsSync(path)).toBe(false); originalSync(fd); synced = true;
  });
  try {
    const hash = writeBoundary(path, { v: 1, phase: "ready" });
    expect(synced).toBe(true); expect(hash).toBe(createHash("sha256").update(fs.readFileSync(path)).digest("hex"));
    expect(fs.readdirSync(root)).toEqual(["ready.json"]); expect(fs.statSync(path).mode & 0o777).toBe(0o600);
    writer.mockRestore(); sync.mockRestore();
    expect(() => writeBoundary(path, { v: 2 })).toThrow(); expect(JSON.parse(fs.readFileSync(path, "utf8")).v).toBe(1);
  } finally { writer.mockRestore(); sync.mockRestore(); fs.rmSync(root, { recursive: true, force: true }); }
});

test("write failure leaves no observable ready record or temporary file", () => {
  const root = fs.realpathSync(fs.mkdtempSync(join(tmpdir(), "boundary-failed-"))), path = join(root, "ready.json");
  const writer = spyOn(fs, "writeSync").mockImplementation(() => { throw new Error("write failed"); });
  try {
    expect(() => writeBoundary(path, { v: 1 })).toThrow(); expect(fs.readdirSync(root)).toEqual([]);
  } finally { writer.mockRestore(); fs.rmSync(root, { recursive: true, force: true }); }
});

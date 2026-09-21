// Drive a REAL Python service through the plugin's own client: proves both sides sign identical bytes.
import { ServiceClient } from "../src/client";
import { Op } from "../src/protocol";

const port = Number(process.env.DRIFT_SERVICE_PORT);
const secret = process.env.DRIFT_SERVICE_SECRET ?? "";
const manifest = process.argv[2];
const client = new ServiceClient({ host: "127.0.0.1", port, secret, timeoutMs: 60_000 });
await client.connect();
const out: Record<string, unknown> = {};
out.status0 = await client.request(Op.STATUS);
out.setup = await client.request(Op.SETUP, { manifest, task: "build the parser", assignments: { "0": "cli", "1": "tests" } });
out.start = await client.request(Op.START, { checkpoint_every: 2 });
out.tick = await client.request(Op.TICK, { epochs: 3 });
out.status1 = await client.request(Op.STATUS);
out.mail = await client.request(Op.MAIL, { sender: 0, slots: 2 });
out.tick2 = await client.request(Op.TICK, { epochs: 1 });
out.checkpoint = await client.request(Op.CHECKPOINT);
out.status2 = await client.request(Op.STATUS);
out.complete = await client.request(Op.COMPLETE);
client.close();
console.log(JSON.stringify(out));

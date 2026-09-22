"""Run the tap, translate and inject path between two toy models on one CPU, with its controls.

Random weights: this shows that the channel carries information and that the controls are
clean. It is NOT a language result and says nothing about recall or understanding.
"""
import argparse
import json

import torch

from drift.adapters.toy import DenseAdapter
from drift.eval.e1 import E1Unit, run_e1
from drift.runtime.decoder import FrozenDecoder
from drift.runtime.toy import ToyModel
from drift.translate.pool import Layout, PoolFormat, Translator

FORMAT = PoolFormat("pool.v1", 3, 8, "cc" * 32)
ARMS = (("floor", "the receiver answers from the question alone"),
        ("ceiling", "the receiver rereads its own context first"),
        ("foreign", "the donor read the context; the receiver gets translated entries"),
        ("hard_off", "the same publication with the gate forced to zero"),
        ("wrong_context", "entries from a different unit, as a leakage control"))


def build():
    torch.manual_seed(0)
    donor = DenseAdapter(FrozenDecoder(ToyModel(seed=1, kvheads=2, dim=6, layers=3)))
    receiver = DenseAdapter(FrozenDecoder(ToyModel(seed=2, kvheads=1, dim=4, layers=3)))
    writer = Translator("donor", Layout("kv_split", 2, 6), {0: 0, 1: 1, 2: 2}, FORMAT, kind="mlp", rank=8)
    reader = Translator("receiver", Layout("kv_split", 1, 4), {0: 0, 1: 1, 2: 2}, FORMAT, kind="mlp", rank=8)
    with torch.no_grad():
        for parameter in list(writer.parameters()) + list(reader.parameters()):
            parameter.mul_(0.2)
    return donor, receiver, writer, reader


def units(count, generator):
    return [E1Unit(f"u{index}", torch.randint(1, 40, (6,), generator=generator),
                   torch.randint(1, 40, (6,), generator=generator),
                   torch.randint(1, 40, (2,), generator=generator)) for index in range(count)]


def summarise(results):
    changed = sum(row["foreign_ids"] != row["floor_ids"] for row in results)
    identical = sum(row["hard_off_ids"] == row["floor_ids"] for row in results)
    return {"units": len(results), "foreign_tokens_published": results[0]["foreign_tokens"],
            "foreign_changed_the_output": changed,
            "hard_off_reproduced_the_floor_exactly": identical,
            "wrong_context_changed_the_output": sum(row["wrong_context_ids"] != row["floor_ids"] for row in results),
            "channel_live": changed > 0, "controls_clean": identical == len(results)}


def render(results, summary):
    print("Drift local link demo: two toy models, one CPU, no network and no MCDMA\n")
    print("Arms")
    for name, description in ARMS:
        print(f"  {name:<14} {description}")
    print("\nPer unit, tokens the receiver consumed")
    header = f"  {'unit':<6}" + "".join(f"{name:>15}" for name, _ in ARMS)
    print(header)
    for row in results:
        counts = row["local_tokens"]
        print(f"  {row['id']:<6}" + "".join(f"{counts[name]:>15}" for name, _ in ARMS))
    print(f"\n  donor published {summary['foreign_tokens_published']} translated entries per unit")
    print(f"  foreign output differed from floor  : {summary['foreign_changed_the_output']}/{summary['units']}")
    print(f"  hard_off reproduced floor exactly   : {summary['hard_off_reproduced_the_floor_exactly']}/{summary['units']}")
    print(f"\n  channel live   : {summary['channel_live']}")
    print(f"  controls clean : {summary['controls_clean']}")
    print("\nBoth models keep frozen random weights, so no text, token id or answer crosses the")
    print("channel and none of this is a language result. It shows the publication path is live")
    print("and that forcing the gate to zero returns the receiver to its exact native output.")
    print("The measured cross-family result is in evidence/note-vs-memory/.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--units", type=int, default=3, help="how many donor/receiver pairs to run")
    parser.add_argument("--new-tokens", type=int, default=4, help="tokens the receiver generates per arm")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--json", action="store_true", help="print the summary as JSON instead of a table")
    arguments = parser.parse_args()
    if arguments.units < 2:
        parser.error("E1 needs at least two units so the wrong-context control has another unit to borrow from")
    donor, receiver, writer, reader = build()
    generator = torch.Generator().manual_seed(arguments.seed)
    results = run_e1(donor, receiver, writer, reader, units(arguments.units, generator),
                     new_tokens=arguments.new_tokens)
    summary = summarise(results)
    print(json.dumps(summary, indent=2)) if arguments.json else render(results, summary)
    return 0 if summary["channel_live"] and summary["controls_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

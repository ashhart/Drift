"""Parse and validate options for the exploratory live coordinator."""
import argparse
import math
from pathlib import Path
from drift.eval.stream_repetition import RepetitionPolicy


def parse_run_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--forward", type=Path, default=None)
    parser.add_argument("--forward-recipe", choices=("legacy", "base", "fanout", "v4"), default="legacy")
    parser.add_argument("--forward-manifest", type=Path)
    parser.add_argument("--reverse", type=Path, default=Path("local/live/stacked2_rev.npz"))
    parser.add_argument("--gain-forward", type=float)
    parser.add_argument("--gain-reverse", type=float, default=1.0)
    parser.add_argument("--copies", type=int, default=3, help="copies of each Qwen entry written into GLM's reserved span")
    parser.add_argument("--reserve", type=int, default=1024)
    parser.add_argument("--epoch-tokens", type=int, default=16)
    parser.add_argument("--max-new", type=int, default=260)
    parser.add_argument("--max-pending-publications", type=int, default=8)
    parser.add_argument("--max-publication-rows", type=int, default=4096)
    parser.add_argument("--no-link", action="store_true", help="control: run both models at once but move nothing")
    parser.add_argument("--io-timeout", type=float, default=120, help="maximum seconds per file transfer or setup command")
    parser.add_argument("--max-seconds", type=float, default=1800, help="maximum live-session wall time before abort")
    parser.add_argument("--repetition-threshold", type=float, default=0.6, help="uncalibrated engineering abort threshold; freeze on development data before trials")
    parser.add_argument("--repetition-window", type=int, default=128)
    parser.add_argument("--repetition-min-words", type=int, default=32)
    args = parser.parse_args()
    if args.forward_recipe == "legacy":
        if args.forward_manifest is not None:
            parser.error("--forward-manifest requires an explicit forward recipe")
        args.forward = args.forward or Path("local/live/stacked2.npz")
        args.gain_forward = 1.5 if args.gain_forward is None else args.gain_forward
    elif args.forward_manifest is None or args.forward is not None:
        parser.error("explicit forward recipes require --forward-manifest and prohibit --forward")
    if any(value is not None and not math.isfinite(value) for value in (args.gain_forward, args.gain_reverse)):
        parser.error("gain powers must be finite")
    try:
        repetition_policy = RepetitionPolicy(window_words=args.repetition_window, min_words=args.repetition_min_words,
                                             threshold=args.repetition_threshold)
    except ValueError as error:
        parser.error(str(error))
    if any(not math.isfinite(v) or v <= 0 for v in (args.io_timeout, args.max_seconds)):
        parser.error("timeouts must be finite and positive")
    if min(args.copies, args.reserve, args.epoch_tokens, args.max_new, args.max_publication_rows) <= 0:
        parser.error("copies, reserve, epoch-tokens and max-new must be positive")
    return args, repetition_policy, parser

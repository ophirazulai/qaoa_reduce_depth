"""Command-line interface; importing this module does not import Qiskit."""

import argparse
import math
import sys

from .compiler import DEFAULT_TARGET, DEFAULT_WEIGHTS, configure_threads


def positive_int(value):
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_weight(value):
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weight must be finite and nonnegative") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("weight must be finite and nonnegative")
    return parsed


def parser():
    root = argparse.ArgumentParser(description="Search seeds and SABRE weights for minimum D2")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a complete local or LSF sweep")
    run.add_argument("--circuits", default="circuits", help="input QPY directory (recursive)")
    run.add_argument("--output", default="output", help="new or empty output directory")
    run.add_argument("--seeds", type=positive_int, default=150, help="number of seeds (default: 150)")
    run.add_argument("--seed-start", type=int, default=0)
    run.add_argument("--weights", type=nonnegative_weight, nargs="+", default=list(DEFAULT_WEIGHTS))
    run.add_argument("--target", default=DEFAULT_TARGET, help="BackendV2 module:factory (FakeBoston)")
    run.add_argument("--optimization-level", type=int, choices=range(4), default=3)
    run.add_argument("--sabre-trials", type=positive_int, default=4,
                     help="fixed internal layout/swap trials, independent of machine CPUs")
    run.add_argument("--executor", choices=("local", "lsf"), default="local")
    run.add_argument("--workers", type=positive_int, default=5,
                     help="local processes; ignored for LSF (default: 5)")
    run.add_argument("--transpilations-per-job", type=positive_int, default=5,
                     help="trials per LSF job/local batch (default: 5)")
    run.add_argument("--lsf-queue", default="normal")
    run.add_argument("--lsf-project")
    run.add_argument("--lsf-wall-minutes", type=positive_int, default=120)
    run.add_argument("--lsf-memory-gb", type=positive_int, default=10)
    run.add_argument("--lsf-memory-unit", choices=("KB", "MB", "GB"), default="GB",
                     help="site's LSF memory unit for -M and rusage (default: GB)")
    run.add_argument("--bsub", default="bsub")
    worker = commands.add_parser("worker", help="internal: execute a planned batch")
    worker.add_argument("--plan", required=True)
    worker.add_argument("--shard", required=True, type=int)
    collect = commands.add_parser("collect", help="rebuild reports from completed batch results")
    collect.add_argument("--output", default="output")
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    configure_threads()
    from .sweep import collect, execute_shard, prepare, run_local

    try:
        if args.command == "worker":
            return 0 if execute_shard(args.plan, args.shard) else 1
        if args.command == "collect":
            return 0 if collect(args.output) else 1
        args.weights = sorted(set(args.weights))
        if args.seed_start < 0 or args.seed_start + args.seeds > 2**32:
            root.error("seed range must fit unsigned 32-bit integers")
        if args.executor == "lsf":
            import shutil
            if shutil.which(args.bsub) is None:
                root.error(f"LSF executable not found: {args.bsub}")
        plan_path = prepare(args)
        print(f"Prepared sweep: {plan_path}", flush=True)
        if args.executor == "local":
            run_local(plan_path, args.workers)
        else:
            from .lsf import run_lsf
            run_lsf(plan_path)
        complete = collect(plan_path.parent)
        print(f"{'Complete' if complete else 'INCOMPLETE'}: {plan_path.parent / 'report.md'}")
        return 0 if complete else 1
    except KeyboardInterrupt:
        print("Interrupted. Existing results are retained; use collect after workers finish. "
              "Accepted LSF jobs may still be running; inspect bjobs/bsub.log.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

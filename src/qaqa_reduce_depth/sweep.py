"""Plan, execute, and collect an exhaustive finite transpilation sweep."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import time
import uuid
from urllib.parse import quote

from .compiler import QISKIT_REVISION, configure_threads, make_manager, measure, rank, resolve_target
from .storage import atomic_write, read_json, sha256, write_json


def discover(directory):
    paths = []
    for root, directories, files in os.walk(directory, followlinks=False):
        directories[:] = sorted(d for d in directories if not (Path(root) / d).is_symlink())
        for name in sorted(files):
            path = Path(root) / name
            if path.suffix.lower() == ".qpy":
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"Input must be a regular QPY file: {path}")
                paths.append(path)
    if not paths:
        raise ValueError(f"No .qpy inputs found in {directory}")
    return sorted(paths)


def prepare(args):
    configure_threads()
    source, output = Path(args.circuits).resolve(), Path(args.output).resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Input and output directories must not overlap")
    if not source.is_dir():
        raise ValueError(f"Input directory does not exist: {source}")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; use collect for an existing run")
    paths = discover(source)
    backend = resolve_target(args.target)
    make_manager(backend, args.seed_start, args.weights[0], args.optimization_level,
                 args.sabre_trials)
    created = not output.exists()
    output.mkdir(parents=True, exist_ok=True)
    try:
        return write_plan(args, source, output, paths, backend)
    except BaseException:
        # A failed preparation is not a run. Remove what it created (the
        # directory was verified empty above) so the same --output can be
        # reused once the input problem is fixed.
        shutil.rmtree(output / "_work", ignore_errors=True)
        (output / "plan.json").unlink(missing_ok=True)
        if created:
            try:
                output.rmdir()
            except OSError:
                pass
        raise


def write_plan(args, source, output, paths, backend):
    from qiskit import qpy
    from qiskit.circuit import ControlFlowOp

    inputs = []
    for file_index, path in enumerate(paths):
        snapshot = output / "_work" / "inputs" / f"{file_index:06d}.qpy"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, snapshot)
        try:
            with snapshot.open("rb") as stream:
                circuits = qpy.load(stream)
        except Exception as exc:
            raise ValueError(f"Cannot read QPY file {path}: {exc}") from exc
        if not circuits:
            raise ValueError(f"QPY file contains no circuits: {path}")
        digest = sha256(snapshot)
        for index, circuit in enumerate(circuits):
            if circuit.num_qubits > backend.num_qubits:
                raise ValueError(f"{path} circuit {index}: {circuit.num_qubits} qubits exceeds "
                                 f"target capacity {backend.num_qubits}")
            if any(isinstance(item.operation, ControlFlowOp)
                   or getattr(item.operation, "condition", None) is not None
                   for item in circuit.data):
                raise ValueError(f"{path} circuit {index}: only static circuits are supported")
            inputs.append({
                "case_id": f"f{file_index:06d}-c{index:04d}",
                "source": path.relative_to(source).as_posix(), "circuit_index": index,
                "name": circuit.name, "num_qubits": circuit.num_qubits,
                "snapshot": snapshot.relative_to(output).as_posix(), "sha256": digest,
            })
    tasks = []
    for case in inputs:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            for weight in [None, *args.weights]:
                tasks.append({"task_id": len(tasks), "case_id": case["case_id"],
                              "seed": seed, "weight": weight})
    # The same bounded batches are also useful locally: avoid repeatedly
    # initializing backends, while parallelizing even a single input circuit.
    size = args.transpilations_per_job
    plan = {
        "schema_version": 1, "run_id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": vars(args), "inputs": inputs,
        "shards": [tasks[start:start + size] for start in range(0, len(tasks), size)],
        "expected_qiskit_revision": QISKIT_REVISION,
        "versions": {name: importlib.metadata.version(name)
                     for name in ("qiskit", "qiskit-ibm-runtime")},
        "qiskit_direct_url": importlib.metadata.distribution("qiskit").read_text("direct_url.json"),
    }
    write_json(output / "plan.json", plan)
    return output / "plan.json"


def shard_directory(output, shard_index):
    return Path(output) / "_work" / "shards" / f"{shard_index:06d}"


def execute_shard(plan_path, shard_index):
    configure_threads()
    from qiskit import qpy

    plan_path = Path(plan_path).resolve()
    output, plan = plan_path.parent, read_json(plan_path)
    config = plan["config"]
    if not 0 <= shard_index < len(plan["shards"]):
        raise ValueError("Shard index is out of range")
    directory = shard_directory(output, shard_index)
    directory.mkdir(parents=True, exist_ok=True)
    # Refuse duplicate concurrent/repeated execution; results belong to one run.
    with (directory / "started").open("x") as stream:
        stream.write(plan["run_id"])
    cases = {case["case_id"]: case for case in plan["inputs"]}
    backend = resolve_target(config["target"])
    rows, winners, cache = [], {}, {}
    for task in plan["shards"][shard_index]:
        started = time.monotonic()
        row = dict(task)
        try:
            case = cases[task["case_id"]]
            snapshot = output / case["snapshot"]
            if case["snapshot"] not in cache:
                if sha256(snapshot) != case["sha256"]:
                    raise ValueError("Input snapshot checksum changed")
                with snapshot.open("rb") as stream:
                    # Retain only the current QPY bundle to bound worker memory.
                    cache = {case["snapshot"]: qpy.load(stream)}
            circuit = cache[case["snapshot"]][case["circuit_index"]]
            manager = make_manager(backend, task["seed"], task["weight"],
                                   config["optimization_level"], config["sabre_trials"])
            compiled = manager.run(circuit.copy())
            row.update(status="ok", **measure(compiled))
            previous = winners.get(task["case_id"])
            if previous is None or rank(row) < rank(previous):
                artifact = directory / f"{task['case_id']}.qpy"
                atomic_write(artifact, lambda stream: qpy.dump(compiled, stream), binary=True)
                winners[task["case_id"]] = {
                    **row, "artifact": artifact.relative_to(output).as_posix(),
                    "artifact_sha256": sha256(artifact),
                }
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        row["seconds"] = time.monotonic() - started
        rows.append(row)
    write_json(directory / "result.json", {
        "run_id": plan["run_id"], "shard_index": shard_index,
        "rows": rows, "winners": list(winners.values()),
    })
    return all(row["status"] == "ok" for row in rows)


def _ignore_interrupt():
    # Ctrl-C signals the whole foreground process group. Only the coordinator
    # reacts to it, and then stops the workers itself.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_local(plan_path, workers):
    plan = read_json(plan_path)
    failures = []
    unrelated = set(multiprocessing.active_children())
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_ignore_interrupt,
                               mp_context=multiprocessing.get_context("spawn"))
    try:
        futures = {pool.submit(execute_shard, str(plan_path), index): index
                   for index in range(len(plan["shards"]))}
        for count, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            try:
                if not future.result():
                    failures.append(f"shard {index}: one or more transpilation failures")
            except Exception as exc:
                failures.append(f"shard {index}: {type(exc).__name__}: {exc}")
            print(f"Completed {count}/{len(futures)} batches", flush=True)
    except BaseException:
        # Interrupted: a pool left to shut down normally would silently run
        # every remaining batch before this process could exit.
        for process in set(multiprocessing.active_children()) - unrelated:
            process.terminate()
        raise
    finally:
        pool.shutdown(cancel_futures=True)
    write_json(plan_path.parent / "execution.json", {"executor": "local", "errors": failures})


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def best_output_paths(inputs):
    """Keep source names, adding collision-free indices for QPY bundles."""
    counts = Counter(case["source"] for case in inputs)
    # Reserve files and directories, including case variants on macOS/Windows.
    reserved = {path.as_posix().casefold() for name in counts
                for path in (Path(name), *Path(name).parents)}
    paths = {}
    for case in inputs:
        source = Path(case["source"])
        destination = source
        if counts[case["source"]] > 1:
            stem = f"{source.stem}__circuit_{case['circuit_index']}"
            destination = source.with_name(f"{stem}{source.suffix}")
            suffix = 1
            while destination.as_posix().casefold() in reserved:
                destination = source.with_name(f"{stem}__{suffix}{source.suffix}")
                suffix += 1
            reserved.add(destination.as_posix().casefold())
        paths[case["case_id"]] = Path("best") / destination
    return paths


def collect(output):
    """Only publish a minimum when every planned trial for that input succeeded."""
    output = Path(output).resolve()
    plan = read_json(output / "plan.json")
    rows, candidates, issues = [], [], []
    expected = Counter(task["case_id"] for shard in plan["shards"] for task in shard)
    for index, tasks in enumerate(plan["shards"]):
        result_path = shard_directory(output, index) / "result.json"
        if not result_path.exists():
            issues.append(f"Missing result for shard {index}")
            continue
        result = read_json(result_path)
        if result["run_id"] != plan["run_id"] or result["shard_index"] != index:
            raise ValueError(f"Result identity mismatch in shard {index}")
        if [{key: row[key] for key in task} for row, task in zip(result["rows"], tasks)] != tasks:
            raise ValueError(f"Result tasks mismatch in shard {index}")
        if len(result["rows"]) != len(tasks):
            raise ValueError(f"Result count mismatch in shard {index}")
        successful = {row["task_id"]: row for row in result["rows"] if row["status"] == "ok"}
        for winner in result["winners"]:
            row = successful.get(winner["task_id"])
            if row is None or any(winner[key] != row[key]
                                  for key in ("case_id", "seed", "weight", "d2", "n2")):
                raise ValueError(f"Winner does not match a successful task in shard {index}")
            artifact = (output / winner["artifact"]).resolve()
            if not artifact.is_relative_to(shard_directory(output, index)):
                raise ValueError("Winner artifact escapes its shard")
            if sha256(artifact) != winner["artifact_sha256"]:
                raise ValueError(f"Winner artifact checksum changed in shard {index}")
        rows.extend(result["rows"])
        candidates.extend(result["winners"])
    complete, selected = True, []
    output_paths = best_output_paths(plan["inputs"])
    success = Counter(row["case_id"] for row in rows if row["status"] == "ok")
    for case in plan["inputs"]:
        case_id = case["case_id"]
        item = {**case, "successful_trials": success[case_id], "expected_trials": expected[case_id]}
        choices = [row for row in candidates if row["case_id"] == case_id]
        if success[case_id] != expected[case_id] or not choices:
            complete = False
            item["status"] = "incomplete"
        else:
            best = min(choices, key=rank)
            case_rows = [row for row in rows if row["case_id"] == case_id]
            # Guard against a malformed result that omitted the actual minimum.
            all_best = min(case_rows, key=rank)
            if best["task_id"] != all_best["task_id"]:
                raise ValueError(f"Shard winners omit the minimum for {case_id}")
            # A shard retains only its overall winner's artifact, so compare
            # the two heuristic modes using all trial measurements.
            baseline = min((row for row in case_rows if row["weight"] is None), key=rank)
            wait_time = min((row for row in case_rows if row["weight"] is not None), key=rank)
            destination = output / output_paths[case_id]
            atomic_write(destination, lambda stream: stream.write(
                (output / best["artifact"]).read_bytes()), binary=True)
            item.update(status="complete", seed=best["seed"], weight=best["weight"],
                        d2=best["d2"], n2=best["n2"],
                        best_baseline_d2=baseline["d2"], best_wait_time_d2=wait_time["d2"],
                        best_baseline_n2=baseline["n2"], best_wait_time_n2=wait_time["n2"],
                        d2_ratio=wait_time["d2"] / baseline["d2"] if baseline["d2"] else None,
                        n2_ratio=wait_time["n2"] / baseline["n2"] if baseline["n2"] else None,
                        output=destination.relative_to(output).as_posix())
        selected.append(item)
    errors = [row for row in rows if row["status"] != "ok"]
    summary = {"run_id": plan["run_id"], "complete": complete,
               "inputs": selected, "errors": errors, "issues": issues}
    write_json(output / "results.json", summary)
    write_json(output / "trials.json", rows)
    config = plan["config"]
    lines = [
        "# Minimum two-qubit-depth transpilation", "",
        f"Status: **{'complete' if complete else 'INCOMPLETE'}**", "",
        f"Target: `{_cell(config['target'])}`. Optimization level: {config['optimization_level']}.",
        f"Seeds: {config['seed_start']}..{config['seed_start'] + config['seeds'] - 1}. "
        f"Weights: {', '.join(format(w, '.8g') for w in config['weights'])}; plus baseline.",
        "Baseline: decay heuristic, wait time disabled. Weighted: wait-time heuristic, no decay.",
        "D2: longest dependency path counting two-qubit gates. N2: total two-qubit gates.",
        "D2 ratio = best wait-time D2 / best baseline D2 across all requested trials per input. "
        "N2 ratio = wait-time N2 / baseline N2 for those same best trials, "
        "selected by D2 and then N2. Below 1 favors wait time; above 1 favors baseline; "
        "1 means equal values.",
        "Weight 0 belongs to the wait-time group. A ratio is N/A when its baseline value is zero; "
        "comparisons are omitted for incomplete inputs.",
        "Tie order: D2, N2, baseline, smaller weight, smaller seed.",
        "Minimum means the best of the requested trials, not a global optimum.", "",
        "| Input file | Circuit index | Status | Seed | Weight | D2 | N2 | Best baseline D2 | "
        "Best wait-time D2 | D2 ratio (wait/baseline) | Best baseline N2 | Best wait-time N2 | "
        "N2 ratio (wait/baseline) | Trials | QPY |",
        "| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: | --- |",
    ]
    for row in selected:
        ready = row["status"] == "complete"
        weight = ("baseline" if row.get("weight") is None else format(row["weight"], ".8g"))
        artifact = f"[QPY]({quote(row['output'])})" if ready else "—"
        d2_ratio = n2_ratio = "—"
        if ready:
            d2_ratio = "N/A" if row["d2_ratio"] is None else format(row["d2_ratio"], ".6g")
            n2_ratio = "N/A" if row["n2_ratio"] is None else format(row["n2_ratio"], ".6g")
        lines.append("| " + " | ".join(map(_cell, [
            row["source"], row["circuit_index"], row["status"], row.get("seed", "—"),
            weight if ready else "—", row.get("d2", "—"), row.get("n2", "—"),
            row.get("best_baseline_d2", "—"), row.get("best_wait_time_d2", "—"), d2_ratio,
            row.get("best_baseline_n2", "—"), row.get("best_wait_time_n2", "—"), n2_ratio,
            f"{row['successful_trials']}/{row['expected_trials']}", artifact,
        ])) + " |")
    if issues or errors:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {_cell(issue)}" for issue in issues)
        lines.extend(f"- Task {row['task_id']}: {_cell(row['error'])}" for row in errors)
    lines.extend(["", "See `plan.json` for the configuration, input checksums and dependency versions,",
                  "`trials.json` for all trial measurements, and `results.json` for selected outputs.", ""])
    atomic_write(output / "report.md", lambda stream: stream.write("\n".join(lines)))
    return complete

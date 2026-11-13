import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from qiskit import qpy
from qiskit.quantum_info import Operator

from qaqa_reduce_depth import lsf
from qaqa_reduce_depth.cli import main, parser
from qaqa_reduce_depth.compiler import DEFAULT_WEIGHTS, measure, rank
from qaqa_reduce_depth.lsf import job_script, submission_command
from qaqa_reduce_depth.storage import read_json, sha256, write_json
from qaqa_reduce_depth.sweep import collect, execute_shard, prepare, shard_directory


def arguments(inputs, output, *extra):
    return ["run", "--circuits", str(inputs), "--output", str(output),
            "--target", "tests.support:backend", "--seeds", "2", "--weights", "0.00001", "0.001",
            "--sabre-trials", "1", "--workers", "2", "--transpilations-per-job", "5", *extra]


def assert_result(output):
    plan = read_json(output / "plan.json")
    summary = read_json(output / "results.json")
    trials = read_json(output / "trials.json")
    assert summary["complete"]
    assert len(trials) == 12
    assert len(summary["inputs"]) == 2
    for selected in summary["inputs"]:
        assert selected["output"] == f"best/bundle__circuit_{selected['circuit_index']}.qpy"
        case_trials = [row for row in trials if row["case_id"] == selected["case_id"]]
        candidate = min(case_trials, key=rank)
        assert all(selected[key] == candidate[key] for key in ("seed", "weight", "d2", "n2"))
        baseline = min((row for row in case_trials if row["weight"] is None), key=rank)
        wait_time = min((row for row in case_trials if row["weight"] is not None), key=rank)
        for metric in ("d2", "n2"):
            assert selected[f"best_baseline_{metric}"] == baseline[metric]
            assert selected[f"best_wait_time_{metric}"] == wait_time[metric]
            if baseline[metric]:
                assert selected[f"{metric}_ratio"] == pytest.approx(
                    wait_time[metric] / baseline[metric])
            else:
                assert selected[f"{metric}_ratio"] is None
        with (output / selected["output"]).open("rb") as stream:
            compiled, = qpy.load(stream)
        assert measure(compiled) == {key: selected[key] for key in ("d2", "n2")}
        case = next(case for case in plan["inputs"] if case["case_id"] == selected["case_id"])
        with (output / case["snapshot"]).open("rb") as stream:
            original = qpy.load(stream)[case["circuit_index"]]
        assert Operator.from_circuit(compiled).equiv(Operator(original))
    assert "Status: **complete**" in (output / "report.md").read_text()


def test_local_parallel_end_to_end(inputs, tmp_path):
    output = tmp_path / "local"
    assert main(arguments(inputs, output)) == 0
    assert_result(output)
    plan = read_json(output / "plan.json")
    assert [len(shard) for shard in plan["shards"]] == [5, 5, 2]
    assert main(["collect", "--output", str(output)]) == 0
    assert main(arguments(inputs, output)) == 1


def test_output_names_preserve_sources_and_avoid_bundle_collisions(inputs, tmp_path):
    with (inputs / "bundle.qpy").open("rb") as stream:
        circuit = qpy.load(stream)[0]
    names = ["single.qpy", "nested/single.qpy", "bundle__circuit_0.qpy",
             "bundle__circuit_0__1.qpy", "bundle__circuit_1.qpy/inside.qpy"]
    for name in names:
        path = inputs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            qpy.dump(circuit, stream)
    output = tmp_path / "named"
    assert main(arguments(inputs, output, "--seeds", "1", "--weights", "0")) == 0
    summary = read_json(output / "results.json")
    expected = {(name, 0): f"best/{name}" for name in names}
    expected.update({("bundle.qpy", 0): "best/bundle__circuit_0__2.qpy",
                     ("bundle.qpy", 1): "best/bundle__circuit_1__1.qpy"})
    actual = {(row["source"], row["circuit_index"]): row["output"]
              for row in summary["inputs"]}
    assert actual == expected
    report = (output / "report.md").read_text()
    for row in summary["inputs"]:
        with (output / row["output"]).open("rb") as stream:
            compiled, = qpy.load(stream)
        assert measure(compiled) == {key: row[key] for key in ("d2", "n2")}
        assert f"[QPY]({row['output']})" in report
    assert collect(output)
    assert read_json(output / "results.json") == summary


def test_lsf_parallel_end_to_end_matches_local(inputs, tmp_path):
    fake_bsub = tmp_path / "fake bsub"
    fake_bsub.write_text(f"#!{sys.executable}\n" + '''
import subprocess, sys
assert '-K' in sys.argv
assert sys.argv[sys.argv.index('-n') + 1] == '1'
print('Job <123> is submitted', flush=True)
script = sys.stdin.read()
with open(sys.argv[sys.argv.index('-oo') + 1], 'w') as out, open(sys.argv[sys.argv.index('-eo') + 1], 'w') as err:
    result = subprocess.run(['/bin/sh'], input=script, text=True, stdout=out, stderr=err)
sys.exit(result.returncode)
''')
    fake_bsub.chmod(0o755)
    output = tmp_path / "LSF output with spaces"
    assert main(arguments(inputs, output, "--executor", "lsf", "--bsub", str(fake_bsub))) == 0
    assert_result(output)
    local = tmp_path / "local"
    assert main(arguments(inputs, local)) == 0
    expected = read_json(local / "results.json")["inputs"]
    actual = read_json(output / "results.json")["inputs"]
    assert actual == expected
    for index in range(3):
        assert read_json(shard_directory(output, index) / "submission.json")["returncode"] == 0


@pytest.mark.parametrize(("batch_size", "expected_sizes"), [(5, [5, 5, 2]), (7, [7, 5])])
def test_lsf_submits_all_batches_independent_of_workers(
        inputs, tmp_path, monkeypatch, batch_size, expected_sizes):
    output = tmp_path / "lsf"
    submitted = []

    class Submission:
        def __init__(self, command, **kwargs):
            name = command[command.index("-J") + 1]
            self.index = int(name.rsplit("-", 1)[1])
            self.returncode = None
            submitted.append(self.index)

        def poll(self):
            assert len(submitted) == len(expected_sizes), "LSF submissions were capped by workers"
            if self.returncode is None:
                success = execute_shard(output / "plan.json", self.index)
                self.returncode = 0 if success else 1
            return self.returncode

        def terminate(self):
            self.returncode = -15

    monkeypatch.setattr(lsf, "subprocess",
                        SimpleNamespace(Popen=Submission, STDOUT=subprocess.STDOUT))
    assert main(arguments(inputs, output, "--executor", "lsf", "--bsub", sys.executable,
                          "--workers", "1", "--transpilations-per-job", str(batch_size))) == 0
    plan = read_json(output / "plan.json")
    assert [len(shard) for shard in plan["shards"]] == expected_sizes
    assert submitted == list(range(len(expected_sizes)))
    assert_result(output)


def test_failed_lsf_submission_is_incomplete(inputs, tmp_path):
    fake_bsub = tmp_path / "bsub"
    fake_bsub.write_text("#!/bin/sh\necho 'submission failed'\nexit 3\n")
    fake_bsub.chmod(0o755)
    output = tmp_path / "failed"
    assert main(arguments(inputs, output, "--executor", "lsf", "--bsub", str(fake_bsub))) == 1
    assert not read_json(output / "results.json")["complete"]
    assert "INCOMPLETE" in (output / "report.md").read_text()
    assert not (output / "best").exists()


def test_missing_or_failed_trials_never_publish_a_minimum(inputs, tmp_path):
    output = tmp_path / "incomplete"
    plan_path = prepare(parser().parse_args(arguments(inputs, output)))
    assert not collect(output)
    assert not (output / "best").exists()
    assert execute_shard(plan_path, 0)
    assert not collect(output)
    snapshot = output / read_json(plan_path)["inputs"][0]["snapshot"]
    snapshot.write_bytes(b"changed")
    assert not execute_shard(plan_path, 1)
    result = read_json(shard_directory(output, 1) / "result.json")
    assert all("checksum" in row["error"] for row in result["rows"])
    assert not collect(output)
    for selected in read_json(output / "results.json")["inputs"]:
        assert selected["status"] == "incomplete"
        for metric in ("d2", "n2"):
            assert not {f"best_baseline_{metric}", f"best_wait_time_{metric}",
                        f"{metric}_ratio"}.intersection(selected)
    table = [line.split("|")[1:-1] for line in (output / "report.md").read_text().splitlines()
             if line.startswith("|")]
    for cells in table[2:]:
        row = dict(zip(map(str.strip, table[0]), map(str.strip, cells)))
        for metric in ("D2", "N2"):
            for column in (f"Best baseline {metric}", f"Best wait-time {metric}",
                           f"{metric} ratio (wait/baseline)"):
                assert row[column] == "—"


@pytest.mark.parametrize(("baseline_d2", "wait_time_d2", "d2_ratio",
                          "baseline_n2", "wait_time_n2", "n2_ratio"), [
    (10, 6, 0.6, 40, 30, 0.75), (6, 9, 1.5, 30, 60, 2.0), (7, 7, 1.0, 25, 25, 1.0),
    (0, 0, None, 0, 0, None), (0, 3, None, 0, 8, None),
])
def test_report_compares_group_minima_from_all_trials(
        tmp_path, baseline_d2, wait_time_d2, d2_ratio, baseline_n2, wait_time_n2, n2_ratio):
    case = {"case_id": "circuit", "source": "input.qpy", "circuit_index": 0}
    tasks = [{"task_id": index, "case_id": case["case_id"], "seed": seed, "weight": weight}
             for index, (seed, weight) in enumerate(
                 (seed, weight) for seed in range(2) for weight in (None, 0.0, 0.1))]
    # The group minima use different seeds after resolving D2 ties with N2.
    # Weight zero is a wait-time trial. Worse-depth trials can have lower N2.
    depths = [baseline_d2, wait_time_d2, wait_time_d2 + 2,
              baseline_d2, wait_time_d2, wait_time_d2 + 1]
    counts = [baseline_n2 + 5, wait_time_n2, wait_time_d2 + 2,
              baseline_n2, wait_time_n2 + 4, wait_time_d2 + 1]
    rows = [{**task, "status": "ok", "d2": depth, "n2": count}
            for task, depth, count in zip(tasks, depths, counts)]
    write_json(tmp_path / "plan.json", {
        "run_id": "comparison", "inputs": [case], "shards": [tasks],
        "config": {"target": "test", "optimization_level": 3, "seed_start": 0,
                   "seeds": 2, "weights": [0.0, 0.1]},
    })
    directory = shard_directory(tmp_path, 0)
    directory.mkdir(parents=True)
    artifact = directory / "winner.qpy"
    artifact.write_bytes(b"saved candidate")
    # Only the overall winner has an artifact, as with real shard results.
    winner = min(rows, key=rank)
    write_json(directory / "result.json", {
        "run_id": "comparison", "shard_index": 0, "rows": rows,
        "winners": [{**winner, "artifact": artifact.relative_to(tmp_path).as_posix(),
                     "artifact_sha256": sha256(artifact)}],
    })

    assert collect(tmp_path)
    selected, = read_json(tmp_path / "results.json")["inputs"]
    assert selected["d2"] == winner["d2"]
    table = [line.split("|")[1:-1] for line in (tmp_path / "report.md").read_text().splitlines()
             if line.startswith("|")]
    report_row = dict(zip(map(str.strip, table[0]), map(str.strip, table[2])))
    for metric, baseline, wait_time, ratio in (
            ("d2", baseline_d2, wait_time_d2, d2_ratio),
            ("n2", baseline_n2, wait_time_n2, n2_ratio)):
        assert selected[f"best_baseline_{metric}"] == baseline
        assert selected[f"best_wait_time_{metric}"] == wait_time
        assert selected[f"{metric}_ratio"] == (pytest.approx(ratio) if ratio is not None else None)
        assert report_row[f"Best baseline {metric.upper()}"] == str(baseline)
        assert report_row[f"Best wait-time {metric.upper()}"] == str(wait_time)
        assert report_row[f"{metric.upper()} ratio (wait/baseline)"] == (
            format(ratio, ".6g") if ratio is not None else "N/A")


def test_duplicate_worker_and_altered_result_rejected(inputs, tmp_path):
    output = tmp_path / "integrity"
    plan_path = prepare(parser().parse_args(arguments(inputs, output)))
    execute_shard(plan_path, 0)
    with pytest.raises(FileExistsError):
        execute_shard(plan_path, 0)
    result_path = shard_directory(output, 0) / "result.json"
    result = read_json(result_path)
    result["rows"][0]["seed"] += 1
    write_json(result_path, result)
    with pytest.raises(ValueError, match="tasks mismatch"):
        collect(output)


def test_defaults_and_validation(inputs, tmp_path):
    defaults = parser().parse_args(["run"])
    assert defaults.seeds == 30
    assert defaults.workers == defaults.transpilations_per_job == 5
    assert defaults.weights == list(DEFAULT_WEIGHTS)
    for extra in (["--seeds", "0"], ["--weights", "nan"], ["--weights", "inf"],
                  ["--weights", "-0.1"], ["--workers", "0"], ["--seed-start", "-1"],
                  ["--seed-start", str(2**32 - 1)]):
        with pytest.raises(SystemExit):
            main(arguments(inputs, tmp_path / "unused", *extra))
    assert main(arguments(inputs, inputs / "output")) == 1
    assert main(arguments(tmp_path / "missing", tmp_path / "output")) == 1


def test_empty_and_malformed_inputs_rejected(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    assert main(arguments(source, tmp_path / "empty")) == 1
    (source / "broken.qpy").write_bytes(b"not a qpy")
    assert main(arguments(source, tmp_path / "malformed")) == 1


def test_lsf_shell_quoting_and_resources(inputs, tmp_path):
    output = tmp_path / "out with ' quotes $dollar"
    plan_path = prepare(parser().parse_args(arguments(inputs, output)))
    config = read_json(plan_path)["config"]
    command = submission_command(plan_path, 0, config)
    assert command[command.index("-M") + 1] == "10"
    config["lsf_memory_unit"] = "MB"
    command = submission_command(plan_path, 0, config)
    assert command[command.index("-M") + 1] == "10240"
    echo_python = tmp_path / "echo python"
    echo_python.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    echo_python.chmod(0o755)
    result = subprocess.run(["/bin/sh"], input=job_script(plan_path, 0, str(echo_python)),
                            text=True, capture_output=True, check=True)
    args = json.loads(result.stdout)
    assert Path(args[args.index("--plan") + 1]) == plan_path

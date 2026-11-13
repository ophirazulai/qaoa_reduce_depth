"""Submit all planned batches using LSF's blocking bsub -K protocol."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from .storage import read_json, write_json
from .sweep import shard_directory


def job_script(plan_path, index, python=None):
    command = [python or sys.executable, "-m", "qaoa_reduce_depth", "worker",
               "--plan", str(Path(plan_path).resolve()), "--shard", str(index)]
    return "#!/bin/sh\nset -eu\nexec " + shlex.join(command) + "\n"


def submission_command(plan_path, index, config, run_id=None):
    directory = shard_directory(Path(plan_path).parent, index)
    units = {"KB": 1024 * 1024, "MB": 1024, "GB": 1}
    memory = config["lsf_memory_gb"] * units[config["lsf_memory_unit"]]
    # Callers submitting many jobs pass run_id: the plan grows with the number
    # of jobs, so re-reading it for each one is quadratic.
    name = f"qaoa-{(run_id or read_json(plan_path)['run_id'])[:10]}-{index}"
    command = [
        config["bsub"], "-K", "-J", name, "-n", "1", "-W", str(config["lsf_wall_minutes"]),
        "-M", str(memory), "-R", f"rusage[mem={memory}] span[hosts=1]",
        "-cwd", str(Path(plan_path).parent),
        "-oo", str(directory / "stdout.log"), "-eo", str(directory / "stderr.log"),
    ]
    if config["lsf_queue"]:
        command.extend(["-q", config["lsf_queue"]])
    if config["lsf_project"]:
        command.extend(["-P", config["lsf_project"]])
    # Script is provided on stdin, avoiding bsub's command argument re-quoting.
    return command


def run_lsf(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    config = plan["config"]
    if shutil.which(config["bsub"]) is None:
        raise ValueError(f"LSF submission executable not found: {config['bsub']}")
    active, errors = {}, []
    next_index = completed = 0
    total = len(plan["shards"])
    try:
        while next_index < total or active:
            while next_index < total:
                index = next_index
                next_index += 1
                directory = shard_directory(plan_path.parent, index)
                directory.mkdir(parents=True, exist_ok=True)
                script = directory / "job.sh"
                script.write_text(job_script(plan_path, index))
                command = submission_command(plan_path, index, config, plan["run_id"])
                metadata = {"command": command, "shard_index": index, "status": "submitting"}
                write_json(directory / "submission.json", metadata)
                try:
                    with script.open("rb") as stdin, (directory / "bsub.log").open("wb") as log:
                        process = subprocess.Popen(command, stdin=stdin, stdout=log,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
                    active[index] = (process, metadata)
                except OSError as exc:
                    metadata.update(status="submission_error", error=str(exc))
                    write_json(directory / "submission.json", metadata)
                    errors.append(f"shard {index}: {exc}")
                    completed += 1
            for index, (process, metadata) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                metadata.update(status="done" if code == 0 else "failed", returncode=code)
                write_json(shard_directory(plan_path.parent, index) / "submission.json", metadata)
                if code:
                    errors.append(f"shard {index}: bsub -K exited {code}; see its bsub.log")
                del active[index]
                completed += 1
                print(f"Completed {completed}/{total} LSF jobs", flush=True)
            if active:
                time.sleep(0.2)
    finally:
        # Stopping a bsub client does not reliably cancel an accepted LSF job.
        # Preserve all logs and identities and never silently retry a submission.
        for process, _ in active.values():
            if process.poll() is None:
                process.terminate()
        write_json(plan_path.parent / "execution.json", {
            "executor": "lsf", "errors": errors,
            "interrupted_shards": list(active), "coordinator_pid": os.getpid(),
        })

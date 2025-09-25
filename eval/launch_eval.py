import argparse
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from shlex import quote
from copy import deepcopy
from slurmpilot import JobCreationInfo, SlurmPilot, unify


def load_tasks_from_path(path):
    with open(path, "r") as f:
        lines = f.readlines()
    n_fewshot_to_tasks = defaultdict(list)
    for line in lines:
        if not line.startswith("#"):
            task, n_fewshot = line.split(";")
            n_fewshot_to_tasks[int(n_fewshot.strip())].append(task.strip())
    return n_fewshot_to_tasks

def _clean_header(lines: list[str]) -> list[str]:
    """Drop --error and force --output=logs/%j_eval.out."""
    out = []
    for l in lines:
        l = l.strip()
        if l.startswith("#SBATCH --error"):
            continue
        if l.startswith("#SBATCH --output"):
            out.append("#SBATCH --output=logs/%j_eval.out")
        else:
            out.append(l)
    return out

def sbatch_commands(job: JobCreationInfo) -> list[str]:
    """
    Convert a JobCreationInfo into a list of ready‑to‑paste `sbatch …` commands.

    If `python_args` is a list we create one command per element; otherwise we
    build a single command.  Nothing is written to disk – the script is wrapped
    with `--wrap` so you can execute directly from your terminal.
    """
    header_lines = job.sbatch_preamble(is_job_array=False).strip().splitlines()  
    header_lines = _clean_header(header_lines)
    
    header_flags = " ".join(line.replace("#SBATCH", "").strip() for line in header_lines)

    def _wrap(arg_string: str = "") -> str:
        if job.python_binary and job.python_binary.lower() != "bash":
            entry_call = f"{quote(job.python_binary)} {quote(job.entrypoint)} {arg_string}"
        else: 
            entry_call = f"bash {quote(job.entrypoint)} {arg_string}"
        if job.bash_setup_command:
            entry_call = f"{job.bash_setup_command.strip()}\n{entry_call}"
        return f"--wrap={quote(entry_call)}"

    commands = []
    if isinstance(job.python_args, list):
        for idx, arg in enumerate(job.python_args):
            j = deepcopy(job)
            j.jobname = f"{job.jobname}_{idx}"
            header_lines_unique = j.sbatch_preamble(is_job_array=False).strip().splitlines()
            header_lines_unique = _clean_header(header_lines_unique)
            header_flags_unique = " ".join(l.replace("#SBATCH", "").strip() for l in header_lines_unique)
            commands.append(f"sbatch {header_flags_unique} {_wrap(arg)}")
    else:
        arg_string = "" if job.python_args is None else job.python_args
        commands.append(f"sbatch {header_flags} {_wrap(arg_string)}")

    return commands

def main():
    # TODO make tasks configurable
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        help="name of file containing models to evaluate (one per line).",
        required=False,
    )
    parser.add_argument(
        "--tasks_file",
        type=str,
        help="name of file containing tasks to evaluate (one per line).",
        required=False,
        default=str(Path(__file__).parent / "tasks.txt"),
    )
    parser.add_argument(
        "--model",
        type=str,
        help="name of a model to evaluate, incompatible with model_file.",
        required=False,
    )
    parser.add_argument(
        "--cluster",
        type=str,
        help="name of a cluster to launch experiments on, for instance leonardo",
        required=True,
    )
    parser.add_argument(
        "--account",
        type=str,
        help="name of an account to use",
        required=False,
    )
    parser.add_argument(
        "--partition",
        type=str,
        help="name of partition to use",
        required=True,
    )
    parser.add_argument(
        "--hf_home",
        type=str,
        help="location of HF_HOME which you use when calling setup_node.sh",
        required=True,
    )
    parser.add_argument(
        "--venv_path",
        type=str,
        help="location of VENV_PATH which you use when calling setup_node.sh",
        required=True,
    )
    parser.add_argument(
        "--eval_output_path",
        type=str,
        help="location where evaluation json files will be written",
        required=True,
    )
    parser.add_argument(
        "--symlink_path",
        type=str,
        help="location where to store symlinks for models to be evaluated",
        required=False,
    )
    parser.add_argument(
        "--max_jobs",
        type=int,
        help="maximum number of jobs to launch",
        required=False,
    )
    parser.add_argument(
        "--start",
        type=int,
        help="index to start",
        required=False,
    )

    args = parser.parse_args()

    assert bool(args.model_file is not None) ^ bool(args.model is not None), (
        "Exactly one of model or model_file argument should be used."
    )

    models_file = args.model_file
    cluster = args.cluster
    partition = args.partition
    account = args.account
    hf_home = args.hf_home
    venv_path = args.venv_path
    eval_output_path = args.eval_output_path
    symlink_path = args.symlink_path
    jobname = "openeurollm/eval"

    n_fewshot_to_tasks = load_tasks_from_path(Path(__file__).parent / args.tasks_file)
    print(f"Going to eval {dict(n_fewshot_to_tasks)}")

    if models_file:
        # read the models from the provided file
        with open(models_file, "r") as f:
            model_paths = f.readlines()
        # remove "\n" at the end of each string
        model_paths = [x.strip() for x in model_paths]
    else:
        # use the provided model
        model_paths = [args.model]

    # TODO allow to configure dispatch stategy
    # loop over all models first, then all tasks
    python_args = [
        f"{task} {n_fewshot} {model_path}"
        for model_path in model_paths
        for n_fewshot, tasks in n_fewshot_to_tasks.items()
        for task in tasks
    ]

    # we set things here that depends on $USER which is known at runtime as opposed to other env vars
    bash_setup_command = f"""
# ml Python  # cluster specific
# ml Cuda  # cluster specific
source {venv_path}/bin/activate
export HF_HOME={hf_home}
export LM_EVAL_OUTPUT_PATH={eval_output_path}
# export CUDA_VISIBLE_DEVICES=0,1,2,3  # number of GPU specific
    """
    if symlink_path:
      bash_setup_command += f"\nexport SYMLINK_PATH={symlink_path}"
    if args.start:
        python_args = python_args[args.start:]
    if args.max_jobs is not None:
        print(f"{len(python_args)} jobs before filtering.")
        python_args = python_args[:args.max_jobs]

    print(f"{len(python_args)} jobs.")
    job = JobCreationInfo(
        cluster=cluster,
        partition=partition,
        jobname=unify(jobname),
        account=account,
        entrypoint="main_script_loop.sh",
        src_dir=str(Path(__file__).parent),
        python_binary="bash",
        python_args=python_args,
        bash_setup_command=bash_setup_command,
        n_gpus=1,
        n_concurrent_jobs=min(len(python_args), 32),
        max_runtime_minutes=24 * 60 - 1,
        env={
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
            "WANDB_MODE": "offline",
            "HF_HUB_OFFLINE": "1",
            "BATCH_SIZE": 36,
        },
    )
    
    cmds = sbatch_commands(job)
    for cmd in cmds:
        print(f"Running: {cmd}")
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print(f"[✓] Success:\n{stdout}")
        else:
            print(f"[!] Error (exit {process.returncode}):\n{stderr}")
    # api = SlurmPilot(clusters=[cluster])
    # api.schedule_job(job_info=job)


if __name__ == '__main__':
    main()
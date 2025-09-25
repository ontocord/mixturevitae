import argparse
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_iterations_from_checkpoint(
    checkpoint_path: Path,
) -> Tuple[List[str], List[Path]]:
    """
    Determine iterations by scanning the checkpoint directory structure.
    Looks for directories in the format 'iter_0002000', etc.
    """
    iterations = []
    ckpt_iter_paths = []
    # Check if checkpoint_path exists and is a directory
    if not checkpoint_path.exists() or not checkpoint_path.is_dir():
        logging.warning(
            f"Warning: Checkpoint path {checkpoint_path} does not exist or is not a directory"
        )
        raise ValueError(
            f"Checkpoint path {checkpoint_path} does not exist or is not a directory"
        )

    # Look for iteration directories (format: iter_XXXXXXX)
    for item in checkpoint_path.iterdir():
        if item.is_dir() and item.name.startswith("iter_"):
            iter_num = item.name.split("_")[1]
            if iter_num.isdigit():
                iterations.append(iter_num)
                ckpt_iter_paths.append(item)

    for path in ckpt_iter_paths:
        if os.path.islink(path):
            iter_num = path.name.split("_")[1]
            for item in Path(os.path.realpath(path)).parent.iterdir():
                if (
                    item.is_dir()
                    and item.name.startswith("iter_")
                    and item.name.split("_")[1].isdigit()
                    and int(item.name.split("_")[1]) < int(iter_num)
                ):
                    logging.debug(
                        f"Adding {item} to ckpt_iter_paths for {int(iter_num)} > {int(item.name.split('_')[1])}"
                    )
                    ckpt_iter_paths.append(item)
                    iterations.append(item.name.split("_")[1])

    # Sort iterations numerically and sort ckpt_iter_paths accordingly
    sorted_pairs = sorted(zip(iterations, ckpt_iter_paths), key=lambda x: int(x[0]))
    iterations, ckpt_iter_paths = zip(*sorted_pairs) if sorted_pairs else ([], [])
    iterations = list(iterations)
    ckpt_iter_paths = list(ckpt_iter_paths)

    if not iterations and not ckpt_iter_paths:
        logging.info(f"No iterations found in {checkpoint_path}")
    return iterations, ckpt_iter_paths


def extract_checkpoints_from_logs(
    slurm_log_dir: Union[str, Path],
) -> List[Tuple[Path, Path]]:
    """
    Extract checkpoint paths from SLURM log files.
    Looks for files that have completed training and extracts the checkpoint path.

    Args:
        slurm_log_dir: Directory containing SLURM log files

    Returns:
        List of tuples containing (checkpoint_path, log_file_path)
    """
    slurm_log_dir = Path(slurm_log_dir)
    checkpoint_paths_and_logs = []

    for slurm_log_file in slurm_log_dir.glob("*.out"):
        with open(slurm_log_file, "r") as f:
            content = f.read()
            if "[after training is done]" in content:
                # Extract checkpoint path from the line after [after training is done]
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if "[after training is done]" in line and i + 1 < len(lines):
                        # Look for checkpoint path in the next line
                        next_lines = lines[i + 1 :]
                        for next_line in next_lines:
                            if "/leonardo_work/AIFAC_L01_028/" in next_line:
                                checkpoint_path = next_line.strip()
                                checkpoint_path = (
                                    checkpoint_path.split(" to ")[-1]
                                    .strip()
                                    .split(" ")[0]
                                )
                                if os.path.exists(checkpoint_path):
                                    checkpoint_paths_and_logs.append(
                                        (Path(checkpoint_path), slurm_log_file)
                                    )
                                    break
                                else:
                                    logging.info(
                                        f"Checkpoint path {checkpoint_path} does not exist"
                                    )
                                    break

    return checkpoint_paths_and_logs


def extract_model_size(log_path: Path) -> str:
    with open(log_path, "r") as f:
        log_file = f.read()

    if "Total number of parameters in billions" not in log_file:
        raise ValueError("billions not found")

    with open(log_path, "r") as f:
        log_file_lines = f.readlines()

    for line in log_file_lines:
        if "Total number of parameters in billions" in line:
            num = line.strip().split("billions: ")[-1]
            if "1.3" in num:
                return "1.3"
            elif "0.4" in num:
                return "0.4"
            elif "1.7" in num:
                return "1.7"
            elif "0.13" in num:
                return "0.13"


def get_model_config_from_command_line(log_path: Path) -> Optional[Dict[str, int]]:
    """
    Extract model configuration by parsing the arguments block in the log file.

    Args:
        log_path: Path to the log file

    Returns:
        dict: Model configuration parameters or None if not found
    """
    defaults = {
        "0.13": {"FFN_HIDDEN_SIZE": 2256},
        "0.4": {"FFN_HIDDEN_SIZE": 3840},
        "1.3": {"FFN_HIDDEN_SIZE": 5440},
        "1.7": {"FFN_HIDDEN_SIZE": 8192},
    }

    try:
        with open(log_path, "r") as f:
            lines = f.readlines()

        in_args_block = False
        args_lines = []
        for line in lines:
            if " arguments " in line and "------------------------" in line:
                in_args_block = True
                continue
            if " end of arguments " in line and "---------------------" in line:
                break
            if in_args_block:
                args_lines.append(line)

        if not args_lines:
            logging.info(f"No arguments block found in {log_path}")
            return None

        config = {}

        param_map = {
            "num_layers": "NUM_LAYERS",
            "hidden_size": "HIDDEN_SIZE",
            "ffn_hidden_size": "FFN_HIDDEN_SIZE",
            "num_attention_heads": "NUM_ATTN_HEADS",
            "seq_length": "SEQ_LENGTH",
            "max_position_embeddings": "MAX_POSITION_EMBEDDINGS",
        }

        for line in args_lines:
            # e.g. [lrdn1299:0]:  hidden_size ..................................... 512
            line_content = line.split("]:", 1)[-1].strip()
            for param, config_key in param_map.items():
                if line_content.startswith(param):
                    match = re.match(rf"{param}\s*\.+\s*(\S+)", line_content)
                    if match:
                        val_str = match.group(1).strip()
                        try:
                            config[config_key] = int(val_str)
                        except ValueError:
                            logging.warning(
                                f"Could not parse integer value for {param}: {val_str}"
                            )
                        break

        model_size = extract_model_size(log_path=log_path)
        logging.debug(f"model size is {model_size}")

        try:
            for v in param_map.values():
                if v not in config:
                    if (
                        model_size
                        and model_size in defaults
                        and v in defaults.get(model_size, {})
                    ):
                        config[v] = defaults[model_size][v]
                        logging.debug(f"setting {v} as :{defaults[model_size][v]}")
        except Exception as e:
            logging.error(e)

        if not config:
            logging.debug(
                f"No model configuration parameters found in arguments block: {log_path}"
            )
            return None

        return config

    except Exception as e:
        logging.error(f"Error parsing arguments block from log file {log_path}: {e}")
        return None


def get_converted_iterations(save_checkpoints_dir: str, model_name: str) -> List[str]:
    """
    Get the list of iterations that have already been converted for a given model.
    Checks for the existence of model.safetensors in the hf directory to ensure
    conversion was successful.

    Args:
        save_checkpoints_dir: Directory where converted checkpoints are saved
        model_name: Name of the model (log base name)

    Returns:
        List of iteration strings that have already been successfully converted
    """
    converted_iterations = []
    model_dir = Path(save_checkpoints_dir) / model_name

    if not model_dir.exists():
        return converted_iterations

    # Check hf directory for successfully converted iterations
    hf_dir = model_dir / "hf"
    if hf_dir.exists():
        for item in hf_dir.iterdir():
            if item.is_dir() and item.name.startswith("iter_"):
                iter_num = item.name.split("_")[1]
                if iter_num.isdigit():
                    # Check if model.safetensors exists to verify successful conversion
                    safetensors_file = item / "model.safetensors"
                    if safetensors_file.exists():
                        converted_iterations.append(iter_num)
                        logging.debug(
                            f"Found successfully converted iteration {iter_num} for {model_name}"
                        )
                    else:
                        logging.info(
                            f"Iteration {iter_num} for {model_name} exists but model.safetensors missing - conversion likely failed"
                        )
    return sorted(converted_iterations, key=int)


def convert_checkpoint_consolidated(
    log_path: Path,
    iterations: List[str],
    ckpt_iter_paths: List[Path],
    save_checkpoints_dir: str,
    # checkpoint_path: Path,
    opensci_megatron_path: str,
    open_sci_hf_path: str,
    convert_logs_dir: str,
    account: str,
    partition: str,
    container_image: str,
    model_config: Optional[Dict[str, int]] = None,
) -> None:
    """
    Convert a checkpoint using the consolidated approach.
    This replaces the bash script + converter.py approach with direct Python calls.

    Args:
        log_path: Path to the log file
        iterations: List of iterations to convert
        save_checkpoints_dir: Directory to save converted checkpoints
        checkpoint_path: Path to the checkpoint directory
        opensci_megatron_path: Path to Megatron-LM-Open-Sci repository
        open_sci_hf_path: Path to Open-Sci-hf repository
        convert_logs_dir: Directory to save conversion logs
        account: Account to use for conversion
        partition: Partition to use for conversion
        container_image: Container image to use for conversion
        model_config: Model configuration parameters (optional)
    """
    model_config = get_model_config_from_command_line(log_path)

    if not model_config:
        logging.debug(
            f"Skipping {log_path.name}, could not determine model configuration"
        )
        return

    # Get the model name from the log file
    log_base_name = log_path.name.split(".out")[0]

    # Get already converted iterations for this model
    converted_iterations = get_converted_iterations(save_checkpoints_dir, log_base_name)

    # Filter out iterations that have already been converted
    remaining_iterations = []
    remaining_ckpt_iter_paths = []

    for iteration, path in zip(iterations, ckpt_iter_paths):
        if iteration not in converted_iterations:
            remaining_iterations.append(iteration)
            remaining_ckpt_iter_paths.append(path)
        else:
            logging.info(
                f"Iteration {iteration} for {log_base_name} already converted, skipping"
            )
    # If no iterations need to be converted, return early
    if not remaining_iterations:
        logging.info(f"All iterations for {log_base_name} already converted, skipping")
        return

    # Create necessary directories
    os.makedirs(convert_logs_dir, exist_ok=True)
    os.makedirs(save_checkpoints_dir, exist_ok=True)

    # Load the SBATCH template
    sbatch_template_path = os.path.join(
        opensci_megatron_path, "scripts/ckpt/convert_full/template.sbatch"
    )

    try:
        with open(sbatch_template_path, "r") as f:
            sbatch_template = f.read()
            # escape ${} in f-strings with double curly braces
            # escape cat <<EOF > ../config.json\n{...}\nEOF
            cat_eof_data = re.search(
                r"cat <<EOF.*?EOF", sbatch_template, re.DOTALL
            ).group()
            sbatch_template = sbatch_template.replace(cat_eof_data, "<cat_eof_data>")
            sbatch_template = re.sub(
                r"\$\{(.+?)\}", r"\${{\1}}", sbatch_template
            ).replace(r"\$", "$")

        # Process each remaining iteration
        for iteration, path in zip(remaining_iterations, remaining_ckpt_iter_paths):
            logging.info(f"Converting iteration {iteration} for {log_base_name}")
            sbatch_script = sbatch_template.format(
                account=account,
                partition=partition,
                container_image=container_image,
                opensci_megatron_path=opensci_megatron_path,
                open_sci_hf_path=open_sci_hf_path,
                train_logs_path=str(log_path),
                dist_ckpt_path=str(path.parent),
                save_checkpoints_dir=save_checkpoints_dir,
                convert_logs_dir=convert_logs_dir,
                pre_run_cmd="",  # No pre-run command by default
                iteration_to_convert=iteration,
                num_layers=model_config["NUM_LAYERS"],
                num_attn_heads=model_config["NUM_ATTN_HEADS"],
                ffn_hidden_size=model_config["FFN_HIDDEN_SIZE"],
                max_seq_length=model_config["MAX_POSITION_EMBEDDINGS"],
            )

            sbatch_script = sbatch_script.replace("<cat_eof_data>", cat_eof_data)

            # Create a temporary file for the sbatch script
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".sbatch", delete=False
            ) as f:
                f.write(sbatch_script)
                sbatch_script_path = f.name

            subprocess.run(["sbatch", sbatch_script_path])
            logging.info(f"Submitted {sbatch_script_path} for iteration {iteration}")

            # Clean up the temporary file
            os.unlink(sbatch_script_path)

    except Exception as e:
        logging.error(f"Error converting checkpoint: {e}")
        logging.info(f"Skipping checkpoint: {log_path}")


def process_all_checkpoints_consolidated(
    slurm_log_dir: Union[str, Path],
    save_checkpoints_dir: str,
    opensci_megatron_path: str,
    open_sci_hf_path: str,
    convert_logs_dir: str,
    account: str,
    partition: str,
    container_image: str,
    checkpoint_paths_and_logs: Optional[List[Tuple[str,str]]] = None,
) -> None:
    """
    Process all checkpoints found in the logs using the consolidated approach.

    Args:
        slurm_log_dir: Directory containing SLURM log files
        save_checkpoints_dir: Directory to save converted checkpoints
        opensci_megatron_path: Path to Megatron-LM-Open-Sci repository
        open_sci_hf_path: Path to Open-Sci-hf repository
        convert_logs_dir: Directory to save conversion logs
        account: Account to use for conversion
        partition: Partition to use for conversion
        container_image: Container image to use for conversion
        checkpoint_paths_and_logs: [Optional] List of tuples of checkpoint paths and log files.
    """
    # Extract checkpoints
    if not checkpoint_paths_and_logs:  
        checkpoint_paths_and_logs = extract_checkpoints_from_logs(slurm_log_dir)
    
    # Process each checkpoint
    for checkpoint_path, log_path in checkpoint_paths_and_logs:
        # Determine iterations from the checkpoint directory
        checkpoint_iterations, ckpt_iter_paths = get_iterations_from_checkpoint(
            checkpoint_path
        )

        # If no iterations found, skip this checkpoint
        if not len(checkpoint_iterations) > 0:
            logging.info(f"No iterations found in {checkpoint_path}, skipping")
            continue

        convert_checkpoint_consolidated(
            log_path=log_path,
            iterations=checkpoint_iterations,
            ckpt_iter_paths=ckpt_iter_paths,
            save_checkpoints_dir=save_checkpoints_dir,
            opensci_megatron_path=opensci_megatron_path,
            open_sci_hf_path=open_sci_hf_path,
            convert_logs_dir=convert_logs_dir,
            account=account,
            partition=partition,
            container_image=container_image,
        )


def parse_ckpt_log_pair(s: str):
    parts = s.split(',', 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f'Invalid format: "{s}". Must be "checkpoint_dir,log_path"'
        )
    parts[0], parts[1] = Path(parts[0]), Path(parts[1])
    return tuple(parts)

def main():
    """Main function that handles command line arguments and runs the consolidated workflow."""
    parser = argparse.ArgumentParser(
        description="Consolidated checkpoint conversion workflow"
    )

    # Arguments from original checkpoint_conversion_workflow.py
    parser.add_argument(
        "--slurm_log_dir",
        type=str,
        default="/leonardo_work/AIFAC_L01_028/najroldi/pretrain/slurm_output/completed",
        help="Directory containing SLURM log files",
    )
    parser.add_argument(
        "--checkpoint_paths_and_logs",
        type=parse_ckpt_log_pair,
        nargs='+',
        metavar="CKPT,LOG",
        default=None,
        help=(
            "List of checkpoint_dir,log_path pairs, e.g.: "
            "--checkpoint_paths_and_logs ckpt1,log1 ckpt2,log2"
        ),
    )
    parser.add_argument(
        "--save_checkpoints_dir",
        type=str,
        default="/leonardo/home/userexternal/tcarsten/work/converted_checkpoints",
        help="Directory to save converted checkpoints",
    )
    parser.add_argument(
        "--opensci_megatron_path",
        type=str,
        default="/leonardo/home/userexternal/tcarsten/work/timur_megatron_open_sci",
        help="Path to Megatron-LM-Open-Sci repository",
    )
    parser.add_argument(
        "--open_sci_hf_path",
        type=str,
        default="/leonardo/home/userexternal/tcarsten/work/Open-Sci-hf",
        help="Path to Open-Sci-hf repository",
    )
    parser.add_argument(
        "--convert_logs_dir",
        type=str,
        default="/leonardo/home/userexternal/tcarsten/work/slurm_logs",
        help="Directory to save conversion logs",
    )
    parser.add_argument(
        "--account",
        type=str,
        default="CMPNS_E03_068",
        help="Account to use for conversion",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="boost_usr_prod",
        help="Partition to use for conversion",
    )
    parser.add_argument(
        "--container_image",
        type=str,
        default="/leonardo_work/AIFAC_L01_028/shared/container_images/pytorch_24.09-py3_leonardo.sif",
        help="Container image to use for conversion",
    )

    args = parser.parse_args()

    # Run the consolidated workflow
    process_all_checkpoints_consolidated(
        slurm_log_dir=args.slurm_log_dir,
        save_checkpoints_dir=args.save_checkpoints_dir,
        opensci_megatron_path=args.opensci_megatron_path,
        open_sci_hf_path=args.open_sci_hf_path,
        convert_logs_dir=args.convert_logs_dir,
        account=args.account,
        partition=args.partition,
        container_image=args.container_image,
        checkpoint_paths_and_logs=args.checkpoint_paths_and_logs, 
    )


if __name__ == "__main__":
    main()

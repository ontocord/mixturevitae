import argparse
import math
import json
import os
import time
import gzip
import glob
import multiprocessing

try:
    import nltk
    from nltk.tokenize.punkt import PunktLanguageVars

    nltk_available = True
except ImportError:
    PunktLanguageVars = object  # Fallback to the built-in object class
    nltk_available = False

from megatron.training.tokenizer import build_tokenizer
from megatron.training.arguments import _add_tokenizer_args
from megatron.core.datasets import indexed_dataset
from tools.preprocess_data import get_file_name, check_files_exist, Partition
from megatron.core.datasets.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)

import ray
import shutil
import logging

parser = argparse.ArgumentParser()

parser = _add_tokenizer_args(parser)
parser.add_argument(
    "--cpus-per-ray-worker", type=int, default=1, help="Number of CPUs per worker"
)
group = parser.add_argument_group(title="input data")
group.add_argument("--input", type=str, required=True, help="Path to input JSON")
group.add_argument(
    "--json-keys",
    nargs="+",
    default=["text"],
    help="space separate listed of keys to extract from json",
)
group.add_argument(
    "--split-sentences", action="store_true", help="Split documents into sentences."
)
group.add_argument(
    "--keep-newlines",
    action="store_true",
    help="Keep newlines between sentences when splitting.",
)
group = parser.add_argument_group(title="tokenization process")
group.add_argument(
    "--append-eod",
    action="store_true",
    help="Append an <eod> token to the end of a document.",
)
group.add_argument(
    "--lang",
    type=str,
    default="english",
    help="Language to use for NLTK-powered sentence splitting.",
)
group = parser.add_argument_group(title="output data")
group.add_argument(
    "--output-prefix",
    type=str,
    required=True,
    help="Path to binary output file without suffix",
)
group = parser.add_argument_group(title="runtime")
group.add_argument(
    "--workers",
    type=int,
    required=True,
    help=(
        "Number of worker processes to launch."
        "A good default for fast pre-processing "
        "is: (workers * partitions) = available CPU cores."
    ),
)
group.add_argument(
    "--partitions", type=int, default=1, help="Number of file partitions"
)
group.add_argument(
    "--log-interval", type=int, default=1000, help="Interval between progress updates"
)
group.add_argument(
    "--keep-sequential-samples",
    action="store_true",
    help="Ensure ordering of samples in .jsonl files is "
    "preserved when using partitions>1.",
)

args = parser.parse_args()


def preprocess_data(args):
    if args.split_sentences:
        if nltk_available:
            nltk.download("punkt", quiet=True, download_dir=os.environ.get("NLTK_DATA"))
        else:
            raise Exception(
                "nltk library required for sentence splitting is not available."
            )

    in_ss_out_names = []
    if args.partitions == 1:
        file_name, extension = os.path.splitext(args.input)
        sentence_split_file = file_name + "_ss" + extension
        file_names = {
            "partition": args.input,
            "sentence_split": sentence_split_file,
            "output_prefix": args.output_prefix,
        }
        in_ss_out_names.append(file_names)
    else:
        in_file_names = glob.glob(args.input)

        # Count total number of lines across .jsonl files
        if args.keep_sequential_samples:
            total_sample_count = 0
            for filename in in_file_names:
                with open(filename, "r") as fin:
                    for fc, _ in enumerate(fin):
                        pass
                total_sample_count += fc + 1
            partition_size = math.ceil(total_sample_count / args.partitions)

        # create .jsonl parition files
        for idx in range(args.partitions):
            in_ss_out_name = get_file_name(args, idx)
            in_ss_out_names.append(in_ss_out_name)

        # check to see if paritions were already created
        partitions_present = check_files_exist(
            in_ss_out_names, "partition", args.partitions
        )

        # check to see if paritions with split sentences already created
        split_sentences_present = check_files_exist(
            in_ss_out_names, "sentence_split", args.partitions
        )

        if not partitions_present and not split_sentences_present:
            # populate .jsonl partition files from parent files
            partitioned_input_files = []
            for idx in range(args.partitions):
                partitioned_input_file = open(in_ss_out_names[idx]["partition"], "w")
                partitioned_input_files.append(partitioned_input_file)

            index = 0
            if args.keep_sequential_samples:
                line_count = 0
            for in_file_name in in_file_names:
                # support for gzip files
                if in_file_name.endswith(".gz"):
                    fin = gzip.open(in_file_name, "rt")
                else:
                    fin = open(in_file_name, "r", encoding="utf-8")

                for line in fin:
                    partitioned_input_files[index].write(line)
                    if args.keep_sequential_samples:
                        line_count += 1
                        if line_count % partition_size == 0:
                            index += 1
                    else:
                        index = (index + 1) % args.partitions

                fin.close()

            for idx in range(args.partitions):
                partitioned_input_files[idx].close()

    assert args.workers % args.partitions == 0
    partition = Partition(args, args.workers // args.partitions)

    # check to see if paritions with split sentences already created
    split_sentences_present = check_files_exist(
        in_ss_out_names, "sentence_split", args.partitions
    )

    # split sentences in partition files
    if args.split_sentences and not split_sentences_present:
        processes = []
        for name in in_ss_out_names:
            p = multiprocessing.Process(
                target=partition.split_sentences,
                args=((name["partition"], name["sentence_split"]),),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        if args.partitions == 1:
            return

    # encode partition files in parallel
    processes = []
    input_key = "sentence_split" if args.split_sentences else "partition"
    for name in in_ss_out_names:
        p = multiprocessing.Process(
            target=partition.process_json_file,
            args=((name[input_key], name["output_prefix"]),),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if args.partitions == 1:
        return

    # merge bin/idx partitions
    level = "document"
    if args.split_sentences:
        level = "sentence"

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    tokenizer = build_tokenizer(args)

    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix, key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix, key, level)
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )

        for name in in_ss_out_names:
            parition_output_prefix = name["output_prefix"]
            full_partition_output_prefix = "{}_{}_{}".format(
                parition_output_prefix, key, level
            )
            builders[key].add_index(full_partition_output_prefix)
        builders[key].finalize(output_idx_files[key])


def merge_datasets(args):
    prefixes = set()
    for basename in os.listdir(args.input):
        prefix, ext = os.path.splitext(basename)

        if prefix in prefixes:
            continue

        if not os.path.isfile(os.path.join(args.input, basename)):
            continue

        ext_pair = ".bin" if ext == ".idx" else ".idx"
        assert os.path.isfile(
            os.path.join(args.input, prefix) + ext_pair
        ), f"ERROR: {ext_pair} file not provided for {os.path.join(args.input, prefix)}"

        prefixes.add(prefix)

    builder = None
    for prefix in sorted(prefixes):
        if builder is None:
            dataset = IndexedDataset(
                os.path.join(args.input, prefix), multimodal=args.multimodal
            )
            builder = IndexedDatasetBuilder(
                get_bin_path(args.output_prefix),
                dtype=dataset.index.dtype,
                multimodal=args.multimodal,
            )
            del dataset

        builder.add_index(os.path.join(args.input, prefix))

    builder.finalize(get_idx_path(args.output_prefix))


def convert_to_jsonl(input_file, temp_dir, json_keys, input_format="json"):
    # TODO: Add support for other formats
    base_name = os.path.basename(input_file).split(".")[0]
    output_file = os.path.join(temp_dir, f"{base_name}.jsonl")
    if input_format == "json":
        with open(input_file, "r") as f:
            data = json.load(f)
            with open(output_file, "w") as out:
                for line in data:
                    json.dump({k: line[k] for k in json_keys}, out)
                    out.write("\n")
    return output_file


@ray.remote(num_cpus=args.cpus_per_ray_worker)
def preprocess_data_ray(preprocess_data_args):
    # TODO: Convert to jsonl
    preprocess_data(preprocess_data_args)


if __name__ == "__main__":
    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))

    if num_nodes > 1:
        ray.init(address="auto")

    elif args.workers:
        ray.init(num_cpus=args.workers)

    ret = []
    output_dir = args.output_prefix
    os.makedirs(output_dir, exist_ok=True)
    temp_output_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_output_dir, exist_ok=True)

    all_jsonl_files = glob.glob(
        f"{args.input}/*.jsonl"
    )  # TODO: Add support for other formats
    logging.info(f"Found {len(all_jsonl_files)} files")

    for file in all_jsonl_files:
        output_prefix = os.path.join(temp_output_dir, os.path.basename(file))
        input_path = file
        preprocess_data_args = argparse.Namespace(
            input=input_path,
            json_keys=args.json_keys,
            split_sentences=args.split_sentences,
            keep_newlines=args.keep_newlines,
            append_eod=args.append_eod,
            lang=args.lang,
            output_prefix=output_prefix,
            workers=args.workers,
            partitions=args.partitions,
            log_interval=args.log_interval,
            keep_sequential_samples=args.keep_sequential_samples,
            tokenizer_type=args.tokenizer_type,
            vocab_size=args.vocab_size,
            vocab_file=args.vocab_file,
            merge_file=args.merge_file,
            tokenizer_model=args.tokenizer_model,
            tiktoken_pattern=args.tiktoken_pattern,
            tiktoken_num_special_tokens=args.tiktoken_num_special_tokens,
            tiktoken_special_tokens=args.tiktoken_special_tokens,
        )
        preprocess_data_args.rank = 1
        preprocess_data_args.make_vocab_size_divisible_by = 128
        preprocess_data_args.tensor_model_parallel_size = 1
        preprocess_data_args.vocab_extra_ids = 0
        ret.append(preprocess_data_ray.remote(preprocess_data_args))

    start = time.time()
    ray.get(ret)

    logging.info(f"Time taken: {time.time() - start}")
    ray.shutdown()

    logging.info("=====Merging datasets=====\n")

    merge_datasets_args = argparse.Namespace(
        input=temp_output_dir,
        output_prefix=os.path.join(output_dir, "merged"),
        multimodal=False,
    )

    merge_datasets(merge_datasets_args)

    shutil.rmtree(temp_output_dir)
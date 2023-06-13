import argparse
import logging
import os
from multiprocessing import cpu_count

from datasets import Features, Value, load_from_disk

from obelisc.processors import WarcDownloader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(description="Download warc files from Common Crawl pointers.")
    parser.add_argument(
        "idx_job",
        type=int,
        help="Index of the job (between 0 and 199).",
    )
    parser.add_argument(
        "--path_metadata_dataset",
        type=str,
        default="s3://m4-datasets/webdocs/pointers_cc_dataset/",
        help="Path of the dataset containing the metadata to retrieve the warc files.",
    )
    parser.add_argument(
        "--path_save_dir_warc_dataset",
        type=str,
        default="s3://m4-datasets/webdocs/warc_dataset/",
        help="The directory to save the warc dataset.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4 * cpu_count(),
        help="Number of processes to use for the multiprocessing.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    path_save_tmp_files = "/scratch/storage_hugo/"
    if os.path.exists(path_save_tmp_files):
        os.system(f"rm -r {path_save_tmp_files}")
    os.system(f"mkdir {path_save_tmp_files}")

    logger.info("Starting loading the metadata or previous warc dataset")
    path_sync_s3 = os.path.join(args.path_metadata_dataset, str(args.idx_job))
    path_save_disk_input = "/scratch/storage_hugo/pointers_cc_ds/"
    os.system(f"mkdir {path_save_disk_input}")
    command_sync_s3 = f"aws s3 sync {path_sync_s3} {path_save_disk_input}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    metadata_dataset = load_from_disk(path_save_disk_input)
    if ("warc" not in metadata_dataset.column_names) and ("warc_error" not in metadata_dataset.column_names):
        metadata_dataset = metadata_dataset.add_column("warc", [b""] * len(metadata_dataset))
        metadata_dataset = metadata_dataset.add_column("warc_error", [""] * len(metadata_dataset))
    logger.info("Finished loading the metadata or previous warc dataset")

    warc_downloader = WarcDownloader()
    logger.info("Starting downloading the warc files")
    warc_dataset = metadata_dataset.map(
        warc_downloader,
        num_proc=args.num_proc,
        features=Features(
            {
                **metadata_dataset.features,
                "warc": Value("binary"),
                "warc_error": Value("string"),
            }
        ),
    )
    logger.info("Finished downloading the warc files")

    logger.info("Starting saving the warc dataset")
    path_save_disk_output = "/scratch/storage_hugo/warc_ds"
    warc_dataset.save_to_disk(path_save_disk_output)
    path_sync_s3 = os.path.join(args.path_save_dir_warc_dataset, str(args.idx_job))
    command_sync_s3 = f"aws s3 sync {path_save_disk_output} {path_sync_s3}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    logger.info("Finished saving the warc dataset")

    logger.info("Starting computing the success rate")
    num_successes = len([1 for el in warc_dataset["warc_error"] if not el])
    logger.info(f"Success rate: {num_successes} / {len(warc_dataset)} ({num_successes / len(warc_dataset) * 100}%)")
    logger.info("Finished computing the success rate")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -r {path_save_tmp_files}")
    logger.info("Finished deleting the tmp files")

import argparse
import logging
from multiprocessing import cpu_count

from datasets import Features, Value, load_from_disk

from obelics.processors import WarcDownloader


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
        "--path_metadata_dataset",
        type=str,
        default="./large_files/metadata_dataset_10000",
        help="Path of the dataset containing the metadata to retrieve the warc files.",
    )
    parser.add_argument(
        "--path_save_dir_warc_dataset",
        type=str,
        default="./large_files/warc_dataset_10000",
        help="The directory to save the warc dataset.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=cpu_count(),
        help="Number of processes to use for the multiprocessing.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    logger.info("Starting loading the metadata or previous warc dataset")
    metadata_dataset = load_from_disk(args.path_metadata_dataset)
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
    warc_dataset.save_to_disk(args.path_save_dir_warc_dataset)
    logger.info("Finished saving the warc dataset")

    logger.info("Starting computing the success rate")
    num_successes = len([1 for el in warc_dataset["warc_error"] if not el])
    logger.info(f"Success rate: {num_successes} / {len(warc_dataset)} ({num_successes / len(warc_dataset) * 100}%)")
    logger.info("Finished computing the success rate")

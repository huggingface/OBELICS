import argparse
import logging
from multiprocessing import cpu_count

from datasets import load_from_disk

from obelics.processors import HtmlExtractor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(description="Extract html from warc files.")
    parser.add_argument(
        "--path_warc_dataset",
        type=str,
        default="./large_files/warc_dataset_10000",
        help="Path of the dataset containing the warc files to retrieve the html.",
    )
    parser.add_argument(
        "--path_save_dir_html_dataset",
        type=str,
        default="./large_files/html_dataset_10000",
        help="The directory to save the html dataset.",
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

    logger.info("Starting loading the warc or previous html dataset")
    warc_dataset = load_from_disk(args.path_warc_dataset)
    if ("html" not in warc_dataset.column_names) and ("html_error" not in warc_dataset.column_names):
        warc_dataset = warc_dataset.add_column("html", [""] * len(warc_dataset))
        warc_dataset = warc_dataset.add_column("html_error", [""] * len(warc_dataset))
    logger.info("Finished loading the warc or previous html dataset")

    html_extractor = HtmlExtractor()
    logger.info("Starting retrieving the html")
    html_dataset = warc_dataset.map(html_extractor, num_proc=args.num_proc)
    logger.info("Finished retrieving the html")

    logger.info("Starting saving the html dataset")
    html_dataset.save_to_disk(args.path_save_dir_html_dataset)
    logger.info("Finished saving the html dataset")

    logger.info("Starting computing the success rate")
    num_successes = len([1 for el in html_dataset["html_error"] if not el])
    logger.info(f"Success rate: {num_successes} / {len(html_dataset)} ({num_successes / len(html_dataset) * 100}%)")
    logger.info("Finished computing the success rate")

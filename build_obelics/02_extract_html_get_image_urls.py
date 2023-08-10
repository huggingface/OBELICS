import argparse
import logging
import os
from multiprocessing import cpu_count

from datasets import load_from_disk

from obelics.processors import (
    DOMTreeSimplificator,
    HtmlExtractor,
    PreExtractionSimplificator,
    WebDocumentExtractor,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(
        description="Extract html from warc files, simplify them, get the urls of the images."
    )
    parser.add_argument(
        "idx_job",
        type=int,
        help="Index of the job (between 0 and 199).",
    )
    parser.add_argument(
        "--path_warc_dataset",
        type=str,
        default="s3://m4-datasets/webdocs/warc_dataset/",
        help="Path of the dataset containing the warc files to retrieve the html.",
    )
    parser.add_argument(
        "--path_save_file_image_urls",
        type=str,
        default="/scratch/storage_hugo/image_urls.txt",
        help="The file to save the urls of all images.",
    )
    parser.add_argument(
        "--path_save_dir_html_dataset",
        type=str,
        default="s3://m4-datasets/webdocs/html_dataset/",
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

    path_save_tmp_files = "/scratch/storage_hugo/"
    if os.path.exists(path_save_tmp_files):
        os.system(f"rm -r {path_save_tmp_files}")
    os.system(f"mkdir {path_save_tmp_files}")

    logger.info("Starting loading the warc or previous html dataset")
    path_sync_s3 = os.path.join(args.path_warc_dataset, str(args.idx_job))
    path_save_disk_input = f"/scratch/storage_hugo/warc_dataset_{args.idx_job}"
    if os.path.exists(path_save_disk_input):
        os.system(f"rm -r {path_save_disk_input}")
    os.system(f"mkdir {path_save_disk_input}")
    command_sync_s3 = f"aws s3 sync {path_sync_s3} {path_save_disk_input}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    warc_dataset = load_from_disk(path_save_disk_input)
    if ("html" not in warc_dataset.column_names) and ("html_error" not in warc_dataset.column_names):
        warc_dataset = warc_dataset.add_column("html", [""] * len(warc_dataset))
        warc_dataset = warc_dataset.add_column("html_error", [""] * len(warc_dataset))
    logger.info("Finished loading the warc or previous html dataset")

    html_extractor = HtmlExtractor()
    logger.info("Starting retrieving the html")
    html_dataset = warc_dataset.map(html_extractor, num_proc=args.num_proc)
    logger.info("Finished retrieving the html")

    logger.info("Starting computing the success rate for the html extraction")
    num_successes = len([1 for el in html_dataset["html_error"] if not el])
    logger.info(
        f"Success rate for the html extraction: {num_successes} /"
        f" {len(html_dataset)} ({num_successes / len(html_dataset) * 100}%)"
    )
    logger.info("Finished computing the success rate for the html extraction")

    dom_tree_simplificator = DOMTreeSimplificator(
        strip_multiple_linebreaks=True,
        strip_multiple_spaces=True,
        remove_html_comments=True,
        replace_line_break_tags=True,
        unwrap_tags=True,
        strip_tags=True,
        strip_special_divs=True,
        remove_dates=True,
        remove_empty_leaves=True,
        unnest_nodes=True,
        remake_tree=True,
    )
    pre_extraction_simplificator = PreExtractionSimplificator(
        only_text_image_nodes=True,
        format_texts=True,
        merge_consecutive_text_nodes=True,
    )
    web_document_extractor = WebDocumentExtractor(
        html_dataset=html_dataset,
        dom_tree_simplificator=dom_tree_simplificator,
        pre_extraction_simplificator=pre_extraction_simplificator,
        path_save_dir_dataset=None,
        num_proc=args.num_proc,
        path_save_file_image_urls=args.path_save_file_image_urls,
        path_save_dir_downloaded_images=None,
        thread_count=None,
        number_sample_per_shard=None,
        image_size=None,
        resize_mode=None,
        path_save_dir_tmp_datasets_images=None,
        path_save_dir_dataset_images=None,
        path_save_file_map_url_idx=None,
        num_proc_urls_to_images=None,
        path_save_dir_sharded_dataset=None,
        shard_size=None,
    )

    web_document_extractor.html_to_web_documents()
    html_dataset = web_document_extractor.dataset

    web_document_extractor.get_image_urls()
    path_sync_s3 = os.path.join("s3://m4-datasets/webdocs/image_urls/", str(args.idx_job), "image_urls.txt")
    command_sync_s3 = f"aws s3 cp {args.path_save_file_image_urls} {path_sync_s3}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    logger.info("Starting saving the html dataset")
    path_save_disk_output = f"/scratch/storage_hugo/html_dataset_{args.idx_job}"
    if os.path.exists(path_save_disk_output):
        os.system(f"rm -r {path_save_disk_output}")
    html_dataset.save_to_disk(path_save_disk_output)

    path_sync_s3 = os.path.join(args.path_save_dir_html_dataset, str(args.idx_job))
    command_sync_s3 = f"aws s3 sync {path_save_disk_output} {path_sync_s3}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    logger.info("Finished saving the html dataset")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -r {path_save_tmp_files}")
    logger.info("Finished deleting the tmp files")

import argparse
import logging
import os
from multiprocessing import cpu_count

import yaml
from datasets import load_from_disk

from obelisc.processors import (
    CommonCrawlWebDocumentExtractor,
    DOMTreeSimplificator,
    HtmlExtractor,
    PreExtractionSimplificator,
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
        "--path_save_dir_web_document_dataset_without_images",
        type=str,
        default="s3://m4-datasets/webdocs/web_document_dataset_without_images/",
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

    path_save_tmp_files = f"/scratch/storage_hugo_{args.idx_job}/"
    if os.path.exists(path_save_tmp_files):
        os.system(f"rm -r {path_save_tmp_files}")
    os.system(f"mkdir {path_save_tmp_files}")

    logger.info("Starting loading the warc or previous html dataset")
    path_sync_s3 = os.path.join(args.path_warc_dataset, str(args.idx_job))
    path_save_disk_input = os.path.join(path_save_tmp_files, "warc_dataset")
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

    path_save_file_image_urls = os.path.join(path_save_tmp_files, "image_urls.txt")

    with open("./m4/sourcing/data_collection/configs/config_extract_web_documents.yaml") as f:
        extraction_params = yaml.load(f, Loader=yaml.FullLoader)

    dom_tree_simplificator = DOMTreeSimplificator(
        strip_multiple_linebreaks=extraction_params["dom_tree_simplificator"]["strip_multiple_linebreaks"],
        strip_multiple_spaces=extraction_params["dom_tree_simplificator"]["strip_multiple_spaces"],
        remove_html_comments=extraction_params["dom_tree_simplificator"]["remove_html_comments"],
        replace_line_break_tags=extraction_params["dom_tree_simplificator"]["replace_line_break_tags"],
        unwrap_tags=extraction_params["dom_tree_simplificator"]["unwrap_tags"],
        strip_tags=extraction_params["dom_tree_simplificator"]["strip_tags"],
        strip_special_divs=extraction_params["dom_tree_simplificator"]["strip_special_divs"],
        remove_dates=extraction_params["dom_tree_simplificator"]["remove_dates"],
        remove_empty_leaves=extraction_params["dom_tree_simplificator"]["remove_empty_leaves"],
        unnest_nodes=extraction_params["dom_tree_simplificator"]["unnest_nodes"],
        remake_tree=extraction_params["dom_tree_simplificator"]["remake_tree"],
        css_rules=extraction_params["dom_tree_simplificator"]["css_rules"],
        css_rules_replace_with_text=extraction_params["dom_tree_simplificator"]["css_rules_replace_with_text"],
    )
    pre_extraction_simplificator = PreExtractionSimplificator(
        only_text_image_nodes=extraction_params["pre_extraction_simplificator"]["only_text_image_nodes"],
        format_texts=extraction_params["pre_extraction_simplificator"]["format_texts"],
        merge_consecutive_text_nodes=extraction_params["pre_extraction_simplificator"]["merge_consecutive_text_nodes"],
    )
    web_document_extractor = CommonCrawlWebDocumentExtractor(
        html_dataset=html_dataset,
        dom_tree_simplificator=dom_tree_simplificator,
        pre_extraction_simplificator=pre_extraction_simplificator,
        path_save_dir_dataset=None,
        num_proc=args.num_proc,
        path_save_file_image_urls=path_save_file_image_urls,
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
    web_document_dataset_without_images = web_document_extractor.dataset

    web_document_extractor.get_image_urls()
    path_sync_s3 = os.path.join("s3://m4-datasets/webdocs/image_urls_new_rules/", str(args.idx_job), "image_urls.txt")
    command_sync_s3 = f"aws s3 cp {path_save_file_image_urls} {path_sync_s3}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    logger.info("Starting saving the web document dataset without the images")
    path_save_disk_output = os.path.join(path_save_tmp_files, "web_document_dataset_without_images")
    web_document_dataset_without_images.save_to_disk(path_save_disk_output)

    path_sync_s3 = os.path.join(args.path_save_dir_web_document_dataset_without_images, str(args.idx_job))
    command_sync_s3 = f"aws s3 sync {path_save_disk_output} {path_sync_s3}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    logger.info("Finished saving the web document dataset without the images")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -r {path_save_tmp_files}")
    logger.info("Finished deleting the tmp files")

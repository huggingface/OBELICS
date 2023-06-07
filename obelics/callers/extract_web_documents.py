import argparse
import logging
from multiprocessing import cpu_count

import yaml
from datasets import load_from_disk

from obelics.processors import (
    CommonCrawlWebDocumentExtractor,
    DOMTreeSimplificator,
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
    parser = argparse.ArgumentParser(description="Extract web documents.")
    parser.add_argument(
        "--path_config_extract_web_documents",
        type=str,
        default="./obelics/configs/config_extract_web_documents.yaml",
        help="The path of the config file containing the extraction parameters.",
    )
    parser.add_argument(
        "--path_html_dataset",
        type=str,
        default="./large_files/html_documents_10000",
        help="Path of the dataset containing the HTML documents.",
    )
    parser.add_argument(
        "--path_save_dir_dataset",
        type=str,
        default="./large_files/output_extraction/web_documents_10000",
        help="The directory to save the dataset.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=cpu_count(),
        help="Number of processes to use for the multiprocessing.",
    )
    parser.add_argument(
        "--path_save_file_image_urls",
        type=str,
        default="./large_files/output_extraction/image_urls.txt",
        help="The file to save the urls of all images.",
    )
    parser.add_argument(
        "--path_save_dir_downloaded_images",
        type=str,
        default="./large_files/output_extraction/downloaded_images",
        help="The directory to save all images.",
    )
    parser.add_argument(
        "--thread_count",
        type=int,
        default=256,
        help="The number of threads used for downloading the pictures.",
    )
    parser.add_argument(
        "--number_sample_per_shard",
        type=int,
        default=10_000,
        help="The number of images that will be downloaded in one shard.",
    )
    parser.add_argument(
        "--path_save_dir_tmp_datasets_images",
        type=str,
        default="./large_files/output_extraction/tmp_datasets_images",
        help=(
            "The directory to save the temporary datasets containing all images (useful for the code but can be"
            " forgotten after)."
        ),
    )
    parser.add_argument(
        "--path_save_dir_dataset_images",
        type=str,
        default="./large_files/output_extraction/dataset_images",
        help="The directory to save the dataset containing all images.",
    )
    parser.add_argument(
        "--path_save_file_map_url_idx",
        type=str,
        default="./large_files/output_extraction/map_url_idx.json",
        help="The file to save the map to go from urls to indices of the dataset containing all images.",
    )
    parser.add_argument(
        "--num_proc_urls_to_images",
        type=int,
        default=15,
        help="Number of processes to use for the multiprocessing for the step `urls_to_images`. Reduce if OOM errors.",
    )
    parser.add_argument(
        "--path_save_dir_sharded_dataset",
        type=str,
        default="./large_files/output_extraction/web_documents_10000_sharded",
        help="The directory to save the sharded dataset.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=20_000,  # 500 shards for 10M web documents
        help="The size of a shard for the sharded dataset.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    logger.info("Starting loading the HTML dataset")
    html_dataset = load_from_disk(args.path_html_dataset)
    logger.info("Finished loading the HTML dataset")

    with open(args.path_config_extract_web_documents) as f:
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
    path_save_dir_dataset = args.path_save_dir_dataset
    num_proc = args.num_proc
    path_save_file_image_urls = args.path_save_file_image_urls
    path_save_dir_downloaded_images = args.path_save_dir_downloaded_images
    thread_count = args.thread_count
    number_sample_per_shard = args.number_sample_per_shard
    image_size = extraction_params["web_document_extractor"]["image_size"]
    resize_mode = extraction_params["web_document_extractor"]["resize_mode"]
    path_save_dir_tmp_datasets_images = args.path_save_dir_tmp_datasets_images
    path_save_dir_dataset_images = args.path_save_dir_dataset_images
    path_save_file_map_url_idx = args.path_save_file_map_url_idx
    num_proc_urls_to_images = args.num_proc_urls_to_images
    path_save_dir_sharded_dataset = args.path_save_dir_sharded_dataset
    shard_size = args.shard_size

    web_document_extractor = CommonCrawlWebDocumentExtractor(
        html_dataset=html_dataset,
        dom_tree_simplificator=dom_tree_simplificator,
        pre_extraction_simplificator=pre_extraction_simplificator,
        path_save_dir_dataset=path_save_dir_dataset,
        num_proc=num_proc,
        path_save_file_image_urls=path_save_file_image_urls,
        path_save_dir_downloaded_images=path_save_dir_downloaded_images,
        thread_count=thread_count,
        number_sample_per_shard=number_sample_per_shard,
        image_size=image_size,
        resize_mode=resize_mode,
        path_save_dir_tmp_datasets_images=path_save_dir_tmp_datasets_images,
        path_save_dir_dataset_images=path_save_dir_dataset_images,
        path_save_file_map_url_idx=path_save_file_map_url_idx,
        num_proc_urls_to_images=num_proc_urls_to_images,
        path_save_dir_sharded_dataset=path_save_dir_sharded_dataset,
        shard_size=shard_size,
    )

    web_document_extractor.html_to_web_documents()
    web_document_extractor.get_image_urls()

    web_document_extractor.download_images()

    web_document_extractor.create_dataset_images()

    web_document_extractor.urls_to_images()

    web_document_extractor.save_dataset()
    web_document_extractor.save_commit_hash()
    web_document_extractor.save_split_sharded_dataset()

import argparse
import logging
import os
from multiprocessing import cpu_count

from obelisc.processors import WebDocumentExtractor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(description="Download images and create a dataset containing them.")
    parser.add_argument(
        "idx_job",
        type=int,
        help="Index of the job (between 0 and 199).",
    )
    parser.add_argument(
        "--U",
        type=int,
        default=0,
        help="Indicate if the download of the images is already done.",
    )
    parser.add_argument(
        "--download_only",
        type=int,
        default=0,
        help="Indicate if we only want to download the images, and not create the image dataset.",
    )
    parser.add_argument(
        "--path_image_urls",
        type=str,
        default="s3://m4-datasets/webdocs/image_urls_2/",
        help="The path of the file containing the urls of all images.",
    )
    parser.add_argument(
        "--path_save_dir_downloaded_images",
        type=str,
        default="/scratch/storage_hugo/downloaded_images",
        help="The directory to save all images.",
    )
    parser.add_argument(
        "--thread_count",
        type=int,
        default=128,
        help="The number of threads used for downloading the pictures.",
    )
    parser.add_argument(
        "--number_sample_per_shard",
        type=int,
        default=10_000,
        help="The number of images that will be downloaded in one shard.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="The size to resize image to. Not used if --resize_mode=no.",
    )
    parser.add_argument(
        "--resize_mode",
        type=str,
        default="no",
        help="The way to resize pictures, can be no, border or keep_ratio.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=cpu_count(),
        help="Number of processes to use for the multiprocessing.",
    )
    parser.add_argument(
        "--path_save_dir_tmp_datasets_images",
        type=str,
        default="/scratch/storage_hugo/tmp_datasets_images",
        help=(
            "The directory to save the temporary datasets containing all images (useful for the code but can be"
            " forgotten after)."
        ),
    )
    parser.add_argument(
        "--path_save_dir_dataset_images",
        type=str,
        default="s3://m4-datasets/webdocs/image_dataset_2/",
        help="The directory to save the dataset containing all images.",
    )
    parser.add_argument(
        "--path_save_file_map_url_idx",
        type=str,
        default="s3://m4-datasets/webdocs/map_url_idx_2/",
        help="The file to save the map to go from urls to indices of the dataset containing all images.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    path_save_tmp_files = "/scratch/storage_hugo/"
    if args.U == 0:
        if os.path.exists(path_save_tmp_files):
            os.system(f"rm -r {path_save_tmp_files}")
    os.system(f"mkdir -p {path_save_tmp_files}")

    path_save_dir_downloaded_images = args.path_save_dir_downloaded_images
    os.system(f"mkdir -p {path_save_dir_downloaded_images}")

    path_save_dir_tmp_datasets_images = args.path_save_dir_tmp_datasets_images
    os.system(f"mkdir -p {path_save_dir_tmp_datasets_images}")

    path_image_urls = os.path.join(args.path_image_urls, str(args.idx_job), "image_urls.txt")
    path_disk_image_urls = "/scratch/storage_hugo/image_urls.txt"
    command_sync_s3 = f"aws s3 cp {path_image_urls} {path_disk_image_urls}"
    if args.U == 0:
        os.system(command_sync_s3)
        os.system(command_sync_s3)

    path_save_dir_dataset_images = os.path.join(args.path_save_dir_dataset_images, str(args.idx_job))
    path_disk_save_dir_dataset_images = "/scratch/storage_hugo/image_dataset"

    path_save_file_map_url_idx = os.path.join(args.path_save_file_map_url_idx, str(args.idx_job), "map_url_idx.json")
    path_disk_save_file_map_url_idx = "/scratch/storage_hugo/map_url_idx.json"

    web_document_extractor = WebDocumentExtractor(
        html_dataset=None,
        dom_tree_simplificator=None,
        pre_extraction_simplificator=None,
        path_save_dir_dataset=None,
        num_proc=args.num_proc,
        path_save_file_image_urls=path_disk_image_urls,
        path_save_dir_downloaded_images=path_save_dir_downloaded_images,
        thread_count=args.thread_count,
        number_sample_per_shard=args.number_sample_per_shard,
        image_size=args.image_size,
        resize_mode=args.resize_mode,
        path_save_dir_tmp_datasets_images=path_save_dir_tmp_datasets_images,
        path_save_dir_dataset_images=path_disk_save_dir_dataset_images,
        path_save_file_map_url_idx=path_disk_save_file_map_url_idx,
        num_proc_urls_to_images=None,
        path_save_dir_sharded_dataset=None,
        shard_size=None,
    )

    if args.U == 0:
        web_document_extractor.download_images()

    if args.download_only == 0:
        web_document_extractor.create_dataset_images()

        logger.info("Starting computing the success rate for downloading of the images")
        with open(path_disk_image_urls, "r") as file:
            lines = file.readlines()
            num_tot_images = len(lines)
        num_successes = len(web_document_extractor.dataset_images)
        logger.info(
            f"Success rate for downloading of the images: {num_successes} /"
            f" {num_tot_images} ({num_successes / num_tot_images * 100}%)"
        )
        logger.info("Finished computing the success rate for downloading of the images")

        logger.info("Starting saving the image dataset and the map")
        command_sync_s3 = f"aws s3 cp {path_disk_save_file_map_url_idx} {path_save_file_map_url_idx}"
        os.system(command_sync_s3)
        os.system(command_sync_s3)
        os.system(command_sync_s3)

        command_sync_s3 = f"aws s3 sync {path_disk_save_dir_dataset_images} {path_save_dir_dataset_images}"
        os.system(command_sync_s3)
        os.system(command_sync_s3)
        os.system(command_sync_s3)
        logger.info("Finished saving the image dataset and the map")

        logger.info("Starting deleting the tmp files")
        os.system(f"rm -r {path_save_tmp_files}")
        logger.info("Finished deleting the tmp files")

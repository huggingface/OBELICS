import argparse
import logging
import os
from copy import deepcopy
from multiprocessing import cpu_count

import datasets
from datasets import concatenate_datasets, load_from_disk
from PIL import Image, ImageFile


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

URL_BAN_WORDS = ["logo", "button", "icon", "plugin", "widget", "porn", "xxx", "sex"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(
        description="Merge the web document dataset without images with the dataset of images."
    )
    parser.add_argument(
        "idx_job",
        type=int,
        help="Index of the job (between 0 and 199).",
    )
    parser.add_argument(
        "--path_web_document_dataset_without_images",
        type=str,
        default="s3://m4-datasets/webdocs/web_document_dataset_without_images/",
        help="Path of the web document dataset without the images.",
    )
    parser.add_argument(
        "--path_image_dataset_1",
        type=str,
        default="s3://m4-datasets/webdocs/image_dataset/",
        help="Path of the dataset containing the images.",
    )
    parser.add_argument(
        "--path_image_dataset_2",
        type=str,
        default="s3://m4-datasets/webdocs/image_dataset_2/",
        help="Path of the second dataset containing the images.",
    )
    parser.add_argument(
        "--path_save_dir_web_document_dataset",
        type=str,
        default="s3://m4-datasets/webdocs/web_document_dataset/",
        help="Path to save the web document dataset with the images.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=cpu_count(),
        help="Number of processes to use for the multiprocessing.",
    )
    args = parser.parse_args()
    return args


def urls_to_images(web_document_dataset_without_images, image_dataset, map_url_idx, url_ban_words, num_proc):
    def retrieve_image(url):
        if url not in map_url_idx:
            return None
        if any([url_ban_word in url for url_ban_word in url_ban_words]):
            return None
        # Uncomment if one process seems silently killed without throwing any error in the `map` function.
        # It's rare, but approximately 1/100M pages contain many huge pictures that break things.
        # 2M in bytes was chosen by looking at the distribution of the length in bytes of pictures.
        # It would remove only 1/1000 picture.
        # if len(image_dataset[map_url_idx[url]]["image"]) > 2_000_000:
        #     return None
        image = {"path": None, "bytes": image_dataset[map_url_idx[url]]["image"]}
        return image

    def func_urls_to_images_urls_in_images_col(example):
        # Uncomment if one process seems silently killed without throwing any error in the `map` function.
        # It's rare, but approximately 1/100M pages contain many huge pictures that break things.
        # num_images = len([1 for url in example["images"] if url in map_url_idx])
        # if num_images > 50:
        #    example["images"] = [None for url in example["images"]]
        #    return example
        example["images"] = [retrieve_image(url) if url else None for url in example["images"]]
        return example

    logger.info("Starting replacing urls by images")
    new_features = deepcopy(web_document_dataset_without_images.features)
    new_features["images"] = datasets.Sequence(datasets.Image())
    web_document_dataset = web_document_dataset_without_images.map(
        func_urls_to_images_urls_in_images_col,
        features=new_features,
        num_proc=num_proc,
        load_from_cache_file=False,
    )
    logger.info("Finished replacing urls by images")
    return web_document_dataset


if __name__ == "__main__":
    args = get_args()

    path_save_disk_tmp_files = f"/scratch/storage_hugo_{args.idx_job}/"
    if os.path.exists(path_save_disk_tmp_files):
        os.system(f"rm -r {path_save_disk_tmp_files}")
    os.system(f"mkdir {path_save_disk_tmp_files}")

    logger.info("Starting loading the previous web document dataset without the images")
    path_sync_s3 = os.path.join(args.path_web_document_dataset_without_images, str(args.idx_job))
    path_save_disk_web_document_dataset_without_images = os.path.join(
        path_save_disk_tmp_files, "web_document_dataset_without_images"
    )
    os.system(f"mkdir {path_save_disk_web_document_dataset_without_images}")
    command_sync_s3 = f"aws s3 sync {path_sync_s3} {path_save_disk_web_document_dataset_without_images}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    web_document_dataset_without_images = load_from_disk(path_save_disk_web_document_dataset_without_images)
    logger.info("Finished loading the previous web document dataset without the images")

    logger.info("Starting loading the image datasets and the mapping")
    path_sync_s3 = os.path.join(args.path_image_dataset_1, str(args.idx_job))
    path_save_disk_image_dataset_1 = os.path.join(path_save_disk_tmp_files, "image_dataset_1")
    os.system(f"mkdir {path_save_disk_image_dataset_1}")
    command_sync_s3 = f"aws s3 sync {path_sync_s3} {path_save_disk_image_dataset_1}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    image_dataset_1 = load_from_disk(path_save_disk_image_dataset_1)

    path_sync_s3 = os.path.join(args.path_image_dataset_2, str(args.idx_job))
    path_save_disk_image_dataset_2 = os.path.join(path_save_disk_tmp_files, "image_dataset_2")
    os.system(f"mkdir {path_save_disk_image_dataset_2}")
    command_sync_s3 = f"aws s3 sync {path_sync_s3} {path_save_disk_image_dataset_2}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    image_dataset_2 = load_from_disk(path_save_disk_image_dataset_2)

    logger.info("Starting concatenating the image datasets")
    image_dataset = concatenate_datasets([image_dataset_1, image_dataset_2])
    logger.info("Finished concatenating the image datasets")
    logger.info("Starting creating the mapping")
    map_url_idx = {url: idx for idx, url in enumerate(image_dataset["url"])}
    logger.info("Finished creating the mapping")

    logger.info("Finished loading the image datasets and the mapping")

    logger.info("Starting to merge the web document dataset without images and the dataset containing the images")
    web_document_dataset = urls_to_images(
        web_document_dataset_without_images, image_dataset, map_url_idx, URL_BAN_WORDS, args.num_proc
    )
    logger.info("Finished to merge the web document dataset without images and the dataset containing the images")

    logger.info("Starting saving the web document dataset")
    path_save_disk_web_document_dataset = os.path.join(path_save_disk_tmp_files, "web_document_dataset")
    web_document_dataset.save_to_disk(path_save_disk_web_document_dataset, num_proc=args.num_proc)

    path_sync_s3 = os.path.join(args.path_save_dir_web_document_dataset, str(args.idx_job))
    command_sync_s3 = f"aws s3 sync {path_save_disk_web_document_dataset} {path_sync_s3}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    logger.info("Finished saving the web document dataset")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -r {path_save_disk_tmp_files}")
    logger.info("Finished deleting the tmp files")

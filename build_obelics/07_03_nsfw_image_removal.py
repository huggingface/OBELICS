import json
import logging
import os
import sys
from multiprocessing import cpu_count

from datasets import load_from_disk
from PIL import Image, ImageFile


# Useful to avoid DecompressionBombError and truncated image error
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


IDX_JOB = sys.argv[1]
PATH_SAVE_DISK_TMP_FILES = f"/scratch/storage_hugo_{IDX_JOB}/"

PATH_NSFW_IMAGE_URLS_S3 = f"s3://m4-datasets/webdocs/nsfw_image_urls/{IDX_JOB}/nsfw_image_urls.json"
PATH_NSFW_IMAGE_URLS_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "nsfw_image_urls.json")

PATH_WEB_DOCS_S3 = f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup/{IDX_JOB}/"
PATH_WEB_DOCS_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_document_dataset_filtered_imgurldedup")

NUM_PROC = cpu_count()

PATH_SAVE_DISK_WEB_DOCS_NSFW_FILTERED = os.path.join(
    PATH_SAVE_DISK_TMP_FILES, "web_document_dataset_filtered_imgurldedup_nsfwfiltered"
)
PATH_SAVE_S3_WEB_DOCS_NSFW_FILTERED = os.path.join(
    "s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered", str(IDX_JOB)
)


class NSFWFiltering:
    def __init__(self, path_nsfw_image_urls):
        self.path_nsfw_image_urls = path_nsfw_image_urls
        with open(path_nsfw_image_urls) as f:
            self.nsfw_image_urls = set(json.load(f))

    def __call__(self, example):
        image_urls_in_example = [el["src"] for el in json.loads(example["metadata"]) if el]
        if any([url in self.nsfw_image_urls for url in image_urls_in_example]):
            return False
        return True

    def __reduce__(self):
        return self.__class__, (self.path_nsfw_image_urls,)


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")

    logger.info("Starting downloading the set of NSFW image urls")
    command_sync_s3 = f"aws s3 cp {PATH_NSFW_IMAGE_URLS_S3} {PATH_NSFW_IMAGE_URLS_LOCAL}"
    os.system(command_sync_s3)
    logger.info("Finished downloading the set of NSFW image urls")

    logger.info("Starting loading the web docs")
    command_sync_s3 = f"aws s3 sync {PATH_WEB_DOCS_S3} {PATH_WEB_DOCS_LOCAL}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    web_docs_dataset = load_from_disk(PATH_WEB_DOCS_LOCAL)
    num_docs_before_filtering = web_docs_dataset.num_rows
    logger.info("Finished loading the web docs")

    logger.info("Starting removing the documents containing NSFW images")
    nsfw_filtering = NSFWFiltering(path_nsfw_image_urls=PATH_NSFW_IMAGE_URLS_LOCAL)
    web_docs_dataset = web_docs_dataset.filter(nsfw_filtering, num_proc=NUM_PROC)
    logger.info("Finished removing the documents containing NSFW images")

    logger.info("Starting saving the web document dataset after the NSFW filtering")
    web_docs_dataset.save_to_disk(PATH_SAVE_DISK_WEB_DOCS_NSFW_FILTERED, num_proc=NUM_PROC)

    command_sync_s3 = f"aws s3 sync {PATH_SAVE_DISK_WEB_DOCS_NSFW_FILTERED} {PATH_SAVE_S3_WEB_DOCS_NSFW_FILTERED}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    logger.info("Finished saving the web document dataset after the NSFW filtering")

    logger.info(
        f"Number of documents in the web document dataset before the NSFW filtering: {num_docs_before_filtering}"
    )
    logger.info(
        f"Number of documents in the web document dataset after the NSFW filtering: {web_docs_dataset.num_rows}"
    )

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")

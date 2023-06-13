import json
import logging
import os
import sys

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


MAX_NUM_RETRIES_SYNC = 3

IDX_JOB = sys.argv[1]
PATH_SAVE_DISK_TMP_FILES = f"/scratch/storage_hugo_{IDX_JOB}/"

PATH_DUP_URLS_S3 = "s3://m4-datasets/webdocs/dup_urls.json"
PATH_DUP_URLS_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "dup_urls.json")

PATH_WEB_DOCS_S3 = f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered/{IDX_JOB}/"
PATH_WEB_DOCS_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_docs")

NUM_PROC = 10

PATH_SAVE_DISK_WEB_DOCS_URL_DEDUP = os.path.join(
    PATH_SAVE_DISK_TMP_FILES, "web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup"
)
PATH_SAVE_S3_WEB_DOCS_URL_DEDUP = os.path.join(
    "s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup", str(IDX_JOB)
)


class URLDeduplication:
    def __init__(self, path_dup_urls):
        self.path_dup_urls = path_dup_urls
        with open(path_dup_urls) as f:
            self.dup_urls = json.load(f)

    def __call__(self, example):
        general_metadata = json.loads(example["general_metadata"])
        url, warc_filename = general_metadata["url"], general_metadata["warc_filename"]
        if url in self.dup_urls:
            if warc_filename != self.dup_urls[url]:
                return False
        # Bonus: removes documents without any images
        metadata = [meta for meta in json.loads(example["metadata"]) if meta]
        if not metadata:
            return False
        return True

    def __reduce__(self):
        return self.__class__, (self.path_dup_urls,)


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")

    logger.info("Starting downloading the set of duplicated urls")
    command_sync_s3 = f"aws s3 cp {PATH_DUP_URLS_S3} {PATH_DUP_URLS_LOCAL}"
    os.system(command_sync_s3)
    logger.info("Finished downloading the set of duplicated urls")

    logger.info("Starting loading the web docs")
    command_sync_s3 = f"aws s3 sync {PATH_WEB_DOCS_S3} {PATH_WEB_DOCS_LOCAL}"
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)

    web_docs_dataset = load_from_disk(PATH_WEB_DOCS_LOCAL)
    num_docs_before_filtering = web_docs_dataset.num_rows
    logger.info("Finished loading the web docs")

    logger.info("Starting deduplicating documents on URLs")
    url_deduplication = URLDeduplication(path_dup_urls=PATH_DUP_URLS_LOCAL)
    web_docs_dataset = web_docs_dataset.filter(url_deduplication, num_proc=NUM_PROC)
    logger.info("Finished deduplicating documents on URLs")

    logger.info("Starting saving the web document dataset after the URL deduplication")
    web_docs_dataset.save_to_disk(PATH_SAVE_DISK_WEB_DOCS_URL_DEDUP, num_proc=NUM_PROC)

    command_sync_s3 = f"aws s3 sync {PATH_SAVE_DISK_WEB_DOCS_URL_DEDUP} {PATH_SAVE_S3_WEB_DOCS_URL_DEDUP}"
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)
    logger.info("Finished saving the web document dataset after the URL deduplication")

    logger.info(
        "Number of documents in the web document dataset before the URL deduplication and the removal of documents"
        f" without images): {num_docs_before_filtering}"
    )
    logger.info(
        "Number of documents in the web document dataset after the URL deduplication and the removal of documents"
        f" without images): {web_docs_dataset.num_rows}"
    )

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")

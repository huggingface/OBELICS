import json
import logging
import os
import pickle
import sys
from collections import Counter

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

PATH_BAN_URL_WARCFILENAME_S3 = "s3://m4-datasets/webdocs/url_to_warcfilename_to_remove.json"
PATH_BAN_URL_WARCFILENAME_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "url_to_warcfilename_to_remove.json")

PATH_WEB_DOCS_S3 = f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_linededup_finalcleaning/{IDX_JOB}/"
PATH_WEB_DOCS_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_docs")

NUM_PROC = 10

PATH_SAVE_DISK_WEB_DOCS_SET_IMG_URLS_DEDUP = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_docs_setimgurlsdedup")
PATH_SAVE_S3_WEB_DOCS_SET_IMG_URLS_DEDUP = f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_linededup_finalcleaning_setimgurlsdedup/{IDX_JOB}/"

PATH_SAVE_DISK_IMG_URLS_IN_FINAL_WEB_DOCS = os.path.join(PATH_SAVE_DISK_TMP_FILES, "img_urls.pickle")
PATH_SAVE_S3_IMG_URLS_IN_FINAL_WEB_DOCS = (
    f"s3://m4-datasets/webdocs/img_urls_in_final_web_docs_2/{IDX_JOB}/img_urls.pickle"
)


class SetImgURLsDeduplication:
    def __init__(self, path_ban_url_warc_filename):
        self.path_ban_url_warc_filename = path_ban_url_warc_filename
        with open(path_ban_url_warc_filename) as f:
            self.ban_url_warc_filename = json.load(f)

    def __call__(self, example):
        general_metadata = json.loads(example["general_metadata"])
        url, warc_filename = general_metadata["url"], general_metadata["warc_filename"]
        if url in self.ban_url_warc_filename:
            if warc_filename == self.ban_url_warc_filename[url]:
                return False
        return True

    def __reduce__(self):
        return self.__class__, (self.path_ban_url_warc_filename,)


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")

    logger.info("Starting downloading the list of urls and warc filenames to remove")
    command_sync_s3 = f"aws s3 cp {PATH_BAN_URL_WARCFILENAME_S3} {PATH_BAN_URL_WARCFILENAME_LOCAL}"
    os.system(command_sync_s3)
    logger.info("Finished downloading the list of urls and warc filenames to remove")

    logger.info("Starting loading the web docs")
    command_sync_s3 = f"aws s3 sync {PATH_WEB_DOCS_S3} {PATH_WEB_DOCS_LOCAL}"
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)

    web_docs_dataset = load_from_disk(PATH_WEB_DOCS_LOCAL)
    num_docs_before_filtering = web_docs_dataset.num_rows
    logger.info("Finished loading the web docs")

    logger.info("Starting removing the unwanted documents")
    set_img_urls_deduplication = SetImgURLsDeduplication(path_ban_url_warc_filename=PATH_BAN_URL_WARCFILENAME_LOCAL)
    web_docs_dataset = web_docs_dataset.filter(set_img_urls_deduplication, num_proc=NUM_PROC)
    logger.info("Finished removing the unwanted documents")

    logger.info("Starting saving the web document dataset after the removal of the unwanted documents")
    web_docs_dataset.save_to_disk(PATH_SAVE_DISK_WEB_DOCS_SET_IMG_URLS_DEDUP, num_proc=NUM_PROC)

    command_sync_s3 = (
        f"aws s3 sync {PATH_SAVE_DISK_WEB_DOCS_SET_IMG_URLS_DEDUP} {PATH_SAVE_S3_WEB_DOCS_SET_IMG_URLS_DEDUP}"
    )
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)
    logger.info("Finished saving the web document dataset after the removal of the unwanted documents")

    logger.info("Starting saving the image urls in the web document dataset")
    img_urls = [[el["src"] for el in json.loads(md) if el] for md in web_docs_dataset["metadata"]]
    img_urls = [sub_el for el in img_urls for sub_el in el]
    img_urls = Counter(img_urls)

    with open(PATH_SAVE_DISK_IMG_URLS_IN_FINAL_WEB_DOCS, "wb") as f:
        pickle.dump(img_urls, f, pickle.HIGHEST_PROTOCOL)
    command_sync_s3 = (
        f"aws s3 cp {PATH_SAVE_DISK_IMG_URLS_IN_FINAL_WEB_DOCS} {PATH_SAVE_S3_IMG_URLS_IN_FINAL_WEB_DOCS}"
    )
    os.system(command_sync_s3)
    logger.info("Finished saving the image urls in the web document dataset")

    logger.info(
        "Number of documents in the web document dataset before the deduplication on the set of image urls:"
        f" {num_docs_before_filtering}"
    )
    logger.info(
        "Number of documents in the web document dataset after the deduplication on the set of image urls:"
        f" {web_docs_dataset.num_rows}"
    )

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")

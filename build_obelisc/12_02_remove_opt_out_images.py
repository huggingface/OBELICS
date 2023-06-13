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

MAX_NUM_RETRIES_SYNC = 3

PATH_OPT_OUT_URLS_S3 = (  # Made by converting s3://m4-datasets/webdocs/ds_opt_out/ to a list and saving in json
    "s3://m4-datasets/webdocs/list_opt_out.json"
)
PATH_OPT_OUT_URLS_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "list_opt_out.json")

PATH_WEB_DOCS_S3 = f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_linededup_finalcleaning_setimgurlsdedup/{IDX_JOB}/"
PATH_WEB_DOCS_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_docs")

NUM_PROC = cpu_count()

PATH_SAVE_DISK_WEB_DOCS_OPT_OUT_FILTERED = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_docs_optoutrmv")
PATH_SAVE_S3_WEB_DOCS_OPT_OUT_FILTERED = f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_linededup_finalcleaning_setimgurlsdedup_optoutrmv/{IDX_JOB}/"


class OptOutFiltering:
    def __init__(self, path_opt_out_image_urls):
        self.path_opt_out_image_urls = path_opt_out_image_urls
        with open(path_opt_out_image_urls) as f:
            self.opt_out_image_urls = set(json.load(f))

    def __call__(self, example):
        metadata = json.loads(example["metadata"])
        indices_to_remove = [
            ind for ind, meta in enumerate(metadata) if (meta is not None) and (meta["src"] in self.opt_out_image_urls)
        ]
        if indices_to_remove:
            example["texts"] = [el for ind, el in enumerate(example["texts"]) if ind not in indices_to_remove]
            example["images"] = [el for ind, el in enumerate(example["images"]) if ind not in indices_to_remove]
            example["metadata"] = json.dumps([el for ind, el in enumerate(metadata) if ind not in indices_to_remove])
        return example

    def __reduce__(self):
        return self.__class__, (self.path_opt_out_image_urls,)


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")

    logger.info("Starting downloading the list of opt out image urls")
    command_sync_s3 = f"aws s3 cp {PATH_OPT_OUT_URLS_S3} {PATH_OPT_OUT_URLS_LOCAL}"
    os.system(command_sync_s3)
    logger.info("Finished downloading the list of opt out image urls")

    logger.info("Starting loading the web docs")
    command_sync_s3 = f"aws s3 sync {PATH_WEB_DOCS_S3} {PATH_WEB_DOCS_LOCAL}"
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)

    web_docs = load_from_disk(PATH_WEB_DOCS_LOCAL)
    logger.info("Finished loading the web docs")

    logger.info("Starting removing the opt out images")
    opt_out_filtering = OptOutFiltering(path_opt_out_image_urls=PATH_OPT_OUT_URLS_LOCAL)
    web_docs = web_docs.map(opt_out_filtering, num_proc=NUM_PROC)
    logger.info("Finished removing the opt out images")

    logger.info("Starting saving the web document dataset with the opt out images removed")
    web_docs.save_to_disk(PATH_SAVE_DISK_WEB_DOCS_OPT_OUT_FILTERED, num_proc=NUM_PROC)

    command_sync_s3 = (
        f"aws s3 sync {PATH_SAVE_DISK_WEB_DOCS_OPT_OUT_FILTERED} {PATH_SAVE_S3_WEB_DOCS_OPT_OUT_FILTERED}"
    )
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)
    logger.info("Finished saving the web document dataset with the opt out images removed")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")

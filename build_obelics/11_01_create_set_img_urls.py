import json
import logging
import os
import sys

from datasets import load_from_disk


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


MAX_NUM_RETRIES_SYNC = 3

IDX_JOB = int(sys.argv[1])
PATH_SAVE_DISK_TMP_FILES = f"/scratch/storage_hugo_{IDX_JOB}/"

PATH_WEB_DOCS_S3 = f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_linededup_finalcleaning/{IDX_JOB}"
PATH_WEB_DOCS_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_docs")

PATH_SAVE_DISK_SET_IMG_URLS = os.path.join(PATH_SAVE_DISK_TMP_FILES, "set_img_urls.json")
PATH_SAVE_S3_SET_IMG_URLS = f"s3://m4-datasets/webdocs/set_img_urls/{IDX_JOB}/set_img_urls.json"


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")

    logger.info("Starting downloading the web document dataset")
    command_sync_s3 = f"aws s3 sync {PATH_WEB_DOCS_S3} {PATH_WEB_DOCS_LOCAL}"
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)

    web_docs = load_from_disk(PATH_WEB_DOCS_LOCAL)
    logger.info("Finished downloading the web document dataset")

    logger.info("Starting getting the set of image urls for each document")
    metadata = web_docs["metadata"]
    img_urls = [[el["src"] for el in json.loads(md) if el] for md in metadata]
    img_urls = [list(set(img_urls_)) for img_urls_ in img_urls]
    img_urls = ["".join(sorted(img_urls_)) for img_urls_ in img_urls]

    general_metadata = web_docs["general_metadata"]
    urls = [json.loads(gmd)["url"] for gmd in general_metadata]
    warc_filenames = [json.loads(gmd)["warc_filename"] for gmd in general_metadata]

    img_urls_with_general_metadata = [
        [img_urls_, url, warc_filename] for img_urls_, url, warc_filename in zip(img_urls, urls, warc_filenames)
    ]
    logger.info("Finished getting the set of image urls for each document")

    logger.info("Starting saving the set of image urls")
    with open(PATH_SAVE_DISK_SET_IMG_URLS, "w") as f:
        json.dump(img_urls_with_general_metadata, f)
    command_sync_s3 = f"aws s3 cp {PATH_SAVE_DISK_SET_IMG_URLS} {PATH_SAVE_S3_SET_IMG_URLS}"
    os.system(command_sync_s3)
    logger.info("Finished saving the set of image urls")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")

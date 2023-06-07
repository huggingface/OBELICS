import logging
import os
import sys
from multiprocessing import cpu_count

from datasets import load_from_disk
from PIL import Image, ImageFile


# Useful to avoid DecompressionBombError
Image.MAX_IMAGE_PIXELS = None
# Load even truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


IDX_JOB = int(sys.argv[1])
PATH_SAVE_DISK_TMP_FILES = f"/scratch/storage_hugo_{IDX_JOB}/"

PATH_WEB_DOCS_LINE_DEDUP_TEXTS_ONLY_S3 = f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_linededup_texts_only/{IDX_JOB}"
PATH_WEB_DOCS_LINE_DEDUP_TEXTS_ONLY_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_docs_linededup_texts_only")

PATH_WEB_DOCS_S3 = (
    f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup/{IDX_JOB}"
)
PATH_WEB_DOCS_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_docs")

NUM_PROC = cpu_count()

PATH_SAVE_DISK_WEB_DOCS_LINE_DEDUP = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_docs_linededup")
PATH_SAVE_S3_WEB_DOCS_LINE_DEDUP = (
    f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_linededup/{IDX_JOB}"
)


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")

    logger.info(
        "Starting downloading the web document dataset and the texts of the web document dataset line deduplicated"
    )
    command_sync_s3 = (
        f"aws s3 sync {PATH_WEB_DOCS_LINE_DEDUP_TEXTS_ONLY_S3} {PATH_WEB_DOCS_LINE_DEDUP_TEXTS_ONLY_LOCAL}"
    )
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    web_docs_line_dedup_texts_only = load_from_disk(PATH_WEB_DOCS_LINE_DEDUP_TEXTS_ONLY_LOCAL)

    command_sync_s3 = f"aws s3 sync {PATH_WEB_DOCS_S3} {PATH_WEB_DOCS_LOCAL}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    web_docs = load_from_disk(PATH_WEB_DOCS_LOCAL)
    logger.info(
        "Finished downloading the web document dataset and the texts of the web document dataset line deduplicated"
    )

    logger.info("Starting merging the two datasets")
    web_docs = web_docs.remove_columns("texts")
    web_docs = web_docs.add_column("texts", web_docs_line_dedup_texts_only["texts"])
    logger.info("Finished merging the two datasets")

    logger.info("Starting saving the web document dataset line deduplicated")
    web_docs.save_to_disk(PATH_SAVE_DISK_WEB_DOCS_LINE_DEDUP, num_proc=NUM_PROC)

    command_sync_s3 = f"aws s3 sync {PATH_SAVE_DISK_WEB_DOCS_LINE_DEDUP} {PATH_SAVE_S3_WEB_DOCS_LINE_DEDUP}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    logger.info("Finished saving the web document dataset line deduplicated")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")

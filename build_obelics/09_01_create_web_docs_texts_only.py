import logging
import os
import sys
from multiprocessing import cpu_count

from datasets import load_from_disk


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


IDX_JOB = int(sys.argv[1])
PATH_SAVE_DISK_TMP_FILES = f"/scratch/storage_hugo_{IDX_JOB}/"

PATH_WEB_DOCS_S3 = (
    f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup/{IDX_JOB}"
)
PATH_WEB_DOCS_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_docs")

NUM_PROC = cpu_count()

PATH_SAVE_DISK_WEB_DOCS_TEXTS_ONLY = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_docs_texts_only")
PATH_SAVE_S3_WEB_DOCS_TEXTS_ONLY = (
    f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_texts_only/{IDX_JOB}"
)


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")

    logger.info("Starting downloading the web document dataset")
    command_sync_s3 = f"aws s3 sync {PATH_WEB_DOCS_S3} {PATH_WEB_DOCS_LOCAL}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    web_document_dataset = load_from_disk(PATH_WEB_DOCS_LOCAL)
    logger.info("Finished downloading the web document dataset")

    logger.info("Starting removing the columns other than `texts`")
    web_document_dataset = web_document_dataset.remove_columns(
        [c_n for c_n in web_document_dataset.column_names if c_n not in ["texts", "general_metadata"]]
    )
    logger.info("Finished removing the columns other than `texts`")

    logger.info("Starting saving the web document dataset with texts only")
    web_document_dataset.save_to_disk(PATH_SAVE_DISK_WEB_DOCS_TEXTS_ONLY, num_proc=NUM_PROC)

    command_sync_s3 = f"aws s3 sync {PATH_SAVE_DISK_WEB_DOCS_TEXTS_ONLY} {PATH_SAVE_S3_WEB_DOCS_TEXTS_ONLY}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    logger.info("Finished saving the web document dataset with texts only")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")

import json
import logging
import os
import sys
from urllib.parse import urlparse

from datasets import load_from_disk


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


IDX_JOB = sys.argv[1]
PATH_SAVE_DISK_TMP_FILES = f"/scratch/storage_hugo_{IDX_JOB}/"

PATH_DOMAIN_TO_DUPLICATED_TEXTS_S3 = "s3://m4-datasets/webdocs/new_line_dedup_domain_to_duplicated_texts.json"
PATH_DOMAIN_TO_DUPLICATED_TEXTS_LOCAL = os.path.join(
    PATH_SAVE_DISK_TMP_FILES, "new_line_dedup_domain_to_duplicated_texts.json"
)
PATH_REDUCED_DOMAIN_TO_DUPLICATED_TEXTS_LOCAL = os.path.join(
    PATH_SAVE_DISK_TMP_FILES, "reduced_new_line_dedup_domain_to_duplicated_texts.json"
)

PATH_WEB_DOCS_S3 = (
    f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_texts_only/{IDX_JOB}/"
)
PATH_WEB_DOCS_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_docs")

NUM_PROC = 20

PATH_SAVE_DISK_WEB_DOCS_LINE_DEDUP = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_docs_linededup")
PATH_SAVE_S3_WEB_DOCS_LINE_DEDUP = f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_linededup_texts_only/{IDX_JOB}/"


class LineDeduplication:
    def __init__(self, path_domain_to_duplicated_texts):
        self.path_domain_to_duplicated_texts = path_domain_to_duplicated_texts
        with open(path_domain_to_duplicated_texts) as f:
            self.domain_to_duplicated_texts = json.load(f)

    def __call__(self, example):
        domain = urlparse(json.loads(example["general_metadata"])["url"]).netloc
        if domain not in self.domain_to_duplicated_texts:
            return example

        for idx in range(len(example["texts"])):
            if example["texts"][idx] is not None:
                example["texts"][idx] = "\n\n".join(
                    [
                        paragraph
                        for paragraph in example["texts"][idx].split("\n\n")
                        if paragraph not in self.domain_to_duplicated_texts[domain]
                    ]
                )
        return example

    def __reduce__(self):
        return self.__class__, (self.path_domain_to_duplicated_texts,)


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")

    logger.info("Starting downloading the dictionary to go from a domain to the associated duplicated texts")
    command_sync_s3 = f"aws s3 cp {PATH_DOMAIN_TO_DUPLICATED_TEXTS_S3} {PATH_DOMAIN_TO_DUPLICATED_TEXTS_LOCAL}"
    os.system(command_sync_s3)
    logger.info("Finished downloading the dictionary to go from a domain to the associated duplicated texts")

    logger.info("Starting loading the web docs")
    command_sync_s3 = f"aws s3 sync {PATH_WEB_DOCS_S3} {PATH_WEB_DOCS_LOCAL}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    web_docs_dataset = load_from_disk(PATH_WEB_DOCS_LOCAL)
    logger.info("Finished loading the web docs")

    logger.info("Starting reducing the dictionary to go from a domain to the associated duplicated texts")
    domains_in_shard = set([urlparse(json.loads(meta)["url"]).netloc for meta in web_docs_dataset["general_metadata"]])
    with open(PATH_DOMAIN_TO_DUPLICATED_TEXTS_LOCAL) as f:
        domain_to_duplicated_texts = json.load(f)
    reduced_domain_to_duplicated_texts = {k: v for k, v in domain_to_duplicated_texts.items() if k in domains_in_shard}
    with open(PATH_REDUCED_DOMAIN_TO_DUPLICATED_TEXTS_LOCAL, "w") as f:
        json.dump(reduced_domain_to_duplicated_texts, f)
    del domain_to_duplicated_texts
    del reduced_domain_to_duplicated_texts
    logger.info("Finished reducing the dictionary to go from a domain to the associated duplicated texts")

    logger.info("Starting line deduplicating documents")
    line_deduplication = LineDeduplication(
        path_domain_to_duplicated_texts=PATH_REDUCED_DOMAIN_TO_DUPLICATED_TEXTS_LOCAL
    )
    web_docs_dataset = web_docs_dataset.map(line_deduplication, num_proc=NUM_PROC)
    logger.info("Finished line deduplicating documents")

    logger.info("Starting saving the web document dataset after the line deduplication")
    web_docs_dataset.save_to_disk(PATH_SAVE_DISK_WEB_DOCS_LINE_DEDUP, num_proc=NUM_PROC)

    command_sync_s3 = f"aws s3 sync {PATH_SAVE_DISK_WEB_DOCS_LINE_DEDUP} {PATH_SAVE_S3_WEB_DOCS_LINE_DEDUP}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    logger.info("Finished saving the web document dataset after the line deduplication")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")

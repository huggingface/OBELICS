import json
import logging
import os
from urllib.parse import urlparse

from datasets import load_from_disk
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


NUM_SHARDS = 200

PATH_SAVE_DISK_TMP_FILES = "/scratch/storage_hugo/"

PATH_WEB_DOCS_S3 = (
    "s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_texts_only/"
)
PATH_WEB_DOCS_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_docs")

PATH_SAVE_DISK_LINE_DEDUP_DOMAIN_TO_POSITIONS = os.path.join(
    PATH_SAVE_DISK_TMP_FILES, "line_dedup_domain_to_positions.json"
)
PATH_SAVE_S3_LINE_DEDUP_DOMAIN_TO_POSITIONS = "s3://m4-datasets/webdocs/line_dedup_domain_to_positions.json"


def get_domain_to_positions():
    domain_to_positions = {}

    for idx_shard in tqdm(range(NUM_SHARDS)):
        path_subdataset = os.path.join(PATH_WEB_DOCS_LOCAL, str(idx_shard))
        sub_ds = load_from_disk(path_subdataset)
        metadata_sub_ds = sub_ds["general_metadata"]
        domains = [urlparse(json.loads(meta)["url"]).netloc for meta in metadata_sub_ds]

        new_domain_to_pos = {}
        for idx, domain in enumerate(domains):
            new_domain_to_pos[domain] = new_domain_to_pos.get(domain, []) + [idx]
        for domain in new_domain_to_pos:
            if domain not in domain_to_positions:
                domain_to_positions[domain] = {}
            domain_to_positions[domain][str(idx_shard)] = new_domain_to_pos[domain]

    return domain_to_positions


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")

    logger.info("Starting downloading the web document dataset (texts only)")
    command_sync_s3 = f"aws s3 sync {PATH_WEB_DOCS_S3} {PATH_WEB_DOCS_LOCAL}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    logger.info("Finished downloading the web document dataset (texts only)")

    logger.info("Starting creating the dictionary to go from a domain name to positions in the web document dataset")
    domain_to_positions = get_domain_to_positions()
    logger.info("Finished creating the dictionary to go from a domain name to positions in the web document dataset")

    logger.info("Starting saving the domain to positions")
    with open(PATH_SAVE_DISK_LINE_DEDUP_DOMAIN_TO_POSITIONS, "w") as f:
        json.dump(domain_to_positions, f)

    command_sync_s3 = (
        f"aws s3 cp {PATH_SAVE_DISK_LINE_DEDUP_DOMAIN_TO_POSITIONS} {PATH_SAVE_S3_LINE_DEDUP_DOMAIN_TO_POSITIONS}"
    )
    os.system(command_sync_s3)
    logger.info("Finished saving the domain to positions")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")

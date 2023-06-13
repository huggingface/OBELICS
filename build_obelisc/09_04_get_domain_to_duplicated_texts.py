import json
import logging
import os
import sys

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

IDX_JOB = int(sys.argv[1])
PATH_SAVE_DISK_TMP_FILES = f"/scratch/storage_hugo_{IDX_JOB}/"

PATH_WEB_DOCS_S3 = (
    "s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_texts_only/"
)
PATH_WEB_DOCS_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_docs")

PATH_LINE_DEDUP_DOMAIN_TO_POSITIONS_S3 = (
    f"s3://m4-datasets/webdocs/line_dedup_domain_to_positions_sharded/{IDX_JOB}/line_dedup_domain_to_positions.json"
)
PATH_LINE_DEDUP_DOMAIN_TO_POSITIONS_LOCAL = os.path.join(
    PATH_SAVE_DISK_TMP_FILES, "line_dedup_domain_to_positions.json"
)

PATH_SAVE_DISK_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS = os.path.join(
    PATH_SAVE_DISK_TMP_FILES, "line_dedup_domain_to_duplicated_texts.json"
)
PATH_SAVE_S3_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS = f"s3://m4-datasets/webdocs/line_dedup_domain_to_duplicated_texts_sharded/{IDX_JOB}/line_dedup_domain_to_duplicated_texts.json"


def get_domain_to_duplicated_texts(domain_to_positions):
    shard_to_domain_to_positions = {
        str(idx_shard): {
            domain: domain_to_positions[domain][str(idx_shard)]
            for domain in domain_to_positions
            if str(idx_shard) in domain_to_positions[domain]
        }
        for idx_shard in range(NUM_SHARDS)
    }
    domain_to_duplicated_texts = {}

    for idx_shard in tqdm(range(NUM_SHARDS)):
        ds_shard = load_from_disk(os.path.join(PATH_WEB_DOCS_LOCAL, str(idx_shard)), keep_in_memory=True)

        for domain in shard_to_domain_to_positions[str(idx_shard)]:
            if domain not in domain_to_duplicated_texts:
                domain_to_duplicated_texts[domain] = {}

            positions = shard_to_domain_to_positions[str(idx_shard)][domain]

            for pos in positions:
                tot_texts = [txt for txt in ds_shard[pos]["texts"] if txt]
                tot_texts = [text.split("\n\n") for text in tot_texts]
                tot_texts = [paragraph for text in tot_texts for paragraph in text]
                for text in tot_texts:
                    domain_to_duplicated_texts[domain][text] = domain_to_duplicated_texts[domain].get(text, 0) + 1

    domain_to_duplicated_texts = {
        domain: {k: v for k, v in domain_to_duplicated_texts[domain].items() if v > 1}
        for domain in domain_to_duplicated_texts
    }
    return domain_to_duplicated_texts


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")

    logger.info(
        "Starting downloading the web document dataset (texts only) and to dictionary to go from a domain to positions"
    )
    command_sync_s3 = f"aws s3 sync {PATH_WEB_DOCS_S3} {PATH_WEB_DOCS_LOCAL}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    command_sync_s3 = f"aws s3 cp {PATH_LINE_DEDUP_DOMAIN_TO_POSITIONS_S3} {PATH_LINE_DEDUP_DOMAIN_TO_POSITIONS_LOCAL}"
    os.system(command_sync_s3)

    with open(PATH_LINE_DEDUP_DOMAIN_TO_POSITIONS_LOCAL) as f:
        domain_to_positions = json.load(f)
    logger.info(
        "Finished downloading the web document dataset (texts only) and to dictionary to go from a domain to positions"
    )

    logger.info("Starting finding the duplicated texts for each domain")
    domain_to_duplicated_texts = get_domain_to_duplicated_texts(domain_to_positions)
    logger.info("Finished finding the duplicated texts for each domain")

    logger.info("Starting saving the domain to duplicated texts")
    with open(PATH_SAVE_DISK_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS, "w") as f:
        json.dump(domain_to_duplicated_texts, f)

    command_sync_s3 = (
        "aws s3 cp"
        f" {PATH_SAVE_DISK_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS} {PATH_SAVE_S3_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS}"
    )
    os.system(command_sync_s3)
    logger.info("Finished saving the domain to duplicated texts")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")

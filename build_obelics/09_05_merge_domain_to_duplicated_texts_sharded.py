"""
srun --pty --cpus-per-task=96 bash -i
conda activate /fsx/m4/conda/shared-m4-2023-03-10
"""


import json
import logging
import os

from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


NUM_SHARDS = 200

PATH_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS_S3 = (
    "s3://m4-datasets/webdocs/line_dedup_domain_to_duplicated_texts_sharded/"
)
PATH_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS_LOCAL = "/scratch/line_dedup_domain_to_duplicated_texts_sharded/"

PATH_SAVE_DISK_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS_FULL = "/scratch/line_dedup_domain_to_duplicated_texts.json"
PATH_SAVE_S3_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS_FULL = (
    "s3://m4-datasets/webdocs/line_dedup_domain_to_duplicated_texts.json"
)

PATH_SAVE_DISK_NEW_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS_FULL = (
    "/scratch/new_line_dedup_domain_to_duplicated_texts.json"
)
PATH_SAVE_S3_NEW_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS_FULL = (
    "s3://m4-datasets/webdocs/new_line_dedup_domain_to_duplicated_texts.json"
)


if __name__ == "__main__":
    logger.info("Starting downloading the dictionaries to go from a domain to the associated duplicated texts")
    command_sync_s3 = (
        "aws s3 sync"
        f" {PATH_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS_S3} {PATH_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS_LOCAL}"
    )
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    logger.info("Finished downloading the dictionaries to go from a domain to the associated duplicated texts")

    logger.info("Starting merging the sub dictionaries")
    all_domain_to_duplicated_texts = []
    for idx_shard in tqdm(range(NUM_SHARDS)):
        with open(
            os.path.join(
                PATH_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS_LOCAL,
                str(idx_shard),
                "line_dedup_domain_to_duplicated_texts.json",
            )
        ) as f:
            all_domain_to_duplicated_texts.append(json.load(f))

    domain_to_duplicated_texts = {
        k: v for sub_dict in tqdm(all_domain_to_duplicated_texts) for k, v in sub_dict.items()
    }
    logger.info("Finished merging the sub dictionaries")

    logger.info("Starting saving the dictionary to go from a domain to the associated duplicated texts")
    with open(PATH_SAVE_DISK_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS_FULL, "w") as f:
        json.dump(domain_to_duplicated_texts, f)

    command_sync_s3 = (
        "aws s3 cp"
        f" {PATH_SAVE_DISK_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS_FULL} {PATH_SAVE_S3_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS_FULL}"
    )
    os.system(command_sync_s3)
    logger.info("Finished saving the dictionary to go from a domain to the associated duplicated texts")

    # Find the strategy
    # data = {k: v for k, v in domain_to_duplicated_texts.items() if len(v) > 0}
    # keys = list(data.keys())
    # print([(idx, len(data[keys[idx]])) for idx in range(len(keys))])
    # print("\n\n".join([k + f"\t{str(v)}" for k, v in {k: v for k, v in data[keys[5258]].items() if v > 2}.items()]))

    logger.info(
        "Starting making a smaller version of the dictionary, based on only what we will remove in the line"
        " deduplication"
    )
    new_domain_to_duplicated_texts = {
        k: {txt: counter for txt, counter in v.items() if counter > 2}
        for k, v in tqdm(domain_to_duplicated_texts.items())
    }
    new_domain_to_duplicated_texts = {
        k: {txt: counter for txt, counter in v.items() if "END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED" not in txt}
        for k, v in tqdm(new_domain_to_duplicated_texts.items())
    }
    new_domain_to_duplicated_texts = {k: v for k, v in new_domain_to_duplicated_texts.items() if len(v) > 0}
    logger.info(
        "Finished making a smaller version of the dictionary, based on only what we will remove in the line"
        " deduplication"
    )

    logger.info("Starting saving the new dictionary to go from a domain to the associated duplicated texts")
    with open(PATH_SAVE_DISK_NEW_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS_FULL, "w") as f:
        json.dump(new_domain_to_duplicated_texts, f)

    command_sync_s3 = (
        "aws s3 cp"
        f" {PATH_SAVE_DISK_NEW_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS_FULL} {PATH_SAVE_S3_NEW_LINE_DEDUP_DOMAIN_TO_DUPLICATED_TEXTS_FULL}"
    )
    os.system(command_sync_s3)
    logger.info("Finished saving the new dictionary to go from a domain to the associated duplicated texts")

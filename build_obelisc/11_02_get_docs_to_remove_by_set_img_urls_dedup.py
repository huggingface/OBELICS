"""
srun --pty --cpus-per-task=96 bash -i
conda activate /fsx/m4/conda/shared-m4-2023-03-10
"""


import json
import logging
import os
from collections import Counter

from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


MAX_NUM_RETRIES_SYNC = 3

NUM_SHARDS = 200

PATH_SAVE_DISK_TMP_FILES = "/scratch/storage_hugo/"

PATH_SET_IMG_URLS_S3 = "s3://m4-datasets/webdocs/set_img_urls/"
PATH_SET_IMG_URLS_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "set_img_urls")

PATH_SAVE_DISK_URL_TO_WARCFILENAME_TO_REMOVE = os.path.join(
    PATH_SAVE_DISK_TMP_FILES, "url_to_warcfilename_to_remove.json"
)
PATH_SAVE_S3_URL_TO_WARCFILENAME_TO_REMOVE = "s3://m4-datasets/webdocs/url_to_warcfilename_to_remove.json"


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")

    logger.info("Starting downloading the sets of image urls")
    command_sync_s3 = f"aws s3 sync {PATH_SET_IMG_URLS_S3} {PATH_SET_IMG_URLS_LOCAL}"
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)
    logger.info("Finished downloading the sets of image urls")

    logger.info("Starting opening the sets of image urls")
    sets_img_urls = []
    for idx_shard in tqdm(range(NUM_SHARDS)):
        with open(os.path.join(PATH_SET_IMG_URLS_LOCAL, str(idx_shard), "set_img_urls.json")) as f:
            sets_img_urls.append(json.load(f))
    logger.info("Finished opening the sets of image urls")

    logger.info("Starting finding the documents to remove")
    all_sets_img_urls = [[el[0] for el in set_img_urls] for set_img_urls in tqdm(sets_img_urls)]
    all_sets_img_urls = [sub_el for el in all_sets_img_urls for sub_el in el]
    all_sets_img_urls = Counter(all_sets_img_urls)

    dup_sets_img_urls = {k: v for k, v in all_sets_img_urls.items() if v > 1}

    print(f"Total number of documents: {sum(list(all_sets_img_urls.values()))}")
    # Total number of documents: 158_875_682
    print(
        "Number of documents for which at least one other document has the same images:"
        f" {sum(list(dup_sets_img_urls.values()))}"
    )
    # Number of documents for which at least one other document has the same images: 30_198_113
    print(
        f"We can remove {sum(list(dup_sets_img_urls.values())) - len(dup_sets_img_urls)} documents with the"
        " deduplication on the set of image urls"
    )
    # We can remove 17_708_135 documents with the deduplication on the set of image urls

    all_sets_img_urls = [el for set_img_urls in tqdm(sets_img_urls) for el in set_img_urls]
    dup_set_img_urls_to_warcfilename = {}
    for set_img_urls_, url, warc_filename in tqdm(all_sets_img_urls):
        if set_img_urls_ in dup_sets_img_urls:
            dup_set_img_urls_to_warcfilename[set_img_urls_] = dup_set_img_urls_to_warcfilename.get(
                set_img_urls_, []
            ) + [(url, warc_filename)]

    # We keep the latest document of a group with a common set of images and remove the other documents
    url_to_warcfilename_to_remove = {
        k: sorted(v, key=lambda x: x[1])[:-1] for k, v in tqdm(dup_set_img_urls_to_warcfilename.items())
    }
    url_to_warcfilename_to_remove = {
        url: warc_filename for _, v in tqdm(url_to_warcfilename_to_remove.items()) for url, warc_filename in v
    }
    logger.info("Starting finding the documents to remove")

    logger.info("Starting saving the combination of urls and warc filenames to remove")
    with open(PATH_SAVE_DISK_URL_TO_WARCFILENAME_TO_REMOVE, "w") as f:
        json.dump(url_to_warcfilename_to_remove, f)

    command_sync_s3 = (
        f"aws s3 cp {PATH_SAVE_DISK_URL_TO_WARCFILENAME_TO_REMOVE} {PATH_SAVE_S3_URL_TO_WARCFILENAME_TO_REMOVE }"
    )
    os.system(command_sync_s3)
    logger.info("Finished saving the combination of urls and warc filenames to remove")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")

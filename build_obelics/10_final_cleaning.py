import json
import logging
import os
import pickle
import sys
from collections import Counter
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

PATH_WEB_DOCS_S3 = (
    f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_linededup/{IDX_JOB}"
)
PATH_WEB_DOCS_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_docs")

NUM_PROC = cpu_count()

PATH_SAVE_DISK_WEB_DOCS_FINAL_CLEANING = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_docs_finalcleaning")
PATH_SAVE_S3_WEB_DOCS_FINAL_CLEANING = f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_linededup_finalcleaning/{IDX_JOB}"

PATH_SAVE_DISK_IMG_URLS_IN_FINAL_WEB_DOCS = os.path.join(PATH_SAVE_DISK_TMP_FILES, "img_urls.pickle")
PATH_SAVE_S3_IMG_URLS_IN_FINAL_WEB_DOCS = (
    f"s3://m4-datasets/webdocs/img_urls_in_final_web_docs/{IDX_JOB}/img_urls.pickle"
)


def func_map_final_cleaning_node_level(example):
    texts = example["texts"]
    images = example["images"]
    metadata = json.loads(example["metadata"])
    assert len(texts) == len(images) == len(metadata)

    new_texts = []
    new_images = []
    new_metadata = []

    previous_is_text = False
    for text, image, meta in zip(texts, images, metadata):
        if text is not None:
            assert (image is None) and (meta is None)
            if text == "":
                continue
            if previous_is_text:
                new_texts[-1] = new_texts[-1] + "\n\n" + text
            else:
                new_texts.append(text)
                new_images.append(None)
                new_metadata.append(None)
                previous_is_text = True
        elif image is not None:
            assert (text is None) and (meta is not None)
            new_texts.append(None)
            new_images.append(image)
            new_metadata.append(meta)
            previous_is_text = False
        elif meta is not None:
            raise ValueError("metadata cannot be != None if text and image are None")

    assert len(new_texts) == len(new_images) == len(new_metadata)
    example["texts"] = new_texts
    example["images"] = new_images
    example["metadata"] = json.dumps(new_metadata)

    return example


def func_filter_final_cleaning_doc_level(example):
    texts_example = example["texts"]
    texts = [txt for txt in texts_example if txt]
    images = [txt for txt in texts_example if not txt]
    if not texts or not images:
        return False
    return True


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")

    logger.info("Starting downloading the web document dataset")
    command_sync_s3 = f"aws s3 sync {PATH_WEB_DOCS_S3} {PATH_WEB_DOCS_LOCAL}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    web_docs = load_from_disk(PATH_WEB_DOCS_LOCAL)
    logger.info("Finished downloading the web document dataset")

    logger.info("Starting doing the final cleaning")
    web_docs = web_docs.map(func_map_final_cleaning_node_level, num_proc=NUM_PROC)
    web_docs = web_docs.filter(func_filter_final_cleaning_doc_level, num_proc=NUM_PROC)
    logger.info("Finished doing the final cleaning")

    logger.info("Starting saving the web document dataset after the final cleaning")
    web_docs.save_to_disk(PATH_SAVE_DISK_WEB_DOCS_FINAL_CLEANING, num_proc=NUM_PROC)

    command_sync_s3 = f"aws s3 sync {PATH_SAVE_DISK_WEB_DOCS_FINAL_CLEANING} {PATH_SAVE_S3_WEB_DOCS_FINAL_CLEANING}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    logger.info("Finished saving the web document dataset after the final cleaning")

    logger.info("Starting saving the image urls in the web document dataset")
    img_urls = [[el["src"] for el in json.loads(md) if el] for md in web_docs["metadata"]]
    img_urls = [sub_el for el in img_urls for sub_el in el]
    img_urls = Counter(img_urls)

    with open(PATH_SAVE_DISK_IMG_URLS_IN_FINAL_WEB_DOCS, "wb") as f:
        pickle.dump(img_urls, f, pickle.HIGHEST_PROTOCOL)
    command_sync_s3 = (
        f"aws s3 cp {PATH_SAVE_DISK_IMG_URLS_IN_FINAL_WEB_DOCS} {PATH_SAVE_S3_IMG_URLS_IN_FINAL_WEB_DOCS}"
    )
    os.system(command_sync_s3)
    logger.info("Finished saving the image urls in the web document dataset")

    logger.info(f"Number of documents in the web document dataset after the final cleaning: {web_docs.num_rows}")
    logger.info(
        "Number of images (with duplicates) in the web document dataset after the final cleaning:"
        f" {sum(list(img_urls.values()))}"
    )

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")

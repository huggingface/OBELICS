import json
import logging
import os
import pickle
import sys
from collections import Counter
from copy import deepcopy

import datasets
from datasets import load_from_disk
from PIL import Image, ImageFile

from m4.sourcing.data_collection.processors.web_document_filtering import FilteringFunctions
from m4.sourcing.data_collection.utils.filtering_utils import SPECIAL_CHARACTERS


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


SPAM_WORDS = [
    "facebook",
    "twitter",
    "instagram",
    "reddit",
    "youtube",
    "pinterest",
    "flickr",
    "share",
    "tweet",
    "post",
    "comment",
    "subscribe",
    "newletter",
    "blogger",
    "bloggers",
    "interested",
    "might",
    "like",
    "sign-up",
    "sign",
    "log",
    "logged",
    "access",
    "contact",
    "content",
    "privacy",
    "policy",
    "website",
    "cookies",
    "cookie",
    "licensed",
    "password",
    "account",
    "follow",
    "terms",
    "mailing",
    "list",
    "download",
    "loading",
    "click",
]
SPAM_WORD_RATIO_CUTOFF = 0.12

IDX_JOB = int(sys.argv[1])
PATH_SAVE_DISK_TMP_FILES = f"/scratch/storage_hugo_{IDX_JOB}/"

MAX_NUM_RETRIES_SYNC = 3

PATH_WEB_DOCS_S3 = f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_linededup_finalcleaning_setimgurlsdedup_optoutrmv/{IDX_JOB}"
PATH_WEB_DOCS_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_docs")

NUM_PROC = 20

PATH_SAVE_DISK_WEB_DOCS_FINAL_PROCESSING = os.path.join(PATH_SAVE_DISK_TMP_FILES, "web_docs_finalprocessing")
PATH_SAVE_S3_WEB_DOCS_FINAL_PROCESSING = f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_linededup_finalcleaning_setimgurlsdedup_optoutrmv_finalprocessing/{IDX_JOB}"

PATH_SAVE_DISK_WEB_DOCS_FINAL_PROCESSING_IMAGES_REPLACED_BY_URLS = os.path.join(
    PATH_SAVE_DISK_TMP_FILES, "web_docs_finalprocessing_replaceimgbyurl"
)
PATH_SAVE_S3_WEB_DOCS_FINAL_PROCESSING_IMAGES_REPLACED_BY_URLS = f"s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_linededup_finalcleaning_setimgurlsdedup_optoutrmv_finalprocessing_replaceimgbyurl/{IDX_JOB}"

PATH_SAVE_DISK_IMG_URLS_IN_FINAL_WEB_DOCS = os.path.join(PATH_SAVE_DISK_TMP_FILES, "img_urls.pickle")
PATH_SAVE_S3_IMG_URLS_IN_FINAL_WEB_DOCS = (
    f"s3://m4-datasets/webdocs/img_urls_in_final_web_docs_3/{IDX_JOB}/img_urls.pickle"
)


def remove_duplicated_images(texts, images, metadata):
    indices_to_remove = set()

    set_image_urls = set()
    for idx, meta in enumerate(metadata):
        if meta:
            url = meta["src"]
            if url not in set_image_urls:
                set_image_urls.add(url)
            else:
                indices_to_remove.add(idx)

    if indices_to_remove:
        texts = [el for ind, el in enumerate(texts) if ind not in indices_to_remove]
        images = [el for ind, el in enumerate(images) if ind not in indices_to_remove]
        metadata = [el for ind, el in enumerate(metadata) if ind not in indices_to_remove]
    return texts, images, metadata


def compute_spam_word_ratio(txt):
    words = FilteringFunctions.get_words_from_text(
        text=txt, lower_case=True, strip_words=True, strip_characters=SPECIAL_CHARACTERS
    )
    if not words:
        return 0
    spam_word_ratio = len([word for word in words if word in SPAM_WORDS]) / len(words)
    return spam_word_ratio


def remove_spam_paragraphs(texts, images, metadata):
    new_texts = []
    for text in texts:
        if text is None:
            new_texts.append(None)
        else:
            paragraphs = text.split("\n\n")
            new_paragraphs = [
                paragraph for paragraph in paragraphs if compute_spam_word_ratio(paragraph) < SPAM_WORD_RATIO_CUTOFF
            ]
            new_text = "\n\n".join(new_paragraphs)
            new_texts.append(new_text)
    return new_texts, images, metadata


def merge_consecutive_END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED(texts, images, metadata):
    new_texts = []
    for text in texts:
        if text is None:
            new_texts.append(None)
        else:
            paragraphs = text.split("\n\n")
            indices_to_remove = set()
            last_is_eos = False
            for ind, paragraph in enumerate(paragraphs):
                if last_is_eos:
                    if paragraph.strip() == "END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED":
                        indices_to_remove.add(ind)
                    else:
                        last_is_eos = False
                else:
                    if paragraph.strip() == "END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED":
                        last_is_eos = True
            new_paragraphs = [el for ind, el in enumerate(paragraphs) if ind not in indices_to_remove]
            new_text = "\n\n".join(new_paragraphs)
            new_texts.append(new_text)
    return new_texts, images, metadata


def remove_end_END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED(texts, images, metadata):
    if len(texts) > 0:
        last_text = texts[-1]
        if last_text:
            paragraphs = last_text.split("\n\n")
            if (len(paragraphs) > 0) and (paragraphs[-1] == "END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED"):
                paragraphs = paragraphs[:-1]
            last_text = "\n\n".join(paragraphs)
            texts[-1] = last_text
    return texts, images, metadata


def final_cleaning_node_level(texts, images, metadata):
    new_texts = []
    new_images = []
    new_metadata = []

    previous_is_text = False
    for text, image, meta in zip(texts, images, metadata):
        if text is not None:
            assert image is None
            assert meta is None
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
    return new_texts, new_images, new_metadata


def func_map_final_processing_node_level(example):
    texts = example["texts"]
    images = example["images"]
    metadata = json.loads(example["metadata"])
    assert len(texts) == len(images) == len(metadata)

    new_texts, new_images, new_metadata = remove_duplicated_images(texts, images, metadata)
    new_texts, new_images, new_metadata = remove_spam_paragraphs(new_texts, new_images, new_metadata)
    new_texts, new_images, new_metadata = final_cleaning_node_level(new_texts, new_images, new_metadata)
    new_texts, new_images, new_metadata = merge_consecutive_END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED(
        new_texts, new_images, new_metadata
    )
    new_texts, new_images, new_metadata = remove_end_END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED(
        new_texts, new_images, new_metadata
    )
    new_texts, new_images, new_metadata = final_cleaning_node_level(new_texts, new_images, new_metadata)

    example["texts"] = new_texts
    example["images"] = new_images
    example["metadata"] = json.dumps(new_metadata)

    return example


def remove_texts_only_END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED(example):
    texts_example = example["texts"]
    texts = [txt for txt in texts_example if txt]
    return not all([txt.strip() == "END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED" for txt in texts])


def final_cleaning_doc_level(example):
    texts_example = example["texts"]
    texts = [txt for txt in texts_example if txt]
    images = [txt for txt in texts_example if not txt]
    if not texts or not images:
        return False
    return True


def func_filter_final_processing_doc_level(example):
    if not remove_texts_only_END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED(example):
        return False
    if not final_cleaning_doc_level(example):
        return False
    return True


def func_map_replace_images_by_urls(example):
    metadata = json.loads(example["metadata"])
    image_urls = [meta["src"] if meta else None for meta in metadata]
    example["images"] = image_urls
    return example


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

    logger.info("Starting doing the final processing")
    web_docs = web_docs.map(func_map_final_processing_node_level, num_proc=NUM_PROC)
    web_docs = web_docs.filter(func_filter_final_processing_doc_level, num_proc=NUM_PROC)
    logger.info("Finished doing the final processing")

    logger.info("Starting saving the web document dataset after the final processing")
    web_docs.save_to_disk(PATH_SAVE_DISK_WEB_DOCS_FINAL_PROCESSING, num_proc=NUM_PROC)

    command_sync_s3 = (
        f"aws s3 sync {PATH_SAVE_DISK_WEB_DOCS_FINAL_PROCESSING} {PATH_SAVE_S3_WEB_DOCS_FINAL_PROCESSING}"
    )
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)
    logger.info("Finished saving the web document dataset after the final processing")

    logger.info("Starting replacing the images by their URLs")
    new_features = deepcopy(web_docs.features)
    new_features["images"] = datasets.Sequence(datasets.Value("string"))
    web_docs = web_docs.map(func_map_replace_images_by_urls, features=new_features, num_proc=NUM_PROC)
    logger.info("Finished replacing the images by their URLs")

    logger.info(
        "Starting saving the web document dataset after the final processing and the replacement of images to URLs"
    )
    web_docs.save_to_disk(PATH_SAVE_DISK_WEB_DOCS_FINAL_PROCESSING_IMAGES_REPLACED_BY_URLS, num_proc=NUM_PROC)

    command_sync_s3 = (
        "aws s3 sync"
        f" {PATH_SAVE_DISK_WEB_DOCS_FINAL_PROCESSING_IMAGES_REPLACED_BY_URLS} {PATH_SAVE_S3_WEB_DOCS_FINAL_PROCESSING_IMAGES_REPLACED_BY_URLS}"
    )
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)
    logger.info(
        "Finished saving the web document dataset after the final processing and the replacement of images to URLs"
    )

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

    logger.info(f"Number of documents in the web document dataset after the final processing: {web_docs.num_rows}")
    logger.info(
        "Number of images (with duplicates) in the web document dataset after the final processing:"
        f" {sum(list(img_urls.values()))}"
    )

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")

import argparse
import json
import logging
import os
import pickle

from datasets import load_from_disk
from PIL import Image, ImageFile


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(description="Remove the images that are too duplicated.")
    parser.add_argument(
        "idx_job",
        type=int,
        help="Index of the job (between 0 and 199).",
    )
    parser.add_argument(
        "--path_web_document_dataset_filtered",
        type=str,
        default="s3://m4-datasets/webdocs/web_document_dataset_filtered/",
        help="Path of the web document dataset filtered.",
    )
    parser.add_argument(
        "--path_tot_image_urls_in_web_document_dataset_filtered_too_duplicated",
        type=str,
        default="s3://m4-datasets/webdocs/tot_image_urls_in_web_document_dataset_filtered_too_duplicated.pickle",
        help="Path of the file containing the image urls to remove.",
    )
    parser.add_argument(
        "--path_save_web_document_dataset_filtered_imgurldedup",
        type=str,
        default="s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup/",
        help="Path to save the web document dataset filtered with the deduplication of image urls.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=48,
        help="Number of processes to use for the multiprocessing.",
    )
    args = parser.parse_args()
    return args


class ImageURLDeduplication:
    def __init__(self, path_image_urls_to_remove):
        self.path_image_urls_to_remove = path_image_urls_to_remove
        with open(path_image_urls_to_remove, "rb") as f:
            self.image_urls_to_remove = set(pickle.load(f))

    def __call__(self, web_document):
        metadata = json.loads(web_document["metadata"])

        indices_to_remove = set(
            [
                ind
                for ind, meta in enumerate(metadata)
                if (meta is not None) and (meta["src"] in self.image_urls_to_remove)
            ]
        )

        if indices_to_remove:
            web_document["texts"] = [
                el for ind, el in enumerate(web_document["texts"]) if ind not in indices_to_remove
            ]
            web_document["images"] = [
                el for ind, el in enumerate(web_document["images"]) if ind not in indices_to_remove
            ]
            web_document["metadata"] = json.dumps(
                [el for ind, el in enumerate(metadata) if ind not in indices_to_remove]
            )

        return web_document

    def __reduce__(self):
        return self.__class__, (self.path_image_urls_to_remove,)


if __name__ == "__main__":
    args = get_args()

    path_save_disk_tmp_files = f"/scratch/storage_hugo_{args.idx_job}/"
    if os.path.exists(path_save_disk_tmp_files):
        os.system(f"rm -r {path_save_disk_tmp_files}")
    os.system(f"mkdir {path_save_disk_tmp_files}")

    logger.info("Starting loading the web document dataset filtered and the set of image urls to remove")
    path_sync_s3 = os.path.join(args.path_web_document_dataset_filtered, str(args.idx_job))
    path_save_disk_web_document_dataset_filtered = os.path.join(
        path_save_disk_tmp_files, "web_document_dataset_filtered"
    )
    os.system(f"mkdir {path_save_disk_web_document_dataset_filtered}")
    command_sync_s3 = f"aws s3 sync {path_sync_s3} {path_save_disk_web_document_dataset_filtered}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    web_document_dataset_filtered = load_from_disk(path_save_disk_web_document_dataset_filtered)

    path_sync_s3 = args.path_tot_image_urls_in_web_document_dataset_filtered_too_duplicated
    path_save_disk_image_urls_to_remove = os.path.join(path_save_disk_tmp_files, "image_urls_to_remove.pickle")
    command_sync_s3 = f"aws s3 cp {path_sync_s3} {path_save_disk_image_urls_to_remove}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    logger.info("Finished loading the web document dataset filtered and the set of image urls to remove")

    logger.info("Starting removing the images too deduplicated")
    image_url_deduplication = ImageURLDeduplication(path_image_urls_to_remove=path_save_disk_image_urls_to_remove)
    web_document_dataset_filtered_imgurldedup = web_document_dataset_filtered.map(
        image_url_deduplication,
        num_proc=args.num_proc,
    )
    logger.info("Finished removing the images too deduplicated")

    logger.info("Starting saving the web document dataset filtered with the deduplication of image urls")
    path_save_disk_web_document_dataset_filtered_imgurldedup = os.path.join(
        path_save_disk_tmp_files, "web_document_dataset_filtered_imgurldedup"
    )
    web_document_dataset_filtered_imgurldedup.save_to_disk(
        path_save_disk_web_document_dataset_filtered_imgurldedup, num_proc=args.num_proc
    )

    path_sync_s3 = os.path.join(args.path_save_web_document_dataset_filtered_imgurldedup, str(args.idx_job))
    command_sync_s3 = f"aws s3 sync {path_save_disk_web_document_dataset_filtered_imgurldedup} {path_sync_s3}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    logger.info("Finished saving the web document dataset filtered with the deduplication of image urls")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -r {path_save_disk_tmp_files}")
    logger.info("Finished deleting the tmp files")

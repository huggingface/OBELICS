# srun --pty --ntasks=1 --cpus-per-task=96 bash -i
# conda activate /fsx/m4/conda/shared-m4-2023-03-10


import os
import pickle
from collections import Counter

from tqdm import tqdm


PATH_S3_IMAGE_URLS_IN_WEBDOCS = "s3://m4-datasets/webdocs/image_urls_in_web_document_dataset_filtered/"
NUM_SHARDS = 200
THRESHOLD_TOO_DUPLICATED = 10


if __name__ == "__main__":
    path_save_disk_image_urls_in_webdocs = "/scratch/image_urls_in_webdocs"
    command_sync_s3 = f"aws s3 sync {PATH_S3_IMAGE_URLS_IN_WEBDOCS} {path_save_disk_image_urls_in_webdocs}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    all_counters = []
    for idx_shard in tqdm(range(NUM_SHARDS)):
        with open(
            os.path.join(
                path_save_disk_image_urls_in_webdocs,
                str(idx_shard),
                "image_urls_in_web_document_dataset_filtered.pickle",
            ),
            "rb",
        ) as f:
            all_counters.append(pickle.load(f))

    tot_counter = Counter()
    for counter in tqdm(all_counters):
        tot_counter.update(counter)

    with open("/scratch/tot_image_urls_in_web_document_dataset_filtered.pickle", "wb") as f:
        pickle.dump(tot_counter, f, pickle.HIGHEST_PROTOCOL)

    command_sync_s3 = (
        "aws s3 cp /scratch/tot_image_urls_in_web_document_dataset_filtered.pickle"
        " s3://m4-datasets/webdocs/tot_image_urls_in_web_document_dataset_filtered.pickle"
    )
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    tot_image_urls_in_web_document_dataset_filtered_too_duplicated = [
        k for k, v in tot_counter.items() if v > THRESHOLD_TOO_DUPLICATED
    ]

    with open("/scratch/tot_image_urls_in_web_document_dataset_filtered_too_duplicated.pickle", "wb") as f:
        pickle.dump(tot_counter, f, pickle.HIGHEST_PROTOCOL)

    command_sync_s3 = (
        "aws s3 cp /scratch/tot_image_urls_in_web_document_dataset_filtered_too_duplicated.pickle"
        " s3://m4-datasets/webdocs/tot_image_urls_in_web_document_dataset_filtered_too_duplicated.pickle"
    )
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

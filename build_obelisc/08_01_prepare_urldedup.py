"""
srun --pty --cpus-per-task=96 bash -i
conda activate /fsx/m4/conda/shared-m4-2023-03-10
"""


import json
import os
from collections import Counter

from datasets import load_from_disk
from tqdm import tqdm


NUM_SHARDS = 200


PATH_WEB_DOCS_S3 = "s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_texts_only/"
PATH_WEB_DOCS_LOCAL = "/scratch/web_docs_texts_only/"

PATH_SAVE_DISK_DUP_URLS = "/scratch/dup_urls.json"
PATH_SAVE_S3_DUP_URLS = "s3://m4-datasets/webdocs/dup_urls.json"


def unroll_list(list_):
    list_ = [sub_el for el in list_ for sub_el in el]
    return list_


if __name__ == "__main__":
    command_sync_s3 = f"aws s3 sync {PATH_WEB_DOCS_S3} {PATH_WEB_DOCS_LOCAL}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    ds_shards = [
        load_from_disk(os.path.join(PATH_WEB_DOCS_LOCAL, str(idx_shard))) for idx_shard in tqdm(range(NUM_SHARDS))
    ]
    all_urls = []
    for idx_shard in tqdm(range(NUM_SHARDS)):
        all_urls.extend([json.loads(meta)["url"] for meta in ds_shards[idx_shard]["general_metadata"]])
    print(f"Total number of documents: {len(all_urls)}")
    # Total number of documents: 361_209_568
    del all_urls
    del ds_shards

    dup_urls = Counter(all_urls)
    dup_urls = {k: v for k, v in dup_urls.items() if v > 1}
    sum_dup_urls = sum(list(dup_urls.values()))
    print(f"{len(dup_urls)} URLs appear at least twice, for a total of {sum_dup_urls} documents")
    # 80_236_663 URLs appear at least twice, for a total of 196_108_071 documents
    print(f"We can then remove {sum_dup_urls - len(dup_urls)} documents")
    # We can then remove 115_871_408 documents

    # This block of lines was done in multiprocessing, one shard per job (200 jobs), and 24 cpus per job
    # The result is one dup_urls_to_warcfilename per job,
    # saved at /fsx/hugo/trash/dup_urls_to_warcfilename/dup_urls_to_warcfilename_{idx_shard}.json,
    # and we will have to merge them after
    dup_urls_to_warcfilename = {}
    for idx_shard in tqdm(range(NUM_SHARDS)):
        ds_shard = load_from_disk(os.path.join(PATH_WEB_DOCS_LOCAL, str(idx_shard)), keep_in_memory=True)
        urls_shard = [json.loads(meta)["url"] for meta in ds_shard["general_metadata"]]
        for idx, url in enumerate(urls_shard):
            if url in dup_urls:
                dup_urls_to_warcfilename[url] = dup_urls_to_warcfilename.get(url, []) + [
                    json.loads(ds_shard[idx]["general_metadata"])["warc_filename"]
                ]

    # We start from the result of the last operation which was done in multiple jobs
    # We need to merge the different dup_urls_to_warcfilename since we created one per job
    all_dup_urls_to_warcfilename = []
    for idx_shard in tqdm(range(NUM_SHARDS)):
        with open(f"/fsx/hugo/trash/dup_urls_to_warcfilename/dup_urls_to_warcfilename_{idx_shard}.json") as f:
            all_dup_urls_to_warcfilename.append(json.load(f))
    dup_urls_to_warcfilename = {
        url: unroll_list(
            [
                all_dup_urls_to_warcfilename[idx_shard][url]
                for idx_shard in range(NUM_SHARDS)
                if url in all_dup_urls_to_warcfilename[idx_shard]
            ]
        )
        for url in tqdm(dup_urls)
    }
    print(len(dup_urls_to_warcfilename))  # Check
    print(sum([len(dup_urls_to_warcfilename[url]) for url in dup_urls_to_warcfilename]))  # Check

    # We only keep the most recent document in a group with a common url, all the others will be deleted
    dup_urls_to_warcfilename = {
        url: sorted(warc_filenames)[-1] for url, warc_filenames in tqdm(dup_urls_to_warcfilename.items())
    }

    with open(PATH_SAVE_DISK_DUP_URLS, "w") as f:
        json.dump(dup_urls_to_warcfilename, f)
    command_sync_s3 = f"aws s3 cp {PATH_SAVE_DISK_DUP_URLS} {PATH_SAVE_S3_DUP_URLS}"
    os.system(command_sync_s3)

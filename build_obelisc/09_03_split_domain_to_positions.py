import json
import os
import random

from tqdm import tqdm


random.seed(42)

NUM_SHARDS = 200

PATH_LINE_DEDUP_DOMAIN_TO_POSITIONS_S3 = "s3://m4-datasets/webdocs/line_dedup_domain_to_positions.json"
PATH_LINE_DEDUP_DOMAIN_TO_POSITIONS_LOCAL = "/scratch/line_dedup_domain_to_positions.json"

PATH_SAVE_S3_LINE_DEDUP_DOMAIN_TO_POSITIONS_SHARDED = (
    "s3://m4-datasets/webdocs/line_dedup_domain_to_positions_sharded/"
)


if __name__ == "__main__":
    command_sync_s3 = f"aws s3 cp {PATH_LINE_DEDUP_DOMAIN_TO_POSITIONS_S3} {PATH_LINE_DEDUP_DOMAIN_TO_POSITIONS_LOCAL}"
    os.system(command_sync_s3)

    with open(PATH_LINE_DEDUP_DOMAIN_TO_POSITIONS_LOCAL) as f:
        domain_to_positions = json.load(f)

    keys = list(domain_to_positions.keys())
    random.shuffle(keys)

    sublist_size = len(keys) // NUM_SHARDS + 1
    keys_per_shard = [set(keys[i : i + sublist_size]) for i in range(0, len(keys), sublist_size)]

    domain_to_positions_shard = []

    for idx_shard in tqdm(range(NUM_SHARDS)):
        domain_to_positions_shard.append(
            {k: v for k, v in domain_to_positions.items() if k in keys_per_shard[idx_shard]}
        )

        with open(f"/scratch/line_dedup_domain_to_positions_{idx_shard}.json", "w") as f:
            json.dump(domain_to_positions_shard[idx_shard], f)

    for idx_shard in tqdm(range(NUM_SHARDS)):
        path_disk = f"/scratch/line_dedup_domain_to_positions_{idx_shard}.json"
        path_s3 = os.path.join(
            PATH_SAVE_S3_LINE_DEDUP_DOMAIN_TO_POSITIONS_SHARDED, str(idx_shard), "line_dedup_domain_to_positions.json"
        )
        command_sync_s3 = f"aws s3 cp {path_disk} {path_s3}"
        os.system(command_sync_s3)

import json
import logging
import os
from glob import glob
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


class WebDocumentLineDeduplication:
    def __init__(
        self,
        path_sharded_dataset,
        path_save_domain_to_positions,
        path_save_domain_to_duplicated_texts,
        id_shard_to_line_deduplicate,
        num_proc,
        path_save_line_deduplicated_sharded_dataset,
    ):
        self.path_sharded_dataset = path_sharded_dataset
        self.path_save_domain_to_positions = path_save_domain_to_positions
        self.path_save_domain_to_duplicated_texts = path_save_domain_to_duplicated_texts
        self.id_shard_to_line_deduplicate = id_shard_to_line_deduplicate
        self.num_proc = num_proc
        self.path_save_line_deduplicated_sharded_dataset = path_save_line_deduplicated_sharded_dataset

    def get_paths_subdatasets(self):
        self.paths_subdatasets = glob(f"{self.path_sharded_dataset}/*/")

    def remove_empty_els_in_list(self, list_):
        return [el for el in list_ if el is not None]

    def get_domain_to_positions(self):
        logger.info(
            "Starting creating the dictionary to go from a domain name to positions in the web document dataset"
        )
        self.domain_to_positions = {}

        for path_subdatasets in tqdm(self.paths_subdatasets):
            sub_ds = load_from_disk(path_subdatasets)
            sub_ds = sub_ds.remove_columns([c_n for c_n in sub_ds.column_names if c_n != "metadata"])
            metadata_sub_ds = sub_ds["metadata"]
            metadata_sub_ds = [json.loads(meta) for meta in metadata_sub_ds]
            metadata_sub_ds = [self.remove_empty_els_in_list(meta)[0] for meta in metadata_sub_ds]
            domains = [urlparse(meta["document_url"]).netloc for meta in metadata_sub_ds]

            new_domain_to_pos = {}
            for idx, domain in enumerate(domains):
                new_domain_to_pos[domain] = new_domain_to_pos.get(domain, []) + [idx]
            for domain in new_domain_to_pos:
                if domain not in self.domain_to_positions:
                    self.domain_to_positions[domain] = {}
                self.domain_to_positions[domain][path_subdatasets] = new_domain_to_pos[domain]

        with open(self.path_save_domain_to_positions, "w") as f:
            json.dump(self.domain_to_positions, f)
        logger.info(
            "Finished creating and saving the dictionary to go from a domain name to positions in the web document"
            " dataset"
        )

    def get_domain_to_duplicated_texts(self):
        logger.info("Starting finding the duplicated texts for each domain")
        with open(self.path_save_domain_to_positions) as f:
            self.domain_to_positions = json.load(f)

        self.domain_to_duplicated_texts = {}

        for domain in tqdm(self.domain_to_positions):
            duplicated_texts = {}
            positions = self.domain_to_positions[domain]

            for path_subdatasets in positions:
                sub_ds = load_from_disk(path_subdatasets)
                sub_ds = sub_ds.remove_columns([c_n for c_n in sub_ds.column_names if c_n != "texts"])
                idx_pos = positions[path_subdatasets]

                for idx in idx_pos:
                    tot_texts = self.remove_empty_els_in_list(sub_ds[idx]["texts"])
                    tot_texts = [text.split("\n\n") for text in tot_texts]
                    tot_texts = [paragraph for text in tot_texts for paragraph in text]
                    for text in tot_texts:
                        duplicated_texts[text] = duplicated_texts.get(text, 0) + 1

            duplicated_texts = {k: v for k, v in duplicated_texts.items() if v > 1}
            self.domain_to_duplicated_texts[domain] = duplicated_texts

        with open(self.path_save_domain_to_duplicated_texts, "w") as f:
            json.dump(self.domain_to_duplicated_texts, f)
        logger.info("Finished finding and saving the duplicated texts for each domain")

    def line_deduplicate_web_documents(self):
        logger.info(
            f"Starting line deduplicating the web document dataset for shard {self.id_shard_to_line_deduplicate}"
        )
        with open(self.path_save_domain_to_duplicated_texts) as f:
            self.domain_to_duplicated_texts = json.load(f)

        def func_mac_line_deduplicate_web_documents(example):
            metadata = json.loads(example["metadata"])
            domain = urlparse(self.remove_empty_els_in_list(metadata)[0]["document_url"]).netloc

            indices_to_remove = set()
            for idx in range(len(example["texts"])):
                if example["texts"][idx] is not None:
                    example["texts"][idx] = "\n\n".join(
                        [
                            paragraph
                            for paragraph in example["texts"][idx].split("\n\n")
                            if paragraph not in self.domain_to_duplicated_texts[domain]
                        ]
                    )
                    if not example["texts"][idx]:
                        indices_to_remove.add(idx)

            if indices_to_remove:
                example["texts"] = [el for ind, el in enumerate(example["texts"]) if ind not in indices_to_remove]
                example["images"] = [el for ind, el in enumerate(example["images"]) if ind not in indices_to_remove]
                example["metadata"] = json.dumps(
                    [el for ind, el in enumerate(metadata) if ind not in indices_to_remove]
                )

            return example

        os.system(f"mkdir -p {self.path_save_line_deduplicated_sharded_dataset}")

        path_subdataset = os.path.join(self.path_sharded_dataset, f"shard_{self.id_shard_to_line_deduplicate}")
        sub_ds = load_from_disk(path_subdataset)
        sub_ds_line_deduplicated = sub_ds.map(func_mac_line_deduplicate_web_documents, num_proc=self.num_proc)
        name_shard = os.path.basename(os.path.normpath(path_subdataset))
        sub_ds_line_deduplicated.save_to_disk(
            os.path.join(self.path_save_line_deduplicated_sharded_dataset, name_shard)
        )

        logger.info(
            f"Finished line deduplicating the web document dataset for shard {self.id_shard_to_line_deduplicate}"
        )

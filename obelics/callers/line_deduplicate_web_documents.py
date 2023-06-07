import argparse

from PIL import Image

from obelics.processors import WebDocumentLineDeduplication


# Useful to avoid DecompressionBombError
Image.MAX_IMAGE_PIXELS = None


def get_args():
    parser = argparse.ArgumentParser(description="Line deduplicate web documents.")
    parser.add_argument(
        "--path_sharded_dataset",
        type=str,
        default="/gpfsscratch/rech/cnw/commun/local_datasets/web_document_dataset_45M_sharded_filtered_2/train",
        help="Path to the folder containing the shards of the web document dataset.",
    )
    parser.add_argument(
        "--path_save_domain_to_positions",
        type=str,
        default="/gpfswork/rech/cnw/urd43gx/line_dedup/domain_to_positions.json",
        help=(
            "Path of the file to save the dictionary to go from a domain name to positions in the web document"
            " dataset."
        ),
    )
    parser.add_argument(
        "--path_save_domain_to_duplicated_texts",
        type=str,
        default="/gpfswork/rech/cnw/urd43gx/line_dedup/domain_to_duplicated_texts.json",
        help="Path of the file to save the dictionary containing the deduplicated texts for each domain.",
    )
    parser.add_argument(
        "--id_shard_to_line_deduplicate",
        type=int,
        default=2,
        help="Id of the shard to perform line deduplication on.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes for the map operation in the line deduplication.",
    )
    parser.add_argument(
        "--path_save_line_deduplicated_sharded_dataset",
        type=str,
        default="/gpfsscratch/rech/cnw/commun/local_datasets/web_document_dataset_45M_sharded_filtered_2_line_deduplicated/train",
        help="Path to the folder to save the shards of the line deduplicated web document dataset.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    path_sharded_dataset = args.path_sharded_dataset
    path_save_domain_to_positions = args.path_save_domain_to_positions
    path_save_domain_to_duplicated_texts = args.path_save_domain_to_duplicated_texts
    id_shard_to_line_deduplicate = args.id_shard_to_line_deduplicate
    num_proc = args.num_proc
    path_save_line_deduplicated_sharded_dataset = args.path_save_line_deduplicated_sharded_dataset

    web_document_line_deduplication = WebDocumentLineDeduplication(
        path_sharded_dataset=path_sharded_dataset,
        path_save_domain_to_positions=path_save_domain_to_positions,
        path_save_domain_to_duplicated_texts=path_save_domain_to_duplicated_texts,
        id_shard_to_line_deduplicate=id_shard_to_line_deduplicate,
        num_proc=num_proc,
        path_save_line_deduplicated_sharded_dataset=path_save_line_deduplicated_sharded_dataset,
    )

    web_document_line_deduplication.get_paths_subdatasets()

    # web_document_line_deduplication.get_domain_to_positions()

    # web_document_line_deduplication.get_domain_to_duplicated_texts()

    web_document_line_deduplication.line_deduplicate_web_documents()

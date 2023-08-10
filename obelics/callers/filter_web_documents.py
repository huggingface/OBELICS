import argparse
import logging
from multiprocessing import cpu_count

import yaml
from datasets import load_from_disk
from PIL import Image

from obelics.processors import WebDocumentFilteringDocLevel, WebDocumentFilteringNodeLevel
from obelics.utils import (
    DIGITS_RE,
    FLAGGED_WORDS,
    NON_PRINTING_CHARACTERS_RE,
    PUNCTUATION,
    SPECIAL_CHARACTERS,
    STOPWORDS,
    UNICODE_PUNCTUATION,
)


# Useful to avoid DecompressionBombError
Image.MAX_IMAGE_PIXELS = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(description="Web document filtering.")
    parser.add_argument(
        "--path_web_document_dataset",
        type=str,
        default="./large_files/web_document_dataset_100",
        help="Path of the dataset containing the web documents.",
    )
    parser.add_argument(
        "--path_save_dir_web_document_dataset_filtered",
        type=str,
        default="./large_files/web_document_dataset_100_filtered",
        help="The directory to save the filtered web document dataset.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=cpu_count(),
        help="Number of processes to use for the multiprocessing.",
    )
    parser.add_argument(
        "--path_config_filter_web_documents",
        type=str,
        default="./obelics/configs/config_filter_web_documents.yaml",
        help="The path of the config file containing the filtering parameters.",
    )
    parser.add_argument(
        "--path_common_words",
        type=str,
        default="./large_files/common_words.json",  # Find it at https://drive.google.com/file/d/1TeydSroOOmlEuxIcwgsJQ2YF4kPJR6N4/view?usp=sharing
        help="The path of the dictionary containing the common words.",
    )
    parser.add_argument(
        "--path_lang_id_model",
        type=str,
        default="./large_files/lid.176.bin",  # Find it at https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
        help="The path of the lang id model (FastText).",
    )
    parser.add_argument(
        "--path_sentencepiece_model",
        type=str,
        default="./large_files/en.sp.model",  # Find it at https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/en.sp.model
        help="The path of the SentencePiece model.",
    )
    parser.add_argument(
        "--path_kenlm_model",
        type=str,
        default="./large_files/en.arpa.bin",  # Find it at https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/en.arpa.bin
        help="The path of the KenLM model.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    logger.info("Starting loading the web document dataset")
    web_document_dataset = load_from_disk(args.path_web_document_dataset)
    logger.info("Finished loading the web document dataset")

    with open(args.path_config_filter_web_documents) as f:
        filtering_params = yaml.load(f, Loader=yaml.FullLoader)

    web_document_filtering_node_level = WebDocumentFilteringNodeLevel(
        cond_check_format=filtering_params["cond_check_format"],
        valid_formats=filtering_params["valid_formats"],
        cond_check_size_image=filtering_params["cond_check_size_image"],
        original_width_min_cutoff=filtering_params["original_width_min_cutoff"],
        original_width_max_cutoff=filtering_params["original_width_max_cutoff"],
        original_height_min_cutoff=filtering_params["original_height_min_cutoff"],
        original_height_max_cutoff=filtering_params["original_height_max_cutoff"],
        rendered_width_min_cutoff=filtering_params["rendered_width_min_cutoff"],
        rendered_width_max_cutoff=filtering_params["rendered_width_max_cutoff"],
        rendered_height_min_cutoff=filtering_params["rendered_height_min_cutoff"],
        rendered_height_max_cutoff=filtering_params["rendered_height_max_cutoff"],
        aspect_ratio_max_cutoff=filtering_params["aspect_ratio_max_cutoff"],
        cond_remove_non_printing_characters=filtering_params["cond_remove_non_printing_characters"],
        non_printing_characters_re=NON_PRINTING_CHARACTERS_RE,
        cond_standardize_whitespace=filtering_params["cond_standardize_whitespace"],
        cond_check_number_words_node_level=filtering_params["cond_check_number_words_node_level"],
        strip_characters=SPECIAL_CHARACTERS,
        number_words_node_level_min_cutoff=filtering_params["number_words_node_level_min_cutoff"],
        number_words_node_level_max_cutoff=filtering_params["number_words_node_level_max_cutoff"],
        cond_check_character_repetition_ratio_node_level=filtering_params[
            "cond_check_character_repetition_ratio_node_level"
        ],
        character_repetition_length_node_level=filtering_params["character_repetition_length_node_level"],
        character_repetition_node_level_max_cutoff=filtering_params["character_repetition_node_level_max_cutoff"],
        cond_check_word_repetition_ratio_node_level=filtering_params["cond_check_word_repetition_ratio_node_level"],
        word_repetition_length_node_level=filtering_params["word_repetition_length_node_level"],
        word_repetition_node_level_max_cutoff=filtering_params["word_repetition_node_level_max_cutoff"],
        cond_check_special_character_ratio_node_level=filtering_params[
            "cond_check_special_character_ratio_node_level"
        ],
        special_character_ratio_node_level_max_cutoff=filtering_params[
            "special_character_ratio_node_level_max_cutoff"
        ],
        cond_check_stopword_ratio_node_level=filtering_params["cond_check_stopword_ratio_node_level"],
        stopwords=STOPWORDS,
        stopword_ratio_node_level_min_cutoff=filtering_params["stopword_ratio_node_level_min_cutoff"],
        cond_check_flagged_word_ratio_node_level=filtering_params["cond_check_flagged_word_ratio_node_level"],
        flagged_words=FLAGGED_WORDS,
        flagged_word_ratio_node_level_max_cutoff=filtering_params["flagged_word_ratio_node_level_max_cutoff"],
        cond_check_punctuation_ratio_node_level=filtering_params["cond_check_punctuation_ratio_node_level"],
        min_number_words_to_check_punctuation_ratio_node_level=filtering_params[
            "min_number_words_to_check_punctuation_ratio_node_level"
        ],
        punctuation=PUNCTUATION,
        punctuation_ratio_node_level_min_cutoff=filtering_params["punctuation_ratio_node_level_min_cutoff"],
        cond_check_common_word_ratio_node_level=filtering_params["cond_check_common_word_ratio_node_level"],
        path_common_words=args.path_common_words,
        common_word_ratio_node_level_min_cutoff=filtering_params["common_word_ratio_node_level_min_cutoff"],
        cond_check_lang_id_node_level=filtering_params["cond_check_lang_id_node_level"],
        path_lang_id_model=args.path_lang_id_model,
        lang_id_node_level_min_cutoff=filtering_params["lang_id_node_level_min_cutoff"],
        cond_check_perplexity_score_node_level=filtering_params["cond_check_perplexity_score_node_level"],
        digits_re=DIGITS_RE,
        unicode_punctuation=UNICODE_PUNCTUATION,
        path_sentencepiece_model=args.path_sentencepiece_model,
        path_kenlm_model=args.path_kenlm_model,
        perplexity_score_node_level_max_cutoff=filtering_params["perplexity_score_node_level_max_cutoff"],
    )

    logger.info("Starting filtering the web document dataset at node level")
    web_document_dataset_filtered = web_document_dataset.map(web_document_filtering_node_level, num_proc=args.num_proc)
    logger.info("Finished filtering the web document dataset at node level")

    web_document_filtering_doc_level = WebDocumentFilteringDocLevel(
        cond_check_number_images=filtering_params["cond_check_number_images"],
        number_images_min_cutoff=filtering_params["number_images_min_cutoff"],
        number_images_max_cutoff=filtering_params["number_images_max_cutoff"],
        cond_check_number_words_doc_level=filtering_params["cond_check_number_words_doc_level"],
        strip_characters=SPECIAL_CHARACTERS,
        number_words_doc_level_min_cutoff=filtering_params["number_words_doc_level_min_cutoff"],
        number_words_doc_level_max_cutoff=filtering_params["number_words_doc_level_max_cutoff"],
        cond_check_character_repetition_ratio_doc_level=filtering_params[
            "cond_check_character_repetition_ratio_doc_level"
        ],
        character_repetition_length_doc_level=filtering_params["character_repetition_length_doc_level"],
        character_repetition_doc_level_max_cutoff=filtering_params["character_repetition_doc_level_max_cutoff"],
        cond_check_word_repetition_ratio_doc_level=filtering_params["cond_check_word_repetition_ratio_doc_level"],
        word_repetition_length_doc_level=filtering_params["word_repetition_length_doc_level"],
        word_repetition_doc_level_max_cutoff=filtering_params["word_repetition_doc_level_max_cutoff"],
        cond_check_special_character_ratio_doc_level=filtering_params["cond_check_special_character_ratio_doc_level"],
        special_character_ratio_doc_level_max_cutoff=filtering_params["special_character_ratio_doc_level_max_cutoff"],
        cond_check_stopword_ratio_doc_level=filtering_params["cond_check_stopword_ratio_doc_level"],
        stopwords=STOPWORDS,
        stopword_ratio_doc_level_min_cutoff=filtering_params["stopword_ratio_doc_level_min_cutoff"],
        cond_check_flagged_word_ratio_doc_level=filtering_params["cond_check_flagged_word_ratio_doc_level"],
        flagged_words=FLAGGED_WORDS,
        flagged_word_ratio_doc_level_max_cutoff=filtering_params["flagged_word_ratio_doc_level_max_cutoff"],
        cond_check_punctuation_ratio_doc_level=filtering_params["cond_check_punctuation_ratio_doc_level"],
        punctuation=PUNCTUATION,
        punctuation_ratio_doc_level_min_cutoff=filtering_params["punctuation_ratio_doc_level_min_cutoff"],
        cond_check_common_word_ratio_doc_level=filtering_params["cond_check_common_word_ratio_doc_level"],
        path_common_words=args.path_common_words,
        common_word_ratio_doc_level_min_cutoff=filtering_params["common_word_ratio_doc_level_min_cutoff"],
        cond_check_lang_id_doc_level=filtering_params["cond_check_lang_id_doc_level"],
        path_lang_id_model=args.path_lang_id_model,
        lang_id_doc_level_min_cutoff=filtering_params["lang_id_doc_level_min_cutoff"],
        cond_check_perplexity_score_doc_level=filtering_params["cond_check_perplexity_score_doc_level"],
        non_printing_characters_re=NON_PRINTING_CHARACTERS_RE,
        digits_re=DIGITS_RE,
        unicode_punctuation=UNICODE_PUNCTUATION,
        path_sentencepiece_model=args.path_sentencepiece_model,
        path_kenlm_model=args.path_kenlm_model,
        perplexity_score_doc_level_max_cutoff=filtering_params["perplexity_score_doc_level_max_cutoff"],
    )

    logger.info("Starting filtering the web document dataset at doc level")
    web_document_dataset_filtered = web_document_dataset_filtered.filter(
        web_document_filtering_doc_level, num_proc=args.num_proc
    )
    logger.info("Finished filtering the web document dataset at doc level")

    logger.info("Starting saving the filtered web document dataset")
    web_document_dataset_filtered.save_to_disk(
        args.path_save_dir_web_document_dataset_filtered, num_proc=args.num_proc
    )
    logger.info("Finished saving the filtered web document dataset")

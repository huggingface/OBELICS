import json
import random
from datetime import timedelta
from time import time

import streamlit as st
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


Image.MAX_IMAGE_PIXELS = None


class Visualization:
    def __init__(
        self,
        path_web_documents_dataset,
        path_config_filter_web_documents,
        path_common_words,
        path_lang_id_model,
        path_sentencepiece_model,
        path_kenlm_model,
    ):
        self.path_web_documents_dataset = path_web_documents_dataset
        with open(path_config_filter_web_documents) as f:
            self.filtering_params = yaml.load(f, Loader=yaml.FullLoader)
        self.path_common_words = path_common_words
        self.path_lang_id_model = path_lang_id_model
        self.path_sentencepiece_model = path_sentencepiece_model
        self.path_kenlm_model = path_kenlm_model

    def visualization(self):
        self.set_title()
        self.load_dataset()
        self.filtering()
        self.select_mode()
        self.choose_document()
        self.display_document()

    def set_title(self):
        st.title("Visualization of web documents and filtering")

    def load_dataset(self):
        st.header("Select the size of the dataset")

        self.full_dataset = load_from_disk(self.path_web_documents_dataset)

        # Useful the first time we load the full dataset to add a column
        # indicating the original IDs of the documents
        # self.full_dataset = self.full_dataset.add_column("original_idx", [i for i in range(len(self.full_dataset))])
        # self.full_dataset.save_to_disk(self.path_web_documents_dataset)

        opt_sizes = ["100", "300", "1000", "3000", "10000"]
        size_dataset = st.selectbox(
            "Select the size of the dataset",
            options=opt_sizes,
        )

        for opt_size in opt_sizes:
            if size_dataset == opt_size:
                self.full_dataset = self.full_dataset.select(
                    [_ for _ in range(min(int(opt_size), self.full_dataset.num_rows))]
                )
                if "retained_web_document_dataset" not in st.session_state:
                    st.session_state.retained_web_document_dataset = None
                if "discarded_web_document_dataset" not in st.session_state:
                    st.session_state.discarded_web_document_dataset = None

    def filtering(self):
        st.header("Filtering")

        st.subheader("Filtering at node level")

        self.cond_check_format = st.checkbox(
            "Remove images not in valid formats", value=self.filtering_params["cond_check_format"]
        )
        self.valid_formats = st.multiselect(
            "Valid formats",
            options=list(self.filtering_params["valid_formats"]),
            default=self.filtering_params["valid_formats"],
        )

        st.write("-----")

        self.cond_check_size_image = st.checkbox(
            "Remove images not in valid sizes", value=self.filtering_params["cond_check_size_image"]
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            self.original_width_min_cutoff = st.number_input(
                "Minimum original width",
                min_value=1,
                max_value=None,
                value=self.filtering_params["original_width_min_cutoff"],
                step=1,
            )
            self.rendered_width_min_cutoff = st.number_input(
                "Minimum rendered width",
                min_value=1,
                max_value=None,
                value=self.filtering_params["rendered_width_min_cutoff"],
                step=1,
            )
        with col2:
            self.original_width_max_cutoff = st.number_input(
                "Maximum original width",
                min_value=1,
                max_value=None,
                value=self.filtering_params["original_width_max_cutoff"],
                step=1,
            )
            self.rendered_width_max_cutoff = st.number_input(
                "Maximum rendered width",
                min_value=1,
                max_value=None,
                value=self.filtering_params["rendered_width_max_cutoff"],
                step=1,
            )
        with col3:
            self.original_height_min_cutoff = st.number_input(
                "Minimum original height",
                min_value=1,
                max_value=None,
                value=self.filtering_params["original_height_min_cutoff"],
                step=1,
            )
            self.rendered_height_min_cutoff = st.number_input(
                "Minimum rendered height",
                min_value=1,
                max_value=None,
                value=self.filtering_params["rendered_height_min_cutoff"],
                step=1,
            )
        with col4:
            self.original_height_max_cutoff = st.number_input(
                "Maximum original height",
                min_value=1,
                max_value=None,
                value=self.filtering_params["original_height_max_cutoff"],
                step=1,
            )
            self.rendered_height_max_cutoff = st.number_input(
                "Maximum rendered height",
                min_value=1,
                max_value=None,
                value=self.filtering_params["rendered_height_max_cutoff"],
                step=1,
            )
        self.aspect_ratio_max_cutoff = st.number_input(
            "Maximum aspect ratio",
            min_value=1.0,
            max_value=None,
            value=float(self.filtering_params["aspect_ratio_max_cutoff"]),
            step=0.5,
        )

        st.write("-----")

        self.cond_check_number_words_node_level = st.checkbox(
            "Remove paragraphs not having a valid number of words",
            value=self.filtering_params["cond_check_number_words_node_level"],
        )
        col1, col2 = st.columns(2)
        with col1:
            self.number_words_node_level_min_cutoff = st.number_input(
                "Minimum number of words (paragraph level)",
                min_value=0,
                max_value=None,
                value=self.filtering_params["number_words_node_level_min_cutoff"],
                step=1,
            )
        with col2:
            self.number_words_node_level_max_cutoff = st.number_input(
                "Maximum number of words (paragraph level)",
                min_value=0,
                max_value=None,
                value=self.filtering_params["number_words_node_level_max_cutoff"],
                step=1,
            )

        st.write("-----")

        self.cond_check_character_repetition_ratio_node_level = st.checkbox(
            "Remove paragraphs with a too high character repetition ratio",
            value=self.filtering_params["cond_check_character_repetition_ratio_node_level"],
        )
        col1, col2 = st.columns(2)
        with col1:
            self.character_repetition_length_node_level = st.number_input(
                "Character repetition length (node level)",
                min_value=0,
                max_value=None,
                value=self.filtering_params["character_repetition_length_node_level"],
                step=1,
            )
        with col2:
            self.character_repetition_node_level_max_cutoff = st.number_input(
                "Maximum character repetition ratio (node level)",
                min_value=0.0,
                max_value=1.0,
                value=self.filtering_params["character_repetition_node_level_max_cutoff"],
                step=0.01,
            )

        st.write("-----")

        self.cond_check_word_repetition_ratio_node_level = st.checkbox(
            "Remove paragraphs with a too high word repetition ratio",
            value=self.filtering_params["cond_check_word_repetition_ratio_node_level"],
        )
        col1, col2 = st.columns(2)
        with col1:
            self.word_repetition_length_node_level = st.number_input(
                "Word repetition length (node level)",
                min_value=0,
                max_value=None,
                value=self.filtering_params["word_repetition_length_node_level"],
                step=1,
            )
        with col2:
            self.word_repetition_node_level_max_cutoff = st.number_input(
                "Maximum word repetition ratio (node level)",
                min_value=0.0,
                max_value=1.0,
                value=self.filtering_params["word_repetition_node_level_max_cutoff"],
                step=0.01,
            )

        st.write("-----")

        self.cond_check_special_character_ratio_node_level = st.checkbox(
            "Remove paragraphs with a too high special character ratio",
            value=self.filtering_params["cond_check_special_character_ratio_node_level"],
        )
        self.special_character_ratio_node_level_max_cutoff = st.number_input(
            "Maximum special character ratio (paragraph level)",
            min_value=0.0,
            max_value=1.0,
            value=self.filtering_params["special_character_ratio_node_level_max_cutoff"],
            step=0.01,
        )

        st.write("-----")

        self.cond_check_stopword_ratio_node_level = st.checkbox(
            "Remove paragraphs with a too low stop word ratio",
            value=self.filtering_params["cond_check_stopword_ratio_node_level"],
        )
        self.stopword_ratio_node_level_min_cutoff = st.number_input(
            "Minimum stop word ratio (paragraph level)",
            min_value=0.0,
            max_value=1.0,
            value=self.filtering_params["stopword_ratio_node_level_min_cutoff"],
            step=0.01,
        )

        st.write("-----")

        self.cond_check_flagged_word_ratio_node_level = st.checkbox(
            "Remove paragraphs with a too high flagged word ratio",
            value=self.filtering_params["cond_check_flagged_word_ratio_node_level"],
        )
        self.flagged_word_ratio_node_level_max_cutoff = st.number_input(
            "Maximum flagged word ratio (node level)",
            min_value=0.0,
            max_value=1.0,
            value=self.filtering_params["flagged_word_ratio_node_level_max_cutoff"],
            step=0.01,
        )

        st.write("-----")

        self.cond_check_punctuation_ratio_node_level = st.checkbox(
            "Remove paragraphs with a too low punctuation ratio",
            value=self.filtering_params["cond_check_punctuation_ratio_node_level"],
        )
        self.min_number_words_to_check_punctuation_ratio_node_level = st.number_input(
            "Minimum number of words to check punctuation ratio (node level)",
            min_value=0,
            max_value=None,
            value=self.filtering_params["min_number_words_to_check_punctuation_ratio_node_level"],
            step=1,
        )
        self.punctuation_ratio_node_level_min_cutoff = st.number_input(
            "Minimum punctuation ratio (node level)",
            min_value=0.0,
            max_value=1.0,
            value=self.filtering_params["punctuation_ratio_node_level_min_cutoff"],
            step=0.01,
        )

        st.write("-----")

        self.cond_check_common_word_ratio_node_level = st.checkbox(
            "Remove paragraphs with a too low common word ratio",
            value=self.filtering_params["cond_check_common_word_ratio_node_level"],
        )
        self.common_word_ratio_node_level_min_cutoff = st.number_input(
            "Minimum common word ratio (node level)",
            min_value=0.0,
            max_value=1.0,
            value=self.filtering_params["common_word_ratio_node_level_min_cutoff"],
            step=0.01,
        )

        st.write("-----")

        self.cond_check_lang_id_node_level = st.checkbox(
            "Remove paragraphs with a too low language identification confidence score",
            value=self.filtering_params["cond_check_lang_id_node_level"],
        )
        self.lang_id_node_level_min_cutoff = st.number_input(
            "Minimum language identification confidence score (node level)",
            min_value=0.0,
            max_value=1.0,
            value=self.filtering_params["lang_id_node_level_min_cutoff"],
            step=0.01,
        )

        st.write("-----")

        self.cond_check_perplexity_score_node_level = st.checkbox(
            "Remove paragraphs with a too high perplexity score",
            value=self.filtering_params["cond_check_perplexity_score_node_level"],
        )
        self.perplexity_score_node_level_max_cutoff = st.number_input(
            "Maximum perplexity score (node level)",
            min_value=0,
            max_value=None,
            value=self.filtering_params["perplexity_score_node_level_max_cutoff"],
            step=1,
        )

        st.write("-----")

        st.subheader("Filtering at document level")

        self.cond_check_number_images = st.checkbox(
            "Remove documents with too few images", value=self.filtering_params["cond_check_number_images"]
        )
        col1, col2 = st.columns(2)
        with col1:
            self.number_images_min_cutoff = st.number_input(
                "Minimum number of images",
                min_value=0,
                max_value=None,
                value=self.filtering_params["number_images_min_cutoff"],
                step=1,
            )
        with col2:
            self.number_images_max_cutoff = st.number_input(
                "Maximum number of images",
                min_value=0,
                max_value=None,
                value=self.filtering_params["number_images_max_cutoff"],
                step=1,
            )

        st.write("-----")

        self.cond_check_number_words_doc_level = st.checkbox(
            "Remove documents not having a valid number of words",
            value=self.filtering_params["cond_check_number_words_doc_level"],
        )
        col1, col2 = st.columns(2)
        with col1:
            self.number_words_doc_level_min_cutoff = st.number_input(
                "Minimum number of words (doc level)",
                min_value=0,
                max_value=None,
                value=self.filtering_params["number_words_doc_level_min_cutoff"],
                step=1,
            )
        with col2:
            self.number_words_doc_level_max_cutoff = st.number_input(
                "Maximum number of words (doc level)",
                min_value=0,
                max_value=None,
                value=self.filtering_params["number_words_doc_level_max_cutoff"],
                step=1,
            )

        st.write("-----")

        self.cond_check_character_repetition_ratio_doc_level = st.checkbox(
            "Remove documents with a too high character repetition ratio",
            value=self.filtering_params["cond_check_character_repetition_ratio_doc_level"],
        )
        col1, col2 = st.columns(2)
        with col1:
            self.character_repetition_length_doc_level = st.number_input(
                "Character repetition length (doc level)",
                min_value=0,
                max_value=None,
                value=self.filtering_params["character_repetition_length_doc_level"],
                step=1,
            )
        with col2:
            self.character_repetition_doc_level_max_cutoff = st.number_input(
                "Maximum character repetition ratio (doc level)",
                min_value=0.0,
                max_value=1.0,
                value=self.filtering_params["character_repetition_doc_level_max_cutoff"],
                step=0.01,
            )

        st.write("-----")

        self.cond_check_word_repetition_ratio_doc_level = st.checkbox(
            "Remove documents with a too high word repetition ratio",
            value=self.filtering_params["cond_check_word_repetition_ratio_doc_level"],
        )
        col1, col2 = st.columns(2)
        with col1:
            self.word_repetition_length_doc_level = st.number_input(
                "Word repetition length (doc level)",
                min_value=0,
                max_value=None,
                value=self.filtering_params["word_repetition_length_doc_level"],
                step=1,
            )
        with col2:
            self.word_repetition_doc_level_max_cutoff = st.number_input(
                "Maximum word repetition ratio (doc level)",
                min_value=0.0,
                max_value=1.0,
                value=self.filtering_params["word_repetition_doc_level_max_cutoff"],
                step=0.01,
            )

        st.write("-----")

        self.cond_check_special_character_ratio_doc_level = st.checkbox(
            "Remove documents with a too high special character ratio",
            value=self.filtering_params["cond_check_special_character_ratio_doc_level"],
        )
        self.special_character_ratio_doc_level_max_cutoff = st.number_input(
            "Maximum special character ratio (doc level)",
            min_value=0.0,
            max_value=1.0,
            value=self.filtering_params["special_character_ratio_doc_level_max_cutoff"],
            step=0.01,
        )

        st.write("-----")

        self.cond_check_stopword_ratio_doc_level = st.checkbox(
            "Remove documents with a too low stop word ratio",
            value=self.filtering_params["cond_check_stopword_ratio_doc_level"],
        )
        self.stopword_ratio_doc_level_min_cutoff = st.number_input(
            "Minimum stop word ratio (doc level)",
            min_value=0.0,
            max_value=1.0,
            value=self.filtering_params["stopword_ratio_doc_level_min_cutoff"],
            step=0.01,
        )

        st.write("-----")

        self.cond_check_flagged_word_ratio_doc_level = st.checkbox(
            "Remove documents with a too high flagged word ratio",
            value=self.filtering_params["cond_check_flagged_word_ratio_doc_level"],
        )
        self.flagged_word_ratio_doc_level_max_cutoff = st.number_input(
            "Maximum flagged word ratio (doc level)",
            min_value=0.0,
            max_value=1.0,
            value=self.filtering_params["flagged_word_ratio_doc_level_max_cutoff"],
            step=0.01,
        )

        st.write("-----")

        self.cond_check_punctuation_ratio_doc_level = st.checkbox(
            "Remove documents with a too low punctuation ratio",
            value=self.filtering_params["cond_check_punctuation_ratio_doc_level"],
        )
        self.punctuation_ratio_doc_level_min_cutoff = st.number_input(
            "Minimum punctuation ratio (doc level)",
            min_value=0.0,
            max_value=1.0,
            value=self.filtering_params["punctuation_ratio_doc_level_min_cutoff"],
            step=0.01,
        )

        st.write("-----")

        self.cond_check_common_word_ratio_doc_level = st.checkbox(
            "Remove documents with a too low common word ratio",
            value=self.filtering_params["cond_check_common_word_ratio_doc_level"],
        )
        self.common_word_ratio_doc_level_min_cutoff = st.number_input(
            "Minimum common word ratio (doc level)",
            min_value=0.0,
            max_value=1.0,
            value=self.filtering_params["common_word_ratio_doc_level_min_cutoff"],
            step=0.01,
        )

        st.write("-----")

        self.cond_check_lang_id_doc_level = st.checkbox(
            "Remove documents with a too low language identification confidence score",
            value=self.filtering_params["cond_check_lang_id_doc_level"],
        )
        self.lang_id_doc_level_min_cutoff = st.number_input(
            "Minimum language identification confidence score (doc level)",
            min_value=0.0,
            max_value=1.0,
            value=self.filtering_params["lang_id_doc_level_min_cutoff"],
            step=0.01,
        )

        st.write("-----")

        self.cond_check_perplexity_score_doc_level = st.checkbox(
            "Remove documents with a too high perplexity score",
            value=self.filtering_params["cond_check_perplexity_score_doc_level"],
        )
        self.perplexity_score_doc_level_max_cutoff = st.number_input(
            "Maximum perplexity score (doc level)",
            min_value=0,
            max_value=None,
            value=self.filtering_params["perplexity_score_doc_level_max_cutoff"],
            step=1,
        )

        st.write("-----")

        st.subheader("Perform filtering")
        button_filtering = st.button("Perform filtering 💥")
        if button_filtering:
            with st.spinner("Wait for it... 🤞"):
                start_time = time()

                web_document_filtering_node_level = WebDocumentFilteringNodeLevel(
                    cond_check_format=self.cond_check_format,
                    valid_formats=self.valid_formats,
                    cond_check_size_image=self.cond_check_size_image,
                    original_width_min_cutoff=self.original_width_min_cutoff,
                    original_width_max_cutoff=self.original_width_max_cutoff,
                    original_height_min_cutoff=self.original_height_min_cutoff,
                    original_height_max_cutoff=self.original_height_max_cutoff,
                    rendered_width_min_cutoff=self.rendered_width_min_cutoff,
                    rendered_width_max_cutoff=self.rendered_width_max_cutoff,
                    rendered_height_min_cutoff=self.rendered_height_min_cutoff,
                    rendered_height_max_cutoff=self.rendered_height_max_cutoff,
                    aspect_ratio_max_cutoff=self.aspect_ratio_max_cutoff,
                    cond_remove_non_printing_characters=self.filtering_params["cond_remove_non_printing_characters"],
                    non_printing_characters_re=NON_PRINTING_CHARACTERS_RE,
                    cond_standardize_whitespace=self.filtering_params["cond_standardize_whitespace"],
                    cond_check_number_words_node_level=self.cond_check_number_words_node_level,
                    strip_characters=SPECIAL_CHARACTERS,
                    number_words_node_level_min_cutoff=self.number_words_node_level_min_cutoff,
                    number_words_node_level_max_cutoff=self.number_words_node_level_max_cutoff,
                    cond_check_character_repetition_ratio_node_level=self.cond_check_character_repetition_ratio_node_level,
                    character_repetition_length_node_level=self.character_repetition_length_node_level,
                    character_repetition_node_level_max_cutoff=self.character_repetition_node_level_max_cutoff,
                    cond_check_word_repetition_ratio_node_level=self.cond_check_word_repetition_ratio_node_level,
                    word_repetition_length_node_level=self.word_repetition_length_node_level,
                    word_repetition_node_level_max_cutoff=self.word_repetition_node_level_max_cutoff,
                    cond_check_special_character_ratio_node_level=self.cond_check_special_character_ratio_node_level,
                    special_character_ratio_node_level_max_cutoff=self.special_character_ratio_node_level_max_cutoff,
                    cond_check_stopword_ratio_node_level=self.cond_check_stopword_ratio_node_level,
                    stopwords=STOPWORDS,
                    stopword_ratio_node_level_min_cutoff=self.stopword_ratio_node_level_min_cutoff,
                    cond_check_flagged_word_ratio_node_level=self.cond_check_flagged_word_ratio_node_level,
                    flagged_words=FLAGGED_WORDS,
                    flagged_word_ratio_node_level_max_cutoff=self.flagged_word_ratio_node_level_max_cutoff,
                    cond_check_punctuation_ratio_node_level=self.cond_check_punctuation_ratio_node_level,
                    min_number_words_to_check_punctuation_ratio_node_level=self.min_number_words_to_check_punctuation_ratio_node_level,
                    punctuation=PUNCTUATION,
                    punctuation_ratio_node_level_min_cutoff=self.punctuation_ratio_node_level_min_cutoff,
                    cond_check_common_word_ratio_node_level=self.cond_check_common_word_ratio_node_level,
                    path_common_words=path_common_words,
                    common_word_ratio_node_level_min_cutoff=self.common_word_ratio_node_level_min_cutoff,
                    cond_check_lang_id_node_level=self.cond_check_lang_id_node_level,
                    path_lang_id_model=self.path_lang_id_model,
                    lang_id_node_level_min_cutoff=self.lang_id_node_level_min_cutoff,
                    cond_check_perplexity_score_node_level=self.cond_check_perplexity_score_node_level,
                    digits_re=DIGITS_RE,
                    unicode_punctuation=UNICODE_PUNCTUATION,
                    path_sentencepiece_model=self.path_sentencepiece_model,
                    path_kenlm_model=self.path_kenlm_model,
                    perplexity_score_node_level_max_cutoff=self.perplexity_score_node_level_max_cutoff,
                )

                full_dataset_filtered_node_level = self.full_dataset.map(
                    web_document_filtering_node_level, load_from_cache_file=False, writer_batch_size=10000
                )

                web_document_filtering_doc_level = WebDocumentFilteringDocLevel(
                    cond_check_number_images=self.cond_check_number_images,
                    number_images_min_cutoff=self.number_images_min_cutoff,
                    number_images_max_cutoff=self.number_images_max_cutoff,
                    cond_check_number_words_doc_level=self.cond_check_number_words_doc_level,
                    strip_characters=SPECIAL_CHARACTERS,
                    number_words_doc_level_min_cutoff=self.number_words_doc_level_min_cutoff,
                    number_words_doc_level_max_cutoff=self.number_words_doc_level_max_cutoff,
                    cond_check_character_repetition_ratio_doc_level=self.cond_check_character_repetition_ratio_doc_level,
                    character_repetition_length_doc_level=self.character_repetition_length_doc_level,
                    character_repetition_doc_level_max_cutoff=self.character_repetition_doc_level_max_cutoff,
                    cond_check_word_repetition_ratio_doc_level=self.cond_check_word_repetition_ratio_doc_level,
                    word_repetition_length_doc_level=self.word_repetition_length_doc_level,
                    word_repetition_doc_level_max_cutoff=self.word_repetition_doc_level_max_cutoff,
                    cond_check_special_character_ratio_doc_level=self.cond_check_special_character_ratio_doc_level,
                    special_character_ratio_doc_level_max_cutoff=self.special_character_ratio_doc_level_max_cutoff,
                    cond_check_stopword_ratio_doc_level=self.cond_check_stopword_ratio_doc_level,
                    stopwords=STOPWORDS,
                    stopword_ratio_doc_level_min_cutoff=self.stopword_ratio_doc_level_min_cutoff,
                    cond_check_flagged_word_ratio_doc_level=self.cond_check_flagged_word_ratio_doc_level,
                    flagged_words=FLAGGED_WORDS,
                    flagged_word_ratio_doc_level_max_cutoff=self.flagged_word_ratio_doc_level_max_cutoff,
                    cond_check_punctuation_ratio_doc_level=self.cond_check_punctuation_ratio_doc_level,
                    punctuation=PUNCTUATION,
                    punctuation_ratio_doc_level_min_cutoff=self.punctuation_ratio_doc_level_min_cutoff,
                    cond_check_common_word_ratio_doc_level=self.cond_check_common_word_ratio_doc_level,
                    path_common_words=path_common_words,
                    common_word_ratio_doc_level_min_cutoff=self.common_word_ratio_doc_level_min_cutoff,
                    cond_check_lang_id_doc_level=self.cond_check_lang_id_doc_level,
                    path_lang_id_model=self.path_lang_id_model,
                    lang_id_doc_level_min_cutoff=self.lang_id_doc_level_min_cutoff,
                    cond_check_perplexity_score_doc_level=self.cond_check_perplexity_score_doc_level,
                    non_printing_characters_re=NON_PRINTING_CHARACTERS_RE,
                    digits_re=DIGITS_RE,
                    unicode_punctuation=UNICODE_PUNCTUATION,
                    path_sentencepiece_model=self.path_sentencepiece_model,
                    path_kenlm_model=self.path_kenlm_model,
                    perplexity_score_doc_level_max_cutoff=self.perplexity_score_doc_level_max_cutoff,
                )

                st.session_state.retained_web_document_dataset = full_dataset_filtered_node_level.filter(
                    web_document_filtering_doc_level, load_from_cache_file=False, writer_batch_size=10000
                )

                idx_retained_docs = set(st.session_state.retained_web_document_dataset["original_idx"])

                def keep_discarded_docs(web_document):
                    if web_document["original_idx"] not in idx_retained_docs:
                        return True
                    return False

                st.session_state.discarded_web_document_dataset = full_dataset_filtered_node_level.filter(
                    keep_discarded_docs, load_from_cache_file=False, writer_batch_size=10000
                )

            st.balloons()
            end_time = time()
            tot_time = round(end_time - start_time)
            st.success(f"Filtering done in {timedelta(seconds=tot_time)} (HH:MM:SS)!")

    def select_mode(self):
        st.header("Select a mode")
        options_mode = ["All original web documents", "Retained web documents", "Discarded web documents"]
        self.mode = st.selectbox(label="Select a mode", options=options_mode)
        if self.mode == options_mode[0]:
            self.dataset = self.full_dataset
        if self.mode == options_mode[1]:
            self.dataset = st.session_state.retained_web_document_dataset
        if self.mode == options_mode[2]:
            self.dataset = st.session_state.discarded_web_document_dataset

        if self.dataset is None:
            st.warning(
                "To display retained or discarded documents, please perform first the filtering pipeline in the box"
                " above"
            )

    def choose_document(self):
        if self.dataset:
            st.header("Choose a document")
            if st.button("Select a random document"):
                dct_idx = random.randint(a=0, b=self.dataset.num_rows - 1)
            else:
                dct_idx = 0
            idx = st.number_input(
                f"Select a document among the first {self.dataset.num_rows} ones",
                min_value=0,
                max_value=self.dataset.num_rows - 1,
                value=dct_idx,
                step=1,
                help=f"Index between 0 and {self.dataset.num_rows-1}",
            )
            self.current_doc = self.dataset[idx]
            original_idx = self.current_doc["original_idx"]
            st.markdown(f"Original Document Id: {original_idx}")
            self.current_doc_original = self.full_dataset[original_idx]
        else:
            self.current_doc = None

    def display_document(self):
        def display_single_doc(doc, subheader_name=None):
            if subheader_name:
                st.subheader(subheader_name)
            texts = doc["texts"]
            images = doc["images"]
            metadata = json.loads(doc["metadata"])
            for text, image, meta in zip(texts, images, metadata):
                if text:
                    st.text(f"{text}\n\n")
                elif image:
                    st.markdown(f"![img]({meta['src']})\n\n")

        if self.current_doc:
            st.header("Document")

            if self.mode == "All original web documents":
                display_single_doc(self.current_doc)

            else:
                display_original_doc = st.checkbox("Display the original document sibe by side", value=True)
                if not display_original_doc:
                    display_single_doc(doc=self.current_doc, subheader_name="Filtered document")
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        display_single_doc(doc=self.current_doc_original, subheader_name="Original document")
                    with col2:
                        display_single_doc(doc=self.current_doc, subheader_name="Filtered document")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    path_web_documents_dataset = (  # Use a web document format, like OBELICS
        "./large_files/web_document_dataset"
    )
    path_config_filter_web_documents = "./obelics/configs/config_filter_web_documents.yaml"
    path_common_words = "./large_files/common_words.json"  # Find it at https://drive.google.com/file/d/1TeydSroOOmlEuxIcwgsJQ2YF4kPJR6N4/view?usp=sharing
    path_lang_id_model = (  # Find it at https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
        "./large_files/lid.176.bin"
    )
    path_sentencepiece_model = (  # Find it at https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/en.sp.model
        "./large_files/en.sp.model"
    )
    path_kenlm_model = (  # Find it at https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/en.arpa.bin
        "./large_files/en.arpa.bin"
    )
    visualization = Visualization(
        path_web_documents_dataset=path_web_documents_dataset,
        path_config_filter_web_documents=path_config_filter_web_documents,
        path_common_words=path_common_words,
        path_lang_id_model=path_lang_id_model,
        path_sentencepiece_model=path_sentencepiece_model,
        path_kenlm_model=path_kenlm_model,
    )
    visualization.visualization()

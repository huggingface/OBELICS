import base64
import json
from io import BytesIO

import fasttext
import kenlm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sentencepiece
import streamlit as st
from datasets import load_from_disk

from obelisc.processors.web_document_filtering import FilteringFunctions
from obelisc.utils.filtering_utils import (
    DIGITS_RE,
    FLAGGED_WORDS,
    NON_PRINTING_CHARACTERS_RE,
    PUNCTUATION,
    SPECIAL_CHARACTERS,
    STOPWORDS,
    UNICODE_PUNCTUATION,
)


def non_empty_els_from_list(list_):
    return [el for el in list_ if el is not None]


def get_exs_and_stats(web_document_dataset, type_exs, funcs_compute_stats, text_node_level=True):
    exs = []
    for idx_row in range(web_document_dataset.num_rows):
        new_els = non_empty_els_from_list(web_document_dataset[idx_row][type_exs])
        if type_exs == "texts":
            new_els = non_empty_els_from_list(web_document_dataset[idx_row][type_exs])
            if not text_node_level:  # Text at document level
                exs.append("\n\n".join(new_els))
            else:  # Text at paragraph level
                new_els = [txt.split("\n\n") for txt in new_els]
                new_els = [paragraph for txt in new_els for paragraph in txt]
                exs.extend(new_els)
        else:
            exs.extend(new_els)

    all_stats = {}
    all_stats["exs"] = exs

    for stat_name, func_compute_stats in funcs_compute_stats.items():
        all_stats[stat_name] = [round(func_compute_stats(ex), 2) for ex in exs]

    return all_stats


if __name__ == "__main__":
    st.title("Visualization to help choosing the filtering parameters for web documents at node level")
    st.set_page_config(layout="wide")

    path_web_document_dataset = "./large_files/web_document_dataset_45M_shard_2"  # Use a web document format, like OBELISC
    path_common_words = "./large_files/common_words.json"  # Find it at https://drive.google.com/file/d/1TeydSroOOmlEuxIcwgsJQ2YF4kPJR6N4/view?usp=sharing
    path_lang_id_model = "./large_files/lid.176.bin"  # Find it at https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    path_sentencepiece_model = "./large_files/en.sp.model"  # Find it at https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/en.sp.model
    path_kenlm_model = "./large_files/en.arpa.bin"  # Find it at https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/en.arpa.bin

    web_document_dataset = load_from_disk(path_web_document_dataset)
    with open(path_common_words) as f:
        common_words = json.load(f)
    lang_id_model = fasttext.load_model(path_lang_id_model)
    sentencepiece_model = sentencepiece.SentencePieceProcessor()
    sentencepiece_model.load(path_sentencepiece_model)
    kenlm_model = kenlm.Model(path_kenlm_model)

    st.header("Number of web documents to consider")
    num_considered_examples = st.number_input(
        "Choose the number of web documents to consider",
        min_value=0,
        max_value=web_document_dataset.num_rows,
        value=100,
        help=f"Enter a number between 0 and {web_document_dataset.num_rows}",
    )
    web_document_dataset = web_document_dataset.shuffle(seed=42).select(range(num_considered_examples))

    st.header("Statistic to consider")
    statistics_considered = st.multiselect(
        label="Choose the statistics to consider",
        options=[
            "original width",
            "original height",
            "aspect ratio",
            "number of words",
            "character repetition ratio",
            "word repetition ratio",
            "special character ratio",
            "stop word ratio",
            "flagged word ratio",
            "punctuation ratio",
            "common word ratio",
            "langage identification confidence score",
            "perplexity score",
        ],
        default=[
            "number of words",
            "character repetition ratio",
        ],
    )

    all_funcs = {}
    if "original width" in statistics_considered:
        type_exs = "images"
        func_compute_stats = lambda img: img.size[0]  # noqa: E731
        all_funcs["original width"] = func_compute_stats
    if "original height" in statistics_considered:
        type_exs = "images"
        func_compute_stats = lambda img: img.size[1]  # noqa: E731
        all_funcs["original height"] = func_compute_stats
    if "aspect ratio" in statistics_considered:
        type_exs = "images"
        func_compute_stats = lambda img: img.size[0] / img.size[1]  # noqa: E731
        all_funcs["aspect ratio"] = func_compute_stats
    if "number of words" in statistics_considered:
        type_exs = "texts"
        func_compute_stats = lambda txt: len(  # noqa: E731
            FilteringFunctions.split_on_whitespace(text=txt, new_line=True, tab=True)
        )
        all_funcs["number of words"] = func_compute_stats
    if "character repetition ratio" in statistics_considered:
        type_exs = "texts"
        func_compute_stats = lambda txt: FilteringFunctions.compute_character_repetition_ratio(  # noqa: E731
            text=txt, character_repetition_length=10
        )
        all_funcs["character repetition ratio"] = func_compute_stats
    if "word repetition ratio" in statistics_considered:
        type_exs = "texts"
        func_compute_stats = lambda txt: FilteringFunctions.compute_word_repetition_ratio(  # noqa: E731
            text=txt, strip_characters=SPECIAL_CHARACTERS, word_repetition_length=5
        )
        all_funcs["word repetition ratio"] = func_compute_stats
    if "special character ratio" in statistics_considered:
        type_exs = "texts"
        func_compute_stats = lambda txt: FilteringFunctions.compute_special_character_ratio(  # noqa: E731
            text=txt, special_characters=SPECIAL_CHARACTERS
        )
        all_funcs["special character ratio"] = func_compute_stats
    if "stop word ratio" in statistics_considered:
        type_exs = "texts"
        func_compute_stats = lambda txt: FilteringFunctions.compute_stopword_ratio(  # noqa: E731
            text=txt, strip_characters=SPECIAL_CHARACTERS, stopwords=STOPWORDS
        )
        all_funcs["stop word ratio"] = func_compute_stats
    if "flagged word ratio" in statistics_considered:
        type_exs = "texts"
        func_compute_stats = lambda txt: FilteringFunctions.compute_flagged_word_ratio(  # noqa: E731
            text=txt, strip_characters=SPECIAL_CHARACTERS, flagged_words=FLAGGED_WORDS
        )
        all_funcs["flagged word ratio"] = func_compute_stats
    if "punctuation ratio" in statistics_considered:
        type_exs = "texts"
        func_compute_stats = lambda txt: FilteringFunctions.compute_punctuation_ratio(  # noqa: E731
            text=txt, punctuation=PUNCTUATION, min_nb_words=10
        )
        all_funcs["punctuation ratio"] = func_compute_stats
    if "common word ratio" in statistics_considered:
        type_exs = "texts"
        func_compute_stats = lambda txt: FilteringFunctions.compute_common_word_ratio(  # noqa: E731
            text=txt,
            strip_characters=SPECIAL_CHARACTERS,
            common_words=common_words,
        )
        all_funcs["common word ratio"] = func_compute_stats
    if "language identification confidence score" in statistics_considered:
        type_exs = "texts"
        func_compute_stats = lambda txt: FilteringFunctions.compute_lang_id_pred_score(  # noqa: E731
            text=txt, lang_id_model=lang_id_model
        )[1]
        all_funcs["language identification confidence score"] = func_compute_stats
    if "perplexity score" in statistics_considered:
        type_exs = "texts"
        func_compute_stats = lambda txt: FilteringFunctions.compute_perplexity_score(  # noqa: E731
            text=txt,
            non_printing_characters_re=NON_PRINTING_CHARACTERS_RE,
            digits_re=DIGITS_RE,
            unicode_punctuation=UNICODE_PUNCTUATION,
            sentencepiece_model=sentencepiece_model,
            kenlm_model=kenlm_model,
        )
        all_funcs["perplexity score"] = func_compute_stats

    # Check we are not mixing image funcs and text funcs
    # This is still TODO. For now, it will just fail

    stats = get_exs_and_stats(
        web_document_dataset=web_document_dataset, type_exs=type_exs, funcs_compute_stats=all_funcs
    )

    st.header("Distribution of the considered statistic")
    bins = st.number_input("Number of bins", min_value=0, max_value=100, value=25)
    fig, ax = plt.subplots(len(all_funcs), 1)
    if len(all_funcs) == 1:
        ax = [ax]
    i = 0
    for stat_name, stat_list in stats.items():
        if stat_name == "exs":
            continue
        truncated_stat_list = np.sort(stat_list)[int(5 / 100 * len(stat_list)) : int(95 / 100 * len(stat_list))]
        ax[i].hist(truncated_stat_list, bins=bins)
        ax[i].set_title(f"{stat_name}")
        i += 1
    fig.suptitle("Histograms of the considered statistics (both top and bottom 5% values are removed)")
    fig.set_figheight(3 * len(all_funcs))
    st.pyplot(fig)

    st.header("A closer look at the data")
    type_exs = "texts" if (type(stats["exs"][0]) == str) else "images"

    if type_exs == "images":

        def transform_img(img):
            img.thumbnail((50, 50))
            with BytesIO() as buffer:
                img.save(buffer, "png")
                base_64_encoding = base64.b64encode(buffer.getvalue()).decode()
            return f'<img src="data:image/png;base64,{base_64_encoding}">'

        if type_exs == "images":
            stats["exs"] = [transform_img(img) for img in stats.pop("exs")]

        data_frame = pd.DataFrame(stats)
        html_data_frame = data_frame.to_html(escape=False)
        st.markdown(html_data_frame, unsafe_allow_html=True)

    elif type_exs == "texts":
        data_frame = pd.DataFrame(stats)
        st.dataframe(data_frame)

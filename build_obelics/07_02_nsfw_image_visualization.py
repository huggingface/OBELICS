import base64
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from datasets import load_from_disk
from PIL import Image


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Visualization to help choosing the NSFW filtering parameters")

    path_image_dataset_with_nsfw_scores = (  # Find at s3://m4-datasets/trash/image_dataset_25k_with_nsfw_scores/
        "./large_files/image_dataset_25k_with_nsfw_scores"
    )
    image_dataset_with_nsfw_scores = load_from_disk(path_image_dataset_with_nsfw_scores)

    st.header("Number of images to consider")
    num_considered_examples = st.number_input(
        "Choose the number of images to consider",
        min_value=0,
        max_value=image_dataset_with_nsfw_scores.num_rows,
        value=1_000,
        help=f"Enter a number between 0 and {image_dataset_with_nsfw_scores.num_rows}",
    )
    image_dataset_with_nsfw_scores = image_dataset_with_nsfw_scores.select(range(num_considered_examples))

    stats = {}
    stats["images"] = image_dataset_with_nsfw_scores["image"]
    stats["hentai_score"] = [
        round(nsfw_scores_["hentai"], 2) for nsfw_scores_ in image_dataset_with_nsfw_scores["nsfw_scores"]
    ]
    stats["porn_score"] = [
        round(nsfw_scores_["porn"], 2) for nsfw_scores_ in image_dataset_with_nsfw_scores["nsfw_scores"]
    ]
    stats["sexy_score"] = [
        round(nsfw_scores_["sexy"], 2) for nsfw_scores_ in image_dataset_with_nsfw_scores["nsfw_scores"]
    ]

    st.header("Distribution of the statistics")
    bins = st.number_input("Number of bins", min_value=0, max_value=100, value=25)
    fig, ax = plt.subplots(len(stats) - 1, 1)  # -1 for the key "images" which is not a statistic
    i = 0
    for stat_name, stat_list in stats.items():
        if stat_name == "images":
            continue
        truncated_stat_list = np.sort(stat_list)
        ax[i].hist(truncated_stat_list, bins=bins)
        ax[i].set_title(f"{stat_name}")
        i += 1
    fig.suptitle("Histograms of the statistics")
    fig.set_figheight(3 * (len(stats) - 1))
    st.pyplot(fig)

    st.header("A closer look at the data")

    stat_sort = st.selectbox("Descending sort by", options=["hentai score", "porn score", "sexy score"])
    stat_sort = stat_sort.replace(" ", "_")
    idx_sort = np.argsort(stats[stat_sort])[::-1].tolist()
    stats["images"] = [stats["images"][idx] for idx in idx_sort]
    stats["hentai_score"] = [stats["hentai_score"][idx] for idx in idx_sort]
    stats["porn_score"] = [stats["porn_score"][idx] for idx in idx_sort]
    stats["sexy_score"] = [stats["sexy_score"][idx] for idx in idx_sort]

    def transform_img(img):
        img = Image.open(BytesIO(img))
        img.thumbnail((50, 50))
        with BytesIO() as buffer:
            img.save(buffer, "png")
            base_64_encoding = base64.b64encode(buffer.getvalue()).decode()
        return f'<img src="data:image/png;base64,{base_64_encoding}">'

    stats["images"] = [transform_img(img) for img in stats.pop("images")]
    data_frame = pd.DataFrame(stats)
    html_data_frame = data_frame.to_html(escape=False)
    st.markdown(html_data_frame, unsafe_allow_html=True)

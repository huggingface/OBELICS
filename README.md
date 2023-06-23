# OBELISC

OBELISC is an open, massive and curated collection of interleaved image-text web documents, containing 141M documents, 115B text tokens and 353M images.

Dataset page: https://huggingface.co/datasets/HuggingFaceM4/OBELISC
Visualization of OBELISC web documents: https://huggingface.co/spaces/HuggingFaceM4/obelisc_visualization
Paper: Arxiv link soon


## Goal and organization of `obelisc`

The folder `obelisc` is aimed to:
- Download WARC files from Common Crawl dumps (`obelisc/processors/warc_downloader.py`);
- Extract HTML files from WARC files (`obelisc/processors/html_extractor.py`);
- Simplify HTML DOM trees (`obelisc/processors/dom_tree_simplificator.py`);
- Convert the simplified DOM trees to another structure adapted for an extraction (`obelisc/processors/pre_extraction_simplificator.py`);
- Perform an extraction (`obelisc/processors/web_document_extractor.py`);
- Perform a filtering on the extraction (`obelisc/processors/web_document_filtering.py`);
- Perform a line deduplication (`obelisc/processors/web_document_line_deduplication.py`);
- Visualize the results (`obelisc/visualization/*`).

The primary techniques are defined in the sub-folder `processors`, while their invocation is found in `callers`. The configs used for the extraction and the filtering of the documents are in `configs`.

We refer to our paper for details about these steps.

In `visualization`, there are different `streamlit` visualizations:
- `global_visualization.py` to see original web pages and DOM trees, with our simplificated versions pre-filtering;
- `choose_filtering_parameters_web_documents_node_level.py` and `web_document_and_filtering_visualization.py` to see the impact of the filtering at node and document level, and help choosing the filter thresholds.
- `web_document_visualization.py`


## Goal and organization of `build_obelisc`

In the folder `build_obelisc`, we are giving all the scripts that were used for the creation of OBELISC, with numbers indicating the chronology.

These scripts often call methods defined in `obelisc/processors/` but not only, and also define other useful methods.



If you are using this dataset or this code, please cite
```
@inproceedings{
lauren{\c{c}}on2023obe,
title={OBELISC: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents},
author={Hugo Lauren{\c{c}}on and Lucile Saulnier and L{\'e}o Tronchon and Stas Bekman and Amanpreet Singh and Anton Lozhkov and Thomas Wang and Siddharth Karamcheti and Alexander M Rush and Douwe Kiela and Matthieu Cord and Victor Sanh},
year={2023}
}
```

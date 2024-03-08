# OBELICS

**OBELICS is an open, massive and curated collection of interleaved image-text web documents, containing 141M documents, 115B text tokens and 353M images.**

**Dataset page:** https://huggingface.co/datasets/HuggingFaceM4/OBELICS

**Visualization of OBELICS web documents:** https://huggingface.co/spaces/HuggingFaceM4/obelics_visualization

**Paper:** https://arxiv.org/abs/2306.16527


## Goal and organization of [obelics](https://github.com/huggingface/OBELICS/tree/main/obelics)

The folder [obelics](https://github.com/huggingface/OBELICS/tree/main/obelics) is aimed at:
- Download WARC files from Common Crawl dumps ([warc_downloader.py](https://github.com/huggingface/OBELICS/blob/main/obelics/processors/warc_downloader.py));
- Extract HTML files from WARC files ([html_extractor.py](https://github.com/huggingface/OBELICS/blob/main/obelics/processors/html_extractor.py));
- Simplify HTML DOM trees ([dom_tree_simplificator.py](https://github.com/huggingface/OBELICS/blob/main/obelics/processors/dom_tree_simplificator.py));
- Convert the simplified DOM trees to another structure adapted for an extraction ([pre_extraction_simplificator.py](https://github.com/huggingface/OBELICS/blob/main/obelics/processors/pre_extraction_simplificator.py));
- Perform an extraction ([web_document_extractor.py](https://github.com/huggingface/OBELICS/blob/main/obelics/processors/web_document_extractor.py));
- Perform a filtering on the extraction ([web_document_filtering.py](https://github.com/huggingface/OBELICS/blob/main/obelics/processors/web_document_filtering.py));
- Perform a line deduplication ([web_document_line_deduplication.py](https://github.com/huggingface/OBELICS/blob/main/obelics/processors/web_document_line_deduplication.py));
- Visualize the results ([visualization](https://github.com/huggingface/OBELICS/tree/main/obelics/visualization)).

The primary techniques are defined in the sub-folder [processors](https://github.com/huggingface/OBELICS/tree/main/obelics/processors), while their invocation is found in [callers](https://github.com/huggingface/OBELICS/tree/main/obelics/callers). The configs used for the extraction and the filtering of the documents are in [configs](https://github.com/huggingface/OBELICS/tree/main/obelics/configs).

We refer to our paper for details about these steps.

In [visualization](https://github.com/huggingface/OBELICS/tree/main/obelics/visualization), there are different `streamlit` visualizations:
- [global_visualization.py](https://github.com/huggingface/OBELICS/blob/main/obelics/visualization/global_visualization.py) to see original web pages and DOM trees, with our simplificated versions pre-filtering;
- [choose_filtering_parameters_web_documents_node_level.py](https://github.com/huggingface/OBELICS/blob/main/obelics/visualization/choose_filtering_parameters_web_documents_node_level.py) and [web_document_and_filtering_visualization.py](https://github.com/huggingface/OBELICS/blob/main/obelics/visualization/web_document_and_filtering_visualization.py) to see the impact of the filtering at node and document level, and help choosing the filter thresholds.
- [web_document_visualization.py](https://github.com/huggingface/OBELICS/blob/main/obelics/visualization/web_document_visualization.py) for a simple visualization of the final documents.


## Goal and organization of [build_obelics](https://github.com/huggingface/OBELICS/tree/main/build_obelics)

In the folder [build_obelics](https://github.com/huggingface/OBELICS/tree/main/build_obelics), we are giving all the scripts that were used for the creation of OBELICS, with numbers indicating the chronology.

These scripts often call methods defined in [processors](https://github.com/huggingface/OBELICS/tree/main/obelics/processors) but not only, and also define other useful methods.


## Citation

If you are using this dataset or this code, please cite
```
@misc{laurencon2023obelics,
      title={OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents},
      author={Hugo Laurençon and Lucile Saulnier and Léo Tronchon and Stas Bekman and Amanpreet Singh and Anton Lozhkov and Thomas Wang and Siddharth Karamcheti and Alexander M. Rush and Douwe Kiela and Matthieu Cord and Victor Sanh},
      year={2023},
      eprint={2306.16527},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

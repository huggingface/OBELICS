import re

from obelisc.utils import (
    MEDIA_CONTAIN_INTERESTING_ATTRIBUTES_SET,
    TAG_TO_SEP,
    simplify_media_node,
)


class Node:
    def __init__(self, path_in_tree, media_info, text, children):
        self.path_in_tree = path_in_tree
        self.media_info = media_info
        self.text = text
        self.children = children

    @property
    def tag(self):
        return self.path_in_tree[-1][0]

    @property
    def level(self):
        return len(self.path_in_tree)


class Tree:
    def __init__(
        self,
        selectolax_root_node,
        page_url,
    ):
        self.num_nodes = 0
        self.tree = self.make_tree(selectolax_root_node, page_url)

    def make_tree(self, selectolax_node, page_url, path_in_tree=[]):
        tag = selectolax_node.tag
        path_in_tree = path_in_tree + [[tag, self.num_nodes]]
        self.num_nodes += 1

        if tag in MEDIA_CONTAIN_INTERESTING_ATTRIBUTES_SET:
            return Node(
                path_in_tree=path_in_tree,
                media_info=simplify_media_node(selectolax_node, page_url=page_url),
                text="",
                children=[],
            )

        elif tag == "-text":
            return Node(
                path_in_tree=path_in_tree,
                media_info=None,
                text=selectolax_node.text(deep=False, separator="", strip=False),
                children=[],
            )

        return Node(
            path_in_tree=path_in_tree,
            media_info=None,
            text="",
            children=[
                self.make_tree(child, page_url=page_url, path_in_tree=path_in_tree)
                for child in selectolax_node.iter(include_text=True)
            ],
        )

    def traverse(self):
        def traverse_recursive(node, list_nodes):
            list_nodes.append(node)
            for child_node in node.children:
                traverse_recursive(child_node, list_nodes=list_nodes)

        list_nodes = []
        traverse_recursive(self.tree, list_nodes=list_nodes)
        return list_nodes


class PreExtractionSimplificator:
    def __init__(
        self,
        only_text_image_nodes=True,
        format_texts=True,
        merge_consecutive_text_nodes=True,
    ):
        self.only_text_image_nodes = only_text_image_nodes
        self.format_texts = format_texts
        self.merge_consecutive_text_nodes = merge_consecutive_text_nodes

    def __call__(self, selectolax_tree, page_url):
        tree = Tree(selectolax_tree.root, page_url=page_url)
        list_nodes = tree.traverse()

        if self.only_text_image_nodes:
            list_nodes = self._only_text_image_nodes(list_nodes)
        if self.format_texts:
            list_nodes = self._format_texts(list_nodes)
        if self.merge_consecutive_text_nodes:
            list_nodes = self._merge_consecutive_text_nodes(list_nodes)

        return list_nodes

    def _only_text_image_nodes(self, list_nodes):
        list_nodes = [
            node
            for node in list_nodes
            if (node.tag == "-text") or (node.tag == "figure") or ((node.tag == "img") and (node.media_info))
        ]
        return list_nodes

    def _format_texts(self, list_nodes):
        def format_one_text(text):
            if text == "":
                return text
            text = text.replace("\n", " ")
            text = text.replace("\t", " ")
            text = re.sub(r"[ ]{2,}", " ", text)
            beg_sep = " " == text[0]
            end_sep = (" " == text[-1]) and (len(text) > 1)
            text = "\n".join([el.strip() for el in text.split("#BR_TAG#")])
            text = beg_sep * " " + text + end_sep * " "
            return text

        for idx, node in enumerate(list_nodes):
            list_nodes[idx].text = format_one_text(node.text)
        list_nodes = [node for node in list_nodes if (node.tag != "-text") or ((node.tag == "-text") and (node.text))]
        return list_nodes

    def _merge_consecutive_text_nodes(self, list_nodes):
        current_idx = 0
        while current_idx <= len(list_nodes) - 1:
            if list_nodes[current_idx].tag != "-text":
                current_idx += 1
            else:
                if (current_idx == len(list_nodes) - 1) or (
                    (current_idx + 1 <= len(list_nodes) - 1) and (list_nodes[current_idx + 1].tag != "-text")
                ):
                    list_nodes[current_idx].path_in_tree = [["-text", 0]]
                    list_nodes[current_idx].text = list_nodes[current_idx].text.strip()
                    current_idx += 1
                else:
                    seps = set()

                    text_1 = list_nodes[current_idx].text
                    text_2 = list_nodes[current_idx + 1].text

                    for char in ["\n\n", "\n", " "]:
                        if text_1.endswith(char):
                            seps.add(char)
                            text_1 = text_1[: -len(char)]
                        if text_2.startswith(char):
                            seps.add(char)
                            text_2 = text_2[len(char) :]

                    path_1 = list_nodes[current_idx].path_in_tree
                    path_2 = list_nodes[current_idx + 1].path_in_tree

                    start_diff_path = 0
                    for i in range(min(len(path_1), len(path_2))):
                        if path_1[i] != path_2[i]:
                            start_diff_path = i
                            break
                    for tag, _ in path_1[start_diff_path:] + path_2[start_diff_path:]:
                        if tag in TAG_TO_SEP:
                            seps.add(TAG_TO_SEP[tag])

                    if "\n\n" in seps:
                        sep = "\n\n"
                    elif "\n" in seps:
                        sep = "\n"
                    elif " " in seps:
                        sep = " "
                    else:
                        sep = ""

                    list_nodes[current_idx].path_in_tree = path_2
                    list_nodes[current_idx].text = text_1 + sep + text_2
                    del list_nodes[current_idx + 1]

        list_nodes = [
            node for node in list_nodes if (node.tag != "-text") or ((node.tag == "-text") and (node.text.strip()))
        ]
        return list_nodes

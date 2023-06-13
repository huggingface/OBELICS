import re

from obelisc.utils import (
    INTERESTING_TAGS_SET,
    MEDIA_CONTAIN_INTERESTING_ATTRIBUTES_SET,
    UNWRAP_TAGS,
    get_media_src,
    make_selectolax_tree,
)


class DOMTreeSimplificator:
    def __init__(
        self,
        strip_multiple_linebreaks=True,
        strip_multiple_spaces=True,
        remove_html_comments=True,
        replace_line_break_tags=True,
        unwrap_tags=True,
        strip_tags=True,
        strip_special_divs=True,
        remove_dates=True,
        remove_empty_leaves=True,
        unnest_nodes=True,
        remake_tree=True,
        css_rules=None,
        css_rules_replace_with_text=None,
    ):
        self.strip_multiple_linebreaks = strip_multiple_linebreaks
        self.strip_multiple_spaces = strip_multiple_spaces
        self.remove_html_comments = remove_html_comments
        self.replace_line_break_tags = replace_line_break_tags
        self.unwrap_tags = unwrap_tags
        self.strip_tags = strip_tags
        self.strip_special_divs = strip_special_divs
        self.remove_dates = remove_dates
        self.remove_empty_leaves = remove_empty_leaves
        self.unnest_nodes = unnest_nodes
        self.remake_tree = remake_tree
        self.css_rules = css_rules
        self.css_rules_replace_with_text = css_rules_replace_with_text

    def __call__(
        self,
        html_str,
        type_return,
    ):
        if type_return not in ["str", "selectolax_tree"]:
            raise ValueError("`type_return` must be `str` or `selectolax_tree`")

        if self.strip_multiple_linebreaks:
            html_str = self._strip_multiple_linebreaks(html_str)
        if self.strip_multiple_spaces:
            html_str = self._strip_multiple_spaces(html_str)
        if self.remove_html_comments:
            html_str = self._remove_html_comments(html_str)
        if self.replace_line_break_tags:
            html_str = self._replace_line_break_tags(html_str)

        selectolax_tree = make_selectolax_tree(html_str)

        if self.css_rules:
            selectolax_tree = self._remove_nodes_matching_css_rules(selectolax_tree)
        if self.css_rules_replace_with_text:
            selectolax_tree = self._replace_nodes_matching_css_rules_with_text(selectolax_tree)
        if self.unwrap_tags:
            selectolax_tree = self._unwrap_html_tree(selectolax_tree)
        if self.strip_tags:
            selectolax_tree = self._strip_html_tree(selectolax_tree)

        if self.strip_special_divs:
            selectolax_tree = self._strip_special_divs(selectolax_tree)
        if self.remove_dates:
            selectolax_tree = self._remove_dates(selectolax_tree)
        if self.remove_empty_leaves:
            selectolax_tree = self._remove_empty_leaves(selectolax_tree)
        if self.unnest_nodes:
            selectolax_tree = self._unnest_nodes(selectolax_tree)
        if self.remake_tree:
            selectolax_tree = self._remake_tree(selectolax_tree)

        if type_return == "str":
            return selectolax_tree.html
        elif type_return == "selectolax_tree":
            return selectolax_tree

    def _strip_multiple_linebreaks(self, html_str):
        html_str = re.sub(r"[\n]{2,}", "\n", html_str)
        return html_str

    def _strip_multiple_spaces(self, html_str):
        html_str = re.sub(r"[ ]{2,}", " ", html_str)
        return html_str

    def _remove_html_comments(self, html_str):
        html_str = re.sub(r"<!--(?s).*?-->", "", html_str)
        return html_str

    def _replace_line_break_tags(self, html_str):
        html_str = re.sub("<br>|<br/>|<br />|</br>", "#BR_TAG#", html_str)
        return html_str

    def _unwrap_html_tree(self, selectolax_tree):
        selectolax_tree.unwrap_tags(UNWRAP_TAGS)

        # `.unwrap_tags` won't unwrap+remove tags which are empty leaves (i.e. either with no children tags or no text).
        # For instance, `<a href="https://twitter.com/share"><img src="blo.png"></a>` will be unwrapped and mentions
        # of `a` will be removed, but `<a href="https://twitter.com/share"></a>` will stay as is.
        # As a consequence, we call `strip_tags` to remove these unwrap empty tags for good.
        selectolax_tree.strip_tags(UNWRAP_TAGS)

        return selectolax_tree

    def _remove_digits_string(self, string):
        string = re.sub(r"\d+", "", string)
        return string

    def _remove_nodes_matching_css_rules(self, selectolax_tree):
        modification = True
        while modification:
            found_a_node = False
            for css_rule in self.css_rules:
                for node in selectolax_tree.css(css_rule):
                    if node.tag != "html":
                        node.decompose()
                        found_a_node = True
                        break
            if not found_a_node:
                modification = False
        return selectolax_tree

    def _replace_nodes_matching_css_rules_with_text(self, selectolax_tree):
        for css_rule, text in self.css_rules_replace_with_text.items():
            for node in selectolax_tree.css(css_rule):
                node.replace_with(text)
        return selectolax_tree

    def _strip_html_tree(self, selectolax_tree):
        strip_tags_l = [
            node.tag
            for node in selectolax_tree.root.traverse()
            if self._remove_digits_string(node.tag) not in INTERESTING_TAGS_SET
        ]
        strip_tags_l = list(set(strip_tags_l))
        selectolax_tree.strip_tags(strip_tags_l)
        return selectolax_tree

    def _strip_special_divs(self, selectolax_tree):
        special_div_ids = ["footer", "header", "navigation", "nav", "navbar", "menu"]
        modification = True
        while modification:
            # Traverse the tree to find one node to remove, and remove it right then
            # to avoid the recursivity problem with `decompose`
            found_a_node = False
            for node in selectolax_tree.root.traverse():
                if node.tag == "div":
                    attributes = node.attributes
                    if (
                        ("id" in attributes and attributes["id"] in special_div_ids)
                        or ("class" in attributes and attributes["class"] in special_div_ids)
                        or ("title" in attributes and attributes["title"] in special_div_ids)
                    ):
                        node.decompose()
                        found_a_node = True
                        break
            if not found_a_node:
                modification = False
        return selectolax_tree

    def _remove_dates(self, selectolax_tree):
        nodes_to_remove = []
        for node in selectolax_tree.root.traverse():
            if node.tag == "div":
                if node.attributes:
                    if "class" in node.attributes:
                        if node.attributes["class"]:
                            if "date" in node.attributes["class"]:
                                nodes_to_remove += [
                                    child for child in node.iter(include_text=True) if child.tag == "-text"
                                ]
        for node in nodes_to_remove:
            node.decompose()
        return selectolax_tree

    def _remove_empty_leaves(self, selectolax_tree):
        """
        Function used to remove empty leaves iteratively, so it also ends up also removing nodes
        that are higher up in the tree.
        """
        modification = True
        while modification:
            nodes_to_remove = [
                node
                for node in selectolax_tree.root.traverse()
                if (
                    (node.tag not in MEDIA_CONTAIN_INTERESTING_ATTRIBUTES_SET)
                    and (not [child for child in node.iter()])
                    and (not node.text().strip())
                    and (node.tag != "html")
                )
                or (
                    (node.tag in MEDIA_CONTAIN_INTERESTING_ATTRIBUTES_SET)
                    and not get_media_src(node)
                )
            ]
            if nodes_to_remove:
                for node in nodes_to_remove:
                    node.decompose(recursive=False)
            else:
                modification = False
        return selectolax_tree

    def _unnest_nodes(self, selectolax_tree):
        modification = True
        while modification:
            modification = False
            for node in selectolax_tree.root.traverse():
                children = [child for child in node.iter()]
                if len(children) == 1:
                    child = children[0]
                    if node.tag == child.tag:
                        text = node.text(deep=False).strip()
                        if not text:
                            node.replace_with(child)
                            modification = True
                            break
        return selectolax_tree

    def _remake_tree(self, selectolax_tree):
        """It could be interesting to remake a tree after the
        simplifications since it can now merge some text nodes
        that couldn't be merged before"""
        return make_selectolax_tree(selectolax_tree.html)

from selectolax.parser import HTMLParser


def make_selectolax_tree(html_str):
    selectolax_tree = HTMLParser(html_str)
    return selectolax_tree

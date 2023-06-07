import os
import re
from urllib.parse import urlparse


TAG_TO_SEP = {
    "address": "\n",
    "article": "\n",
    "aside": "\n",
    "blink": "",
    "blockquote": "\n\n",
    "body": "",
    "caption": "",
    "center": "\n",
    "dd": "\n",
    "dl": "\n\n",
    "dt": "\n",
    "div": "\n",
    "figcaption": "\n",
    "h": "",
    "h1": "\n\n",
    "h2": "\n\n",
    "h3": "\n\n",
    "h4": "\n\n",
    "h5": "\n\n",
    "h6": "\n\n",
    "hgroup": "\n",
    "html": "",
    "legend": "\n",
    "main": "\n",
    "marquee": "\n",
    "ol": "\n\n",
    "p": "\n\n",
    "section": "\n",
    "summary": "\n",
    "title": "",
    "ul": "\n\n",
}


def get_media_src(node):
    node_attributes = node.attributes
    node_tag = node.tag
    src = None

    if node_tag == "img":
        # Check all possible source type, and keep the first valid one
        img_source_types = [
            "src",
            "data-src",
            "data-src-fg",
            "data-scroll-image",
            "srcset",
            "data-lateloadsrc",
            "data-img-src",
            "data-original",
            "data-gt-lazy-src",
            "data-lazy",
            "data-lazy-src",
            "src2",
        ]
        for source_type in img_source_types:
            if source_type in node_attributes and node_attributes[source_type]:
                if ("," not in node_attributes[source_type]) and (" " not in node_attributes[source_type]):
                    src = node_attributes[source_type]
                    break

    elif node_tag == "video":
        if ("src" in node_attributes) and node_attributes["src"]:
            src = node_attributes["src"]
        else:
            for cnode in node.iter():
                if not src:
                    if cnode.tag == "source":
                        cnode_attributes = cnode.attributes
                        if ("src" in cnode_attributes) and cnode_attributes["src"]:
                            src = cnode_attributes["src"]

    elif node_tag == "audio":
        if ("src" in node_attributes) and node_attributes["src"]:
            src = node_attributes["src"]
        else:
            for cnode in node.iter():
                if not src:
                    if cnode.tag == "source":
                        cnode_attributes = cnode.attributes
                        if ("src" in cnode_attributes) and cnode_attributes["src"]:
                            src = cnode_attributes["src"]
    else:
        return None  # TODO iframes

    # Check on comma because it's non-canonical and should not be used anyway in urls.
    # TODO: have checks on valid URLs
    # Useless (at least for images) since already checked
    if src is not None and (("," in src) or (" " in src)):
        return None

    return src


def format_image_size(size):
    size = re.sub('[”"<>]', "", size)
    try:
        return int(size)
    except ValueError:
        if "px" in size:
            return int(re.sub("[px;]", "", size))
        elif "%" in size:  # That should be the only case where we can't return a integer.
            return size.strip()
        elif "." in size:  # If it's a float, then to make it simple, round it.
            return int(float(size))
        elif "full-width" == size or "auto" == size:
            return "100%"
        else:
            raise ValueError(f"Unrecognized size for image: `{size}`")


def format_filename(filename):
    # TODO: refine this function. fairly imprefect.
    # Potential improvements: `Untitled`, `untitled`, `blank`, check whether each word is in a dictionary
    _, simp_filename = os.path.split(filename)
    simp_filename = simp_filename.split(".")[0]

    if re.findall(
        r"\?[A-Za-z0-9]+=", simp_filename
    ):  # Example `it?ids=2019042515182454151475%3A027064510%3A001&ca=n&coo=y`
        return ""

    simp_filename = re.sub(r"[_-]", " ", simp_filename)  # Example: `Chocolate_Berry_Frozen_Yogurt_Bark`
    simp_filename = re.sub(r"%2[0]*", " ", simp_filename)  # Example: `hearts%2Band%2Bhome%20Bbadge`
    simp_filename = re.sub(r"[0-9]+x[0-9]+", "", simp_filename)  # Example: `104x403`
    simp_filename = re.sub(r"[0-9]+", " ", simp_filename)  # Example: `icon18_wrench_allbkg`
    simp_filename = re.sub(r"[ ]{2,}", " ", simp_filename)  # Example: `icon    wrenchallbkg`

    for r in ["\n", "+", "%B", "%"]:
        simp_filename = simp_filename.replace(r, " ")

    simp_filename = simp_filename.strip()
    if len(simp_filename) <= 1:
        return ""
    else:
        return simp_filename


def format_relative_to_absolute_path(page_url, relative_path):
    if relative_path.startswith("//"):
        abs_path = "http:" + relative_path
    else:
        if "./" in relative_path:
            relative_path = re.sub(r"\.+\/", "", relative_path)
        if not relative_path.startswith("/"):
            relative_path = "/" + relative_path
        domain_name = urlparse(page_url).netloc
        abs_path = "https://" + domain_name + relative_path
    return abs_path


def is_url_valid(url):
    regex = re.compile(
        r"^(?:http|ftp)s?://"  # http:// or https://
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?))"  # domain...
        r"(?:/?|[/?]\S+)$",
        re.IGNORECASE,
    )
    return re.match(regex, url) is not None


def simplify_media_node(node, page_url):
    src = get_media_src(node)

    if not src:
        return None
    unformatted_src = src
    if not src.startswith("http"):
        src = format_relative_to_absolute_path(page_url=page_url, relative_path=src)
    if not is_url_valid(src):
        return None

    node_attributes = node.attributes
    if node.tag == "img":
        new_image = {"document_url": page_url}
        new_image["unformatted_src"] = unformatted_src
        new_image["src"] = src

        formatted_filename = format_filename(src)
        if formatted_filename:
            new_image["formatted_filename"] = formatted_filename

        if ("alt" in node_attributes) and node_attributes["alt"]:
            new_image["alt_text"] = node_attributes["alt"]

        # TODO: eventually, for image sizes we could parse cases like
        # `{'src': 'http://wellbeingteams.org/wp-content/uploads/2017/04/spread600300.jpg',
        # 'width': None, 'height': None, 'alt': None, 'title': 'spread600300', 'class': 'img-responsive wp-image-122',
        # 'srcset': 'https://wellbeingteams.org/wp-content/uploads/2017/04/spread600300-200x100.jpg 200w, https://wellbeingteams.org/wp-content/uploads/2017/04/spread600300-400x200.jpg 400w, https://wellbeingteams.org/wp-content/uploads/2017/04/spread600300.jpg 600w',
        # 'sizes': '(max-width: 800px) 100vw, 400px'}`
        for size in ["width", "height"]:
            if size in node_attributes and node_attributes[size] is not None:
                try:
                    new_image[f"rendered_{size}"] = format_image_size(node_attributes[size])
                except ValueError:
                    pass  # Unrecognized format, generally an error, skipping

        return new_image

    elif node.tag == "video":
        new_video = {"document_url": page_url}
        new_video["src"] = src
        if "width" in node_attributes:
            new_video["width"] = node_attributes["width"]
        if "height" in node_attributes:
            new_video["height"] = node_attributes["height"]
        return new_video

    elif node.tag == "audio":
        new_audio = {"document_url": page_url}
        new_audio["src"] = src
        return new_audio

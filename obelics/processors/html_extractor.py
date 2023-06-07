import io
import sys

from bs4 import BeautifulSoup
from bs4.dammit import EncodingDetector
from warcio.archiveiterator import WARCIterator


# For `soup.decode_content` that can hit the limit
sys.setrecursionlimit(10000)


class HtmlExtractor:
    def __call__(self, example):
        if example["html"] and not example["html_error"]:
            return example

        warc, warc_error = example["warc"], example["warc_error"]
        if warc_error:
            example["html"] = ""
            example["html_error"] = "no WARC"
            return example

        html, html_error = self.get_html_from_warc(warc=warc)
        example["html"] = html
        example["html_error"] = html_error
        return example

    def get_html_from_warc(self, warc):
        page, encoding = None, None

        with io.BytesIO(warc) as stream:
            try:
                for record in WARCIterator(stream):
                    if record.rec_type == "response":
                        page = record.content_stream().read()
                        encoding = record.rec_headers["WARC-Identified-Content-Charset"]
                        break
            except Exception as e:
                return "", str(e)

        if not encoding:
            try:
                for enc in EncodingDetector(page, is_html=True).encodings:
                    # take the first detected encoding
                    encoding = enc
                    break
            except Exception as e:
                return "", str(e)

        if (not page) or (not encoding):
            return "", "Not page or encoding"

        try:
            soup = BeautifulSoup(page, "html.parser", from_encoding=encoding)
        except Exception as e:
            return "", str(e)

        try:
            html_str = soup.decode_contents(formatter="html")
        except Exception as e:
            return "", str(e)

        try:
            html_str.encode()
        except Exception as e:
            return "", str(e)

        return html_str, ""

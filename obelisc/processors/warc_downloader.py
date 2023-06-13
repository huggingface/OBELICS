import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ProxyConnectionError


class WarcDownloader:
    def __init__(self):
        config_boto = Config(
            # region_name="us-east-1", # Location of the CC data, commenting this line since it doesn't help
            retries={"max_attempts": 10, "mode": "standard"}
        )
        self.client = boto3.client("s3", config=config_boto)

    def __call__(self, example):
        if example["warc"] and not example["warc_error"]:
            return example

        warc_filename = example["warc_filename"]
        warc_record_offset = example["warc_record_offset"]
        warc_record_length = example["warc_record_length"]

        warc, warc_error = self.get_warc_from_metadata(
            client=self.client,
            warc_filename=warc_filename,
            warc_record_offset=warc_record_offset,
            warc_record_length=warc_record_length,
        )
        example["warc"] = warc
        example["warc_error"] = warc_error
        return example

    def get_warc_from_metadata(self, client, warc_filename, warc_record_offset, warc_record_length):
        try:
            response = client.get_object(
                Bucket="commoncrawl",
                Key=warc_filename,
                Range=f"bytes={warc_record_offset}-{warc_record_offset + warc_record_length - 1}",
            )
        except (ClientError, ProxyConnectionError) as e:
            return b"", repr(e)
        return response["Body"].read(), ""

    # Needed to make multiprocessing work
    def __reduce__(self):
        return (self.__class__, ())

import asyncio
import os
import time
from typing import List

import aiohttp
from aiolimiter import AsyncLimiter
from datasets import load_from_disk


PATH_DS_IMG_URLS_S3 = (  # Made by concatenating s3://m4-datasets/webdocs/img_urls_in_final_web_docs_2/ + converting to HF dataset
    "s3://m4-datasets/trash/ds_img_urls"
)
PATH_DS_IMG_URLS_LOCAL = "/scratch/ds_img_urls"

PATH_SAVE_DISK_OPT_OUT = "/scratch/ds_opt_out"
PATH_SAVE_S3_OPT_OUT = "s3://m4-datasets/webdocs/ds_opt_out"

MAX_NUM_RETRIES_SYNC = 3

TOKEN = os.environ["SPAWNING_TOKEN"]

BATCH_SIZE = 100_000

headers = {"Authorization": f"API {TOKEN}"}
API_BATCH_SIZE = 1000
RETRY_TIMES = ([1] * 5) + ([10] * 5) + ([10 * 60] * 10) + ([60 * 60] * 10)
TIMEOUT = 60


async def check_spawning(image_urls: List[str], session: aiohttp.ClientSession, semaphore, limiter) -> dict:
    url = "https://opts-api.spawningaiapi.com/api/v2/query/urls"
    if not image_urls:
        return {"urls": []}
    elif len(image_urls) == 1:
        image_urls = image_urls + [""]  # the API requires > 1 urls
    async with semaphore:
        async with limiter:
            resp_body = None
            for retry_time in RETRY_TIMES:
                try:
                    async with session.post(
                        url=url,
                        data="\n".join(image_urls),
                        timeout=TIMEOUT,
                    ) as resp:
                        resp_body = await resp.text()
                        spawning_response = await resp.json()
                        assert "urls" in spawning_response, str(resp_body)
                        return spawning_response
                except Exception:
                    pass
                time.sleep(retry_time)
            with open("erroring_urls.txt", "w") as f:
                f.write("\n".join(image_urls))
            raise RuntimeError(str(resp_body))


async def opt_in_out_task(image_urls: List[str], session, semaphore, limiter) -> tuple:
    spawning_response = await check_spawning(image_urls, session, semaphore, limiter)
    urls_responses = spawning_response["urls"]
    urls_opt_out = [urls_response["optOut"] for urls_response in urls_responses]
    return urls_opt_out


async def parallel_opt_in_out_task(urls: list) -> tuple:
    tasks = []
    semaphore = asyncio.Semaphore(value=10)
    limiter = AsyncLimiter(20, time_period=1)
    async with aiohttp.ClientSession(headers=headers) as session:
        for offset in range(0, len(urls), API_BATCH_SIZE):
            tasks.append(
                asyncio.create_task(
                    opt_in_out_task(urls[offset : offset + API_BATCH_SIZE], session, semaphore, limiter)
                )
            )
        await asyncio.wait(tasks)
    tasks_results = [task.result() for task in tasks]
    tasks_results = [sub_el for el in tasks_results for sub_el in el]
    return tasks_results


def opt_in_out(batch: dict) -> None:
    opt_out_bool = asyncio.run(parallel_opt_in_out_task(batch))
    return opt_out_bool


if __name__ == "__main__":
    command_sync_s3 = f"aws s3 sync {PATH_DS_IMG_URLS_S3} {PATH_DS_IMG_URLS_LOCAL}"
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)

    ds = load_from_disk(PATH_DS_IMG_URLS_LOCAL)

    ds = ds.filter(opt_in_out, input_columns=["url"], batched=True, batch_size=BATCH_SIZE)

    ds.save_to_disk(PATH_SAVE_DISK_OPT_OUT)

    command_sync_s3 = f"aws s3 sync {PATH_SAVE_DISK_OPT_OUT} {PATH_SAVE_S3_OPT_OUT}"
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import bz2
import codecs
import os
import re
import subprocess
from typing import Literal, Optional
from urllib.parse import quote, urlparse

from distributed import Lock
import html

from nemo_curator.datasets import DocumentDataset
from nemo_curator.download.doc_builder import (
    DocumentDownloader,
    DocumentExtractor,
    DocumentIterator,
    download_and_extract,
)

from bs4 import BeautifulSoup
from src.download.download_utils import get_document_url_list, extract_code_blocks, extract_image_urls, extract_url_blocks
from nemo_curator.utils.file_utils import expand_outdir_and_mkdir


class WebDocumentDownloader(DocumentDownloader):

    def __init__(self, download_dir, verbose=False):
        super().__init__()
        self._download_dir = download_dir
        self._verbose = verbose
        self._lock = Lock(name="webdocument_downloader")

    def download(self, url):
        urlpath = urlparse(url).path[1:]
        output_name = urlpath.replace("/", "-")
        output_file = os.path.join(self._download_dir, output_name)
        if os.path.exists(output_file):
            print(f"bz2 file: {output_file} exists. Not downloading")
        else:
            print(f"Downloading {url} and writing to {output_file}")
            # Download with either wget or s5cmd (aws)
            cmd = ["wget", url, "-O", output_file]
            if self._verbose:
                stdout, stderr = None, None
            else:
                stdout, stderr = subprocess.DEVNULL, subprocess.DEVNULL
            with self._lock:
                p = subprocess.run(
                    cmd,
                    stdout=stdout,
                    stderr=stderr,
                )
            if p.returncode != 0:
                print(f"Failed to download {url} to {output_file}")

        return output_file


class WebDocumentIterator(DocumentIterator):

    def __init__(self, base_url:str, log_frequency=1000):
        super().__init__()
        self._base_url = base_url
        self._log_frequency = log_frequency
        self._counter = 0

    def iterate(self, file_path):
        self._counter = 0
        bname = os.path.basename(file_path)
        
        ori_url = bname.replace("-", "/")

        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        # 예시: 콘텐츠가 <article class="bd-article"> 내부에 있다고 가정
        article = soup.find("article", class_="bd-article")
        
        if not article:
            print(f"Article with class 'bd-article' not found in {file_path}")
            return

        # 각 section 태그(단, id 속성이 있는 태그)를 순회
        sections = article.find_all("section", id=True)
        for section in sections:
            self._counter += 1
            if self._counter % self._log_frequency == 0:
                print(f"Processed {self._counter} sections from {file_path}")

            section_id = section.get("id")
            # 제목은 해당 section 내의 h1, h2, h3 등에서 추출
            header_tag = section.find(["h1", "h2", "h3", "h4"])
            header = header_tag.get_text(strip=True) if header_tag else ""

            # 섹션의 전체 텍스트 콘텐츠 추출 (필요에 따라 header를 제거할 수도 있음)
            content = section.get_text(separator="\n", strip=True)

            # URL은 base_url과 section_id를 조합해 생성 (필요에 따라 더 복잡하게 조합 가능)
            url = f"{self._base_url}/{ori_url}#{quote(section_id)}"

            yield {
                "section_id": section_id,
                "header": header,
                "url": url,
                "source_id": bname,
            }, content


class WebDocumentExtractor(DocumentExtractor):

    def __init__(self, base_url: str):
        super().__init__()
        self._base_url = base_url

    def extract(self, content):
        
        soup = BeautifulSoup(content, "html.parser")
        
        url_blocks = extract_url_blocks(soup, self._base_url)
        img_blocks = extract_image_urls(soup, self._base_url)
        code_blocks = extract_code_blocks(soup)
       
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)
        
        # return {"text": text, "meta": }
        yield {
                "code": code_blocks, "urls": url_blocks, "images": img_blocks
            }, text

def download_web_documents(
    target_url: str,
    output_path: str,
    output_type: Literal["jsonl", "parquet"] = "jsonl",
    raw_download_dir: Optional[str] = None,
    keep_raw_download: bool = False,
    force_download: bool = False,
    url_limit: Optional[int] = None,
    record_limit: Optional[int] = None,
) -> DocumentDataset:
    """
    Downloads and extracts articles from a Wikipedia dump.

    This function retrieves a list of Wikipedia dump URLs for the specified language and dump date,
    downloads the compressed bz2 dump file (if it is not already present), and extracts its articles
    using mwparserfromhell. The resulting articles are saved in the specified output format (e.g., "jsonl")
    along with relevant metadata.

    Args:
        target_url (str): document root url
        output_path (str): The root directory where the final extracted files and intermediate outputs
            (if any) are stored.
        output_type (Literal["jsonl", "parquet"], optional): The file format/extension for saving the extracted documents (e.g., "jsonl").
            Defaults to "jsonl". This is not used for the output file, but is used to check if an extracted output
            already exists and read it if so.
        raw_download_dir (Optional[str], optional): Directory used for temporary storage of raw bz2 dump files.
            If None, a subdirectory named "downloads" under output_path is used.
        keep_raw_download (bool, optional): If True, retains the raw bz2 files after extraction.
            Default is False.
        force_download (bool, optional): If False, skips re-downloading or re-extracting files that already exist.
        url_limit (Optional[int], optional): The maximum number of dump file URLs to process. If None, all
            available URLs are processed.
        record_limit (Optional[int], optional): Limit the number of records to extract from each file.
            If None, all available records are extracted.

    Returns:
        DocumentDataset: A dataset object containing the extracted Wikipedia articles along with associated metadata.
    """
    document_urls = get_document_url_list(target_url)
    
    if url_limit:
        document_urls = document_urls[:url_limit]
        
    output_paths = list(
        map(
            lambda url: os.path.join(
                output_path, url.split("/")[-1] + f".{output_type}"
            ),
            document_urls,
        )
    )

    if not raw_download_dir:
        raw_download_dir = os.path.join(output_path, "downloads")
    expand_outdir_and_mkdir(raw_download_dir)

    downloader = WebDocumentDownloader(raw_download_dir)
    base_url = urlparse(url).path[0]
    iterator = WebDocumentIterator(base_url=base_url)
    extractor = WebDocumentExtractor(base_url=base_url)

    output_format = {
        "text": str,
        "title": str,
        "id": str,
        "url": str,
        "language": str,
        "source_id": str,
        "file_name": str,
    }
    
    dataset = download_and_extract(
        document_urls,
        output_paths,
        downloader,
        iterator,
        extractor,
        output_format,
        output_type=output_type,
        keep_raw_download=keep_raw_download,
        force_download=force_download,
        filename_col="file_name"
    )

    return dataset


if __name__ == "__main__":
    
    url = "https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html"
    
    dataset = download_web_documents(url, "./../data")
    
    for data in dataset:
        print(data)
    
    
    
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
import datetime

from bs4 import BeautifulSoup
from src.download.download_utils import get_document_url_list, extract_code_blocks, extract_image_urls, extract_url_blocks
import tqdm


class WebDocumentDownloader():
    
    def __init__(self, target_url, download_dir, chunk_size, chunk_overlap):
        self._download_dir = download_dir
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._urls = get_document_url_list(target_url)
        self._base_url = urlparse(target_url).path[0]
        self._subject = urlparse(target_url).path[1]
        self._counter = 0
        
    def __len__(self):
        return len(self._urls)
        
    def run(self):
        
        for url in tqdm.tqdm(self._urls):
            doc_data, meta_data = self.prepare_document_for_embedding(url, self._chunk_size, self._chunk_overlap)
            yield doc_data, meta_data
        
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

            stdout, stderr = subprocess.DEVNULL, subprocess.DEVNULL
            p = subprocess.run(
                cmd,
                stdout=stdout,
                stderr=stderr,
            )
            if p.returncode != 0:
                print(f"Failed to download {url} to {output_file}")
                
        return self.extract(*self.iterate(output_file))
        
        

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
        
        # 제거할 태그 목록
        unwanted_tags = [
            "script", "style", "nav", "header", "footer", "aside",
            "link", "meta", "input", "button"
        ]
        
        # 각 section 태그(단, id 속성이 있는 태그)를 순회
        sections = article.find_all("section", id=True)

        for section in sections:
            
            # 불필요한 태그 모두 제거
            for tag_name in unwanted_tags:
                for tag in section.find_all(tag_name):
                    tag.decompose()
            self._counter += 1

            section_id = section.get("id")
            # 제목은 해당 section 내의 h1, h2, h3 등에서 추출
            header_tag = section.find(["h1", "h2", "h3", "h4"])
            header = header_tag.get_text(strip=True) if header_tag else ""

            # 섹션의 전체 텍스트 콘텐츠 추출 (필요에 따라 header를 제거할 수도 있음)
            # content = section.get_text(separator="\n", strip=True)
            content = str(section)
            # URL은 base_url과 section_id를 조합해 생성 (필요에 따라 더 복잡하게 조합 가능)
            url = f"{self._base_url}/{ori_url}#{quote(section_id)}"

            return ({
                "section_id": section_id,
                "header": header,
                "section_url": url,
                "source_id": bname,
                "subject": self._subject
            }, content)
            
    def extract(self, meta, content):
        
        soup = BeautifulSoup(content, "html.parser")
        
        url_blocks = extract_url_blocks(soup, self._base_url)
        img_blocks = extract_image_urls(soup, self._base_url)
        code_blocks = extract_code_blocks(soup)
    
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)
        
        extracted_meta = {
                "code": code_blocks, "urls": url_blocks, "images": img_blocks
            }
        extracted_meta.update(meta)
        
        return extracted_meta, text
        
    def prepare_document_for_embedding(self, url: str, chunk_size: int = 2000, chunk_overlap: int = 200):
        
        meta_origin, text = self.download(url)
        
        code_blocks, url_blocks, image_blocks = meta_origin["code"], meta_origin["urls"], meta_origin["images"]
        meta_origin.pop("code")
        meta_origin.pop("urls")
        meta_origin.pop("images")
        
         # 자연어 텍스트를 원하는 청크 크기로 분할합니다.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " "]
        )
        chunks = splitter.split_text(text)
        
        # 각 청크와 관련된 코드 블록(만약 청크 내에 코드 관련 설명이 있었다면)을 메타데이터에 추가합니다.
        # 여기서는 간단하게, 코드 블록이 원래 문서 내에 있었던 위치(예: 주변 문단)에 해당하는지 여부를 확인하는 로직이 추가될 수 있습니다.
        # 예제에서는 단순히, 전체 코드 블록들을 메타데이터로 저장하는 방식입니다.
        meta_dataset = []
        doc_dataset = []
        for chunk in chunks:

            doc_dataset.append(chunk)
            # add more metadata
            current_time = datetime.datetime.now().isoformat() 
            metadata = {"url": url, "created_at": current_time}
            metadata.update(meta_origin)
            
            associated_codes = {}
            for marker, code in code_blocks.items():
                if marker in chunk:
                    associated_codes[marker] = code
                    
            if associated_codes:
                metadata["code"] = associated_codes
                
                for keys in associated_codes.keys():
                    code_blocks.pop(keys)
                    
            
            associated_urls = {}
            for marker, url_val in url_blocks.items():
                if marker in chunk:
                    associated_urls[marker] = url_val

            if associated_urls:
                metadata["urls"] = associated_urls
                
                for key in associated_urls.keys():
                    url_blocks.pop(key)
                    
                    
            associated_images = {}
            for marker, image_val in image_blocks.items():
                if marker in chunk:
                    associated_images[marker] = image_val
            if associated_images:
                metadata["images"] = associated_images
                for key in associated_images.keys():
                    image_blocks.pop(key)

            meta_dataset.append(metadata)
        
        return doc_dataset, meta_dataset
            
            
            
if __name__ == "__main__":
    
    url = "https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html"
    downloader = WebDocumentDownloader("https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html", "./data")
    for doc_data, meta_data in downloader.run():
        print("Document Data:", doc_data)
        # print("Metadata:", meta_data)
    
    
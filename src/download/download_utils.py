import re
from typing import Union, Tuple, Dict, List
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
import copy
from src.config import CODE_BLOCK_MARKER, URL_BOLCK_MARKER, IMAGE_BLOCK_MARKER, MIN_IMAGE_SIZE
import datetime

from urllib.parse import urljoin
import html


def is_valid_url(url):
    try:
        response = requests.head(url, allow_redirects=True)

        return response.status_code >= 200 and response.status_code < 300
    except requests.RequestException as e:
        print(f"URL 확인 중 오류 발생: {e}")
        return False
    
    
def get_document_url_list(url):
    # URL로부터 HTML 코드 읽어오기
    response = requests.get(url)
    response.raise_for_status()
    html_content = response.text

    # BeautifulSoup 객체 생성
    soup = BeautifulSoup(html_content, "html.parser")

    # "Table of Contents"가 있는 nav 태그 찾기
    nav = soup.find("nav", {"aria-label": "Table of Contents"})
    
    # root URL 결정: 인자로 주어진 url의 마지막 슬래시(/) 전까지의 경로
    if url.endswith('/'):
        root = url
    else:
        root = url.rsplit('/', 1)[0] + '/'

    urls = []
    print("Start to get urls ...")
    if nav:
        # 'reference internal' 클래스를 가진 모든 <a> 태그를 순회
        for a_tag in nav.find_all("a", class_="reference internal"):
            href = a_tag.get("href")
            # href가 '#'인 경우, 현재 페이지의 파일명으로 대체
            if href == "#":
                href = url.split("/")[-1]
            # root와 href 결합
            full_url = urljoin(root, href)
            if is_valid_url(full_url):
                urls.append(full_url)
                
    print(f"Get Urls: {len(urls)}")
    return urls

def extract_image_urls(soup: BeautifulSoup, base_url: str, min_dimension:int=100):
    
    image_index = 0
    image_blocks = {}
    for img in soup.find_all("img", src=True):
        # 이미지 크기 판단: width와 height 속성이 존재하는 경우 이를 기준으로 판단
        width = img.get("width")
        height = img.get("height")
        if width is not None and height is not None:
            try:
                width_val = int(width)
                height_val = int(height)
            except ValueError:
                width_val = None
                height_val = None
            if width_val is not None and height_val is not None and (width_val < min_dimension or height_val < min_dimension):
                continue  # 아이콘 등 작은 이미지는 건너뜀
        marker = f"{IMAGE_BLOCK_MARKER}{image_index}"
        image_index += 1
        if not img["src"].startswith("https://"):
            img_src = base_url + "/" + img["src"]
        else:
            img_src = img["src"]
            
        image_info = {"src": img_src}
        
        if img.has_attr("alt"):
            image_info["alt"] = img["alt"]
        if width is not None:
            image_info["width"] = width
        if height is not None:
            image_info["height"] = height
        image_blocks[marker] = image_info
        img.replace_with(marker)
        
    return image_blocks

def extract_code_blocks(soup: BeautifulSoup) -> Dict:
    code_blocks = {}
    index = 0
    # 먼저 <pre> 태그에서 코드 예제를 추출
    for pre in soup.find_all("pre"):
        marker = f"{CODE_BLOCK_MARKER}{index}"
        code_blocks[marker] = html.unescape(pre.get_text())  # 코드 포맷 보존
        pre.replace_with(marker)
        index += 1
    # <pre> 안에 있지 않은 <code> 태그들도 추출 (인라인 코드 포함)
    for code in soup.find_all("code"):
        marker = f"{CODE_BLOCK_MARKER}{index}"
        code_blocks[marker] = html.unescape(code.get_text())
        code.replace_with(marker)
        index += 1
    return code_blocks

def extract_url_blocks(soup: BeautifulSoup, base_url: str) -> Dict:
    
    # URL 정보 보존: <a> 태그의 href를 고유 마커로 치환
    url_blocks = {}
    for i, a in enumerate(soup.find_all("a", href=True)):
        marker = f"{URL_BOLCK_MARKER}{i}"
        if a["href"].startswith("#"):
            url_blocks[marker] = base_url + a["href"]  # URL 정보 보존
        elif a["href"].startswith("https://"):
            url_blocks[marker] = base_url + "/" + a["href"]
        else:
            url_blocks[marker] = a["href"]
            
        a.replace_with(marker)
        
    return url_blocks

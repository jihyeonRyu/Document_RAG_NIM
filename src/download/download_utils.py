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



def html_document_loader(url: Union[str, bytes]) -> Tuple[str, Dict, Dict, Dict]:

    try:
        response = requests.get(url)
        html_content = response.text
    except Exception as e:
        print(f"Failed to load {url} due to exception {e}")
        return ""
    
    try:
        # BeautifulSoup으로 HTML 파싱
        soup = BeautifulSoup(html_content, "html.parser")
        
        # <main id="main-content"> 영역이 있다면 그 영역만 대상으로 함 (메인 콘텐츠 최적화)
        main_content = soup.find("main", {"id": "main-content"})
        if main_content:
            soup = main_content

        # 불필요한 태그 제거: script, style, nav, header, footer, aside, noscript 등
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
            tag.decompose()
        
        # 코드 예제 보존: <pre> 태그를 고유 마커로 치환
        code_blocks = {}
        for i, pre in enumerate(soup.find_all("pre")):
            marker = f"{CODE_BLOCK_MARKER}{i}"
            code_blocks[marker] = pre.get_text()  # 코드 포맷 보존
            pre.replace_with(marker)
            
        
        # URL 정보 보존: <a> 태그의 href를 고유 마커로 치환
        url_blocks = {}
        for i, a in enumerate(soup.find_all("a", href=True)):
            marker = f"{URL_BOLCK_MARKER}{i}"
            url_blocks[marker] = a["href"]  # URL 정보 보존
            a.replace_with(marker)
            
        
        # 이미지 정보 보존: <img> 태그의 src (및 alt 속성) 정보를 고유 마커로 치환
        image_blocks = extract_image_urls(soup, MIN_IMAGE_SIZE)
        
        # 수정된 HTML에서 텍스트 추출
        text = soup.get_text()
        
        # 일반 텍스트의 불필요한 공백 및 다중 줄바꿈 제거
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n", "\n", text)
        
        return text.strip(), code_blocks, url_blocks, image_blocks
    except Exception as e:
        print(f"Exception {e} while loading document")
        return ""
    

def process_document_for_embedding(url: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> Tuple[List, List]:
    # HTML을 로드할 때, 코드 블록은 별도로 추출해서 반환합니다.
    no_code_text, code_blocks, url_blocks, image_blocks = html_document_loader(url)
    
    # 자연어 텍스트를 원하는 청크 크기로 분할합니다.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "]
    )
    chunks = splitter.split_text(no_code_text)
    
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
    

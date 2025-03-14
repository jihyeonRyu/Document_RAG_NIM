"""

https://docs.nvidia.com/<DOCUMENT>  에 있는 Table Content html 리스트를 가져오는 코드

"""

import sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import argparse
from src.config import DATA_PATH
import os
import requests
import tqdm

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

def save_url_list_to_file(urls, filename):
    
    filepath = os.path.join(DATA_PATH, filename)
    with open(filepath, 'w') as f:
        for url in urls:
            f.write(url + "\n")
    
    print(f"Save Urls to {filepath}")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('url', type=str, help="Root URL of the document")
    parser.add_argument('output', type=str, default="urls", help="output txt file name")
    
    args = parser.parse_args()
    
    if not args.output.endswith('.txt'):
        args.output += ".txt"
    
    urls = get_document_url_list(args.url)
    save_url_list_to_file(urls, args.output)
    
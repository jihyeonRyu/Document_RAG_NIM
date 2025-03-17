import os

ROOT_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT_PATH, "data")

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
    
    
CODE_BLOCK_MARKER = "__CODE_EXAMPLE__"
URL_BOLCK_MARKER = "__URL__"
IMAGE_BLOCK_MARKER = "__IMAGE__"
CHUNK_SIZE=2000
CHUNK_OVERLAP=200
MIN_IMAGE_SIZE=100
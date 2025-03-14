import os

ROOT_PATH = os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT_PATH, "data")

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
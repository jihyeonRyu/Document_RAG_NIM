# RAG Pipeline (Local NIM + Langchain + Milvus)

## Object: NVIDIA Document RAG Pipeline 
- https://docs.nvidia.com/<DOCUMENT> 

- [NIM Document](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html)

## Environment Setting

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```

conda create -n rag python=3.12
conda activate rag

pip install -r requirements.txt


sudo apt-get update
sudo apt-get install build-essential

pip install --extra-index-url https://pypi.nvidia.com nemo-curator[all]

```

## Install Milvus Standalone

- [Milvus Doc](https://milvus.io/docs/ko/install_standalone-docker-compose-gpu.md)

```
wget https://github.com/milvus-io/milvus/releases/download/v2.5.6/milvus-standalone-docker-compose-gpu.yml -O docker-compose.yml
```
- docker-compose.yml 파일에 device_ids 에 gpu id를 추가/수정

```
...
standalone:
  ...
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            capabilities: ["gpu"]
            device_ids: ["0"]
...
```

```
# 시작
docker compose up -d
# 끄기
docker compose down

rm -rf volumes

```

## Launch LLM NIM
- [Explore NGC Catalog](https://catalog.ngc.nvidia.com/?filters=&orderBy=weightPopularDESC&query=&page=&pageSize=)

- [NIM Launch Parameters](https://docs.nvidia.com/nim/large-language-models/latest/configuration.html#environment-variables)

- LLM: nvcr.io/nim/meta/llama-3.2-1b-instruct:1.8.1
- Embedding: nvcr.io/nim/nvidia/nv-embedqa-e5-v5:1.1.0
    - https://build.nvidia.com/nvidia/embed-qa-4?snippet_tab=LangChain
- Rerank: nvcr.io/nim/nvidia/llama-3.2-nv-rerankqa-1b-v2:1.3.1

```
export LOCAL_NIM_CACHE=cache
mkdir -p $LOCAL_NIM_CACHE
chmod 777 $LOCAL_NIM_CACHE
export IMG_NAME=nvcr.io/nim/meta/llama-3.2-1b-instruct:1.8.1
export CONTAINER_NAME=llama-3.2-1b-instruct
 
# Multi-GPU
docker run -d --name=$CONTAINER_NAME \
  --gpus all \
  --shm-size=16GB \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u $(id -u) \
  -p 8000:8000 \
  $IMG_NAME
 
# Single-GPU
docker run -d --name=$CONTAINER_NAME \
  --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u $(id -u) \
  -p 8000:8000 \
  $IMG_NAME

```
## Test NIM

```
sudo apt-get install jq

# check model "id" for "model"
curl -s http://0.0.0.0:8000/v1/models | jq
 
curl -s 'POST' \
    'http://0.0.0.0:8000/v1/completions' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "meta/llama-3.2-1b-instruct",
      "prompt": "What is the Machine Learning?",
      "max_tokens": 128
    }' | jq
```
# RAG Pipeline (Local NIM + Langchain + Milvus)
## Environment Setting

```
expert LOCAL_NIM_CACHE=nim_cache
mkdir $LOCAL_NIM_CACHE
chmod 777 $LOCAL_NIM_CACHE
 
pip install --upgrade pip
python -m venv rag_venv
 
source rag_venv/bin/activate
 
pip install -r requirements.txt
```

## Install Milvus Standalone

- [Milvus Doc](https://docs.nvidia.com/nim/large-language-models/latest/configuration.html#environment-variables)

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
- Rerank: nvcr.io/nim/nvidia/llama-3.2-nv-rerankqa-1b-v2:1.3.1

```
export CONTAINER_NAME=llama-3.2-1b-instruct
 
# Multi-GPU
docker run -d --name=$CONTAINER_NAME \
  --runtime=nvidia \
  --gpus all \
  --shm-size=16GB \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u $(id -u) \
  -p 8000:8000 \
  $IMG_NAME
 
# Single-GPU
docker run -d --name=$CONTAINER_NAME \
  --runtime=nvidia \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u $(id -u) \
  -p 8000:8000 \
  $IMG_NAME

```
- Test NIM

```
curl -s http://0.0.0.0:8000/v1/models | jq
 
curl -X 'POST' \
    'http://0.0.0.0:8000/v1/completions' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "",
      "prompt": "What is the Machine Learning?",
      "max_tokens": 128
    }'
```
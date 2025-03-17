from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from typing import List, Tuple
from src.download.web_document import WebDocumentDownloader
from langchain_milvus import BM25BuiltInFunction, Milvus
from src.config import ROOT_PATH, CHUNK_SIZE, CHUNK_OVERLAP, TEMPERATURE, TOP_P, MAX_TOKENS

from langchain_core.documents import Document

class EmbeddingNIM():
    
    def __init__(self, llm_model: str, emb_model: str, api_key: str=None):
        
        self._embedding = NVIDIAEmbeddings(model=emb_model, truncate="NONE", api_key=api_key)

        MILVUS_URI = "http://0.0.0.0:19530"

        self._vectorstore = Milvus(
            embedding_function=self._embedding,
            collection_name="NVIDIA_Document",
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            connection_args={"uri": MILVUS_URI},
            index_params={"index_type": "GPU_IVF_PQ", "metric_type": "IP", "params": {"nprobe": 5}},
            search_params=None,
            consistency_level="Strong",
            enable_dynamic_field=True,
            auto_id=False,
            drop_old=False,  # set to True if seeking to drop the collection with that name if it exists
        )
        
        self._llm = ChatNVIDIA(base_url="http://0.0.0.0:8000/v1", model=llm_model, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, top_p=TOP_P)
        
    
    def insert_embeddings(self, target_url: str, download_dir: str) -> bool:
        
        downloader = WebDocumentDownloader(target_url, download_dir, CHUNK_SIZE, CHUNK_OVERLAP)
        idx = 0
        for document, metadata in downloader.run():
            self._vectorstore.add_texts(document, metadata, partition_names="subject", ids=[str(i+idx) for i in range(len(document))])
            idx += len(document)

    def search(self, query: str, top_k: int=3, condition : str = None) -> List[Tuple[Document, float]] :
        
        results = self._vectorstore.similarity_search_with_score(
                query, k=top_k, expr=condition
        )
        
        # for res, score in results:
        #     print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
            
        return results
    
    
if __name__ == "__main__":
    
    with open(f"{ROOT_PATH}/.env", 'r') as f:
        api_key = f.read()
    
    emb = EmbeddingNIM("meta/llama-3.2-1b-instruct", "NV-Embed-QA", api_key=api_key.strip())
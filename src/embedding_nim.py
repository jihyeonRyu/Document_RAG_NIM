from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from typing import List, Tuple
from src.dataset import process_document_for_embedding
from langchain_milvus import Milvus
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

from langchain_core.documents import Document

class EmbeddingNIM():
    
    def __init__(self, model: str, base_url: str, api_key: str):
        
        self._embedding = NVIDIAEmbeddings(model=model, truncate="NONE", api_key=api_key)

        URI = "http://localhost:19530"

        self._vectorstore = Milvus(
            embedding_function=self._embedding,
            connection_args={"uri": URI, "token": "root:Milvus", "db_name": "milvus_demo"},
            index_params={"index_type": "GPU_IVF_PQ", "metric_type": "IP", "params": {"nprobe": 5}},
            consistency_level="Strong",
            drop_old=False,  # set to True if seeking to drop the collection with that name if it exists
        )
        
    
    def insert_embeddings(self, urls: List[str]) -> bool:
        
        idx = 0
        for i, url in enumerate(urls):
            print("Inserting embeddings for document", url, " --> " , i)
            document, metadata = process_document_for_embedding(url, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            self._vectorstore.add_texts(document, metadata, ids=[str(i+idx) for i in range(len(document))])
            idx += len(document)

    def search(self, query: str, top_k: int=3, condition : str = None) -> List[Tuple[Document, float]] :
        
        results = self._vectorstore.similarity_search_with_score(
                query, k=top_k, expr=condition
        )
        
        # for res, score in results:
        #     print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
            
        return results
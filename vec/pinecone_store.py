import os
from dotenv import load_dotenv; load_dotenv()
from typing import List, Dict, Any
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LC_Pinecone
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INDEX = os.getenv("PINECONE_INDEX", "finflow")
DIM = 1536  # for text-embedding-3-small
METRIC = "cosine"

_pinecone_client = None
_openai_embeddings = None

def _pc() -> Pinecone:
    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return _pinecone_client

def _get_embeddings() -> OpenAIEmbeddings:
    global _openai_embeddings
    if _openai_embeddings is None:
        _openai_embeddings = OpenAIEmbeddings(model=os.getenv("EMBED_MODEL","text-embedding-3-small"))
    return _openai_embeddings

def delete_index():
    pc = _pc()
    names = [i.name for i in pc.list_indexes()]
    if INDEX in names:
        logging.info(f"Pinecone index '{INDEX}' found. Deleting index.")
        pc.delete_index(INDEX)
        logging.info(f"Pinecone index '{INDEX}' deleted.")
    else:
        logging.info(f"Pinecone index '{INDEX}' does not exist. No need to delete.")

def ensure_index():
    pc = _pc()
    names = [i.name for i in pc.list_indexes()]
    if INDEX not in names:
        logging.info(f"Pinecone index '{INDEX}' not found. Creating new index.")
        pc.create_index(
            name=INDEX,
            dimension=DIM,
            metric=METRIC,
            spec=ServerlessSpec(
                cloud=os.getenv("PINECONE_CLOUD","aws"),
                region=os.getenv("PINECONE_REGION","us-east-1")
            )
        )
    else:
        logging.info(f"Pinecone index '{INDEX}' already exists.")
    return pc.Index(INDEX)

def upsert(items: List[Dict[str, Any]]):
    idx = ensure_index()
    if not items:
        logging.warning(f"No items to upsert to Pinecone index '{INDEX}'. Skipping upsert.")
        return True
    logging.info(f"Upserting {len(items)} items to Pinecone index '{INDEX}'.")
    try:
        idx.upsert(vectors=items)  # [{id, values, metadata}]
        logging.info(f"Successfully upserted {len(items)} items.")
        return True
    except Exception as e:
        logging.error(f"Error during upsert to Pinecone: {e}")
        return False

def query(vec, k=5, flt:Dict[str,Any]|None=None):
    idx = ensure_index()
    logging.info(f"Querying Pinecone index '{INDEX}' with k={k} and filter={flt}.")
    try:
        results = idx.query(vector=vec, top_k=k, include_metadata=True, filter=flt or {})
        logging.info(f"Query returned {len(results.matches)} results.")
        return results
    except Exception as e:
        logging.error(f"Error during query from Pinecone: {e}")
        return None

def vectorstore():
    ensure_index()
    emb = _get_embeddings()
    return LC_Pinecone.from_existing_index(index_name=INDEX, embedding=emb)

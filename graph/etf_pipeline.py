from typing import TypedDict, List, Dict, Any
import os, re, requests, tempfile, json
from langgraph.graph import StateGraph, END
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vec.pinecone_store import upsert, vectorstore
import pypdf
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class S(TypedDict):
    files: List[str]                 # 本地PDF路径（一个或两个）
    sources: List[str]               # 每个PDF的来源URL或文件名
    chunks: List[Dict[str,Any]]      # [{id,text,meta}]
    q: str
    retrieved: List[Dict[str,Any]]
    extracted: Dict[str,Any]         # 结构化抽取
    answer_md: str

def _load_pdf(path: str) -> str:
    text = ""
    try:
        r = pypdf.PdfReader(path)
        logging.info(f"PDF '{path}' has {len(r.pages)} pages.")
        extracted_texts = []
        for i, p in enumerate(r.pages):
            page_text = p.extract_text()
            if page_text:
                extracted_texts.append(page_text)
                logging.debug(f"Extracted text from page {i+1} of {path}. Length: {len(page_text)} characters.")
            else:
                logging.warning(f"No text extracted from page {i+1} of PDF: {path} using pypdf.")
        
        text = "\n".join(extracted_texts)
        logging.info(f"Total extracted text from {path} using pypdf. Length: {len(text)} characters.")
        
        if not text.strip():
            logging.warning(f"No overall text extracted from PDF: {path} using pypdf. Attempting with pdftotext.")
            try:
                text = subprocess.check_output(["pdftotext", "-raw", path, "-"], encoding="utf-8", errors="ignore")
                logging.info(f"Total extracted text from {path} using pdftotext. Length: {len(text)} characters.")
                if not text.strip():
                    logging.warning(f"No overall text extracted from PDF: {path} using pdftotext.")
            except FileNotFoundError:
                logging.error("pdftotext not found. Please ensure poppler-utils is installed.")
                text = ""
            except Exception as e:
                logging.error(f"Error loading PDF {path} with pdftotext: {e}")
                text = ""
        return text
    except Exception as e:
        logging.error(f"Error loading PDF {path} with pypdf: {e}")
        
        # Fallback to pdftotext if pypdf fails entirely
        logging.warning(f"pypdf failed for {path}. Attempting with pdftotext as a fallback.")
        try:
            text = subprocess.check_output(["pdftotext", "-raw", path, "-"], encoding="utf-8", errors="ignore")
            logging.info(f"Total extracted text from {path} using pdftotext (fallback). Length: {len(text)} characters.")
            if not text.strip():
                logging.warning(f"No overall text extracted from PDF: {path} using pdftotext (fallback).")
            return text
        except FileNotFoundError:
            logging.error("pdftotext not found. Please ensure poppler-utils is installed.")
            return ""
        except Exception as e:
            logging.error(f"Error loading PDF {path} with pdftotext (fallback): {e}")
            return ""

def _download(url: str) -> str:
    # 简易下载：把 PDF 存到临时文件
    b = requests.get(url, timeout=30).content
    fd, path = tempfile.mkstemp(suffix=".pdf"); os.write(fd, b); os.close(fd)
    return path

def node_ingest(state: S) -> S:
    texts = []
    for idx, src in enumerate(state["sources"]):
        path = state["files"][idx] if state["files"][idx] else _download(src)
        text = _load_pdf(path)
        texts.append((text, src))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = []
    for i,(t,src) in enumerate(texts):
        if not t.strip():
            logging.warning(f"Skipping chunking for empty text from source: {src}")
            continue
        for j, ch in enumerate(splitter.split_text(t)):
            chunks.append({"id": f"{i}:{j}", "text": ch, "meta": {"doc": src, "doc_id": i}})
    logging.info(f"Created {len(chunks)} chunks from {len(texts)} documents.")
    state["chunks"] = chunks
    return state

def node_embed_upsert(state:S)->S:
    emb = OpenAIEmbeddings(model=os.getenv("EMBED_MODEL","text-embedding-3-small"))
    valid_chunks = [c for c in state["chunks"] if c.get("text") and c["text"].strip()]
    if not valid_chunks:
        logging.warning("No valid chunks with text found for embedding and upserting. Skipping.")
        state["chunks"] = [] # Clear chunks if none are valid
        return state
    items = [{"id": c["id"], "values": emb.embed_query(c["text"]), "metadata": {**c["meta"], "text": c["text"]}} for c in valid_chunks]
    upsert(items)
    state["chunks"] = valid_chunks # Update state with only valid chunks
    return state

def node_retrieve(state:S)->S:
    vs = vectorstore()
    # 按 doc_id 过滤；单文档QA取第0个，双文档对比时也各取若干
    doc_id = 0
    docs_scores = vs.similarity_search_with_score(state["q"], k=6, filter={"doc_id": doc_id})
    state["retrieved"] = [{"text": d.page_content, "meta": d.metadata, "score": s} for d,s in docs_scores]
    return state

def node_extract_metrics(state:S)->S:
    """从已检索片段中抽取 ETF 关键指标（JSON）"""
    llm = ChatOpenAI(model=os.getenv("GEN_MODEL","gpt-5-nano"))
    ctx = "\n\n".join([r["text"] for r in state["retrieved"]])
    prompt = f"""From the following ETF fact-sheet snippets, extract a JSON with:
{{
 "expense_ratio": "e.g., 0.03%",
 "aum": "e.g., $450B" or null,
 "inception_date": "YYYY-MM-DD or original format",
 "benchmark": "index name",
 "top_holdings_sample": ["...","..."] (up to 5 if visible)
}}
Only use values present in the context. If unknown, use null.
Context:
{ctx}"""
    llm_output_content = llm.invoke(prompt).content
    try:
        state["extracted"] = json.loads(llm_output_content)
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from LLM output: {llm_output_content}")
        state["extracted"] = {} # Fallback to empty dict on error
    return state

def node_answer(state:S)->S:
    llm = ChatOpenAI(model=os.getenv("GEN_MODEL","gpt-5-nano"))
    ctx = "\n\n".join([f"- {r['text']}" for r in state["retrieved"][:5]])
    q = state["q"]
    prompt = f"""Answer the user's question about the ETF using the context. 
Cite clause numbers 1..N by order (e.g., [1], [2]).
Context:
{ctx}

Question: {q}
Short markdown answer:"""
    state["answer_md"] = llm.invoke(prompt).content
    return state

# Graph wiring
g = StateGraph(S)
g.add_node("ingest", node_ingest)
g.add_node("embed_upsert", node_embed_upsert)
g.add_node("retrieve", node_retrieve)
g.add_node("extract", node_extract_metrics)
g.add_node("answer", node_answer)
g.set_entry_point("ingest")
g.add_edge("ingest","embed_upsert")
g.add_edge("embed_upsert","retrieve")
g.add_edge("retrieve","extract")
g.add_edge("extract","answer")
g.add_edge("answer", END)
etf_app = g.compile()

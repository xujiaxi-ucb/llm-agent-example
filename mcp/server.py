import base64, pandas as pd
from model_context_protocol.server import Server
from graph.etf_pipeline import etf_app

srv = Server("etf-facts-mcp")

@srv.tool()
def load_factsheet(url:str):
    """Load a factsheet PDF by URL and return extracted key metrics."""
    state = {"files": [""], "sources":[url], "q":"Extract key metrics", "chunks":[], "retrieved":[]}
    out = etf_app.invoke(state)
    return {"source": url, "metrics": out.get("extracted", {})}

@srv.tool()
def query_factsheet(url:str, question:str):
    state = {"files": [""], "sources":[url], "q":question, "chunks":[], "retrieved":[]}
    out = etf_app.invoke(state)
    return {"answer_md": out["answer_md"]}

@srv.tool()
def compare_factsheets(url_a:str, url_b:str):
    r1 = load_factsheet(url_a); r2 = load_factsheet(url_b)
    return {"A": r1, "B": r2}

if __name__ == "__main__":
    srv.run()

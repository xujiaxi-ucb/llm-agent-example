import gradio as gr, os, pandas as pd, json, logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()
from graph.etf_pipeline import etf_app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_single(pdf_file, question):
    sources = [pdf_file.name if pdf_file else "data/spy.pdf"]
    files   = [pdf_file.name if pdf_file else "data/spy.pdf"]
    state = {"files": files, "sources": sources, "q": question, "chunks": [], "retrieved": []}
    out = etf_app.invoke(state)
    meta = out.get("extracted", {})
    df = pd.DataFrame([meta])
    return df, out["answer_md"]

def run_compare(pdf_file_a, pdf_file_b):
    sources = [pdf_file_a.name if pdf_file_a else "data/spy.pdf",
               pdf_file_b.name if pdf_file_b else "data/voo.pdf"]
    files   = [pdf_file_a.name if pdf_file_a else "data/spy.pdf", pdf_file_b.name if pdf_file_b else "data/voo.pdf"]
    # Ask the same question for each document to extract metrics
    rows = []
    for i in range(2):
        state = {"files":[files[i]], "sources":[sources[i]], "q":"Extract key metrics", "chunks":[], "retrieved":[]}
        out = etf_app.invoke(state)
        extracted_data = out.get("extracted", {})
        
        # If extracted_data is a string, attempt to parse it as JSON
        if isinstance(extracted_data, str):
            try:
                extracted_data = json.loads(extracted_data)
            except json.JSONDecodeError:
                logging.error(f"Failed to decode JSON from extracted data: {extracted_data}")
                extracted_data = {} # Fallback to empty dict on error
        
        meta = extracted_data
        meta["source"] = sources[i]
        rows.append(meta)
    df = pd.DataFrame(rows)
    # Generate comparison summary
    def safe(x): return x if isinstance(x,str) else str(x)
    summary = f"""**Compare**
- Source A: {safe(rows[0].get('source'))}
- Source B: {safe(rows[1].get('source'))}

**Diff highlights**
- Expense ratio: {rows[0].get('expense_ratio')} vs {rows[1].get('expense_ratio')}
- Benchmark: {rows[0].get('benchmark')} vs {rows[1].get('benchmark')}
- Inception: {rows[0].get('inception_date')} vs {rows[1].get('inception_date')}
"""
    return df, summary

with gr.Blocks() as demo:
    gr.Markdown("# ETF Fact-Sheet QA & Comparator (LangGraph × Pinecone × Gradio)")
    with gr.Tab("Single PDF QA"):
        f = gr.File(label="Upload PDF")
        gr.Markdown("If no file is uploaded, `data/spy.pdf` will be used by default.")
        q = gr.Textbox(value="What is the expense ratio?", label="Question")
        btn = gr.Button("Run")
        table = gr.Dataframe(interactive=False)
        md = gr.Markdown()
        btn.click(run_single, inputs=[f,q], outputs=[table, md])

    with gr.Tab("Compare two PDFs"):
        f1 = gr.File(label="PDF A")
        f2 = gr.File(label="PDF B")
        gr.Markdown("If no files are uploaded, `data/spy.pdf` and `data/voo.pdf` will be used by default for comparison.")
        btn2 = gr.Button("Compare")
        table2 = gr.Dataframe(interactive=False); md2 = gr.Markdown()
        btn2.click(run_compare, inputs=[f1,f2], outputs=[table2, md2])

app = demo

if __name__ == "__main__":
    demo.launch(share=True)

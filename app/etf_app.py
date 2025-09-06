import gradio as gr, os, pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()
from graph.etf_pipeline import etf_app

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
    # 逐文档提问同一问题以便抽取指标
    rows = []
    for i in range(2):
        state = {"files":[files[i]], "sources":[sources[i]], "q":"Extract key metrics", "chunks":[], "retrieved":[]}
        out = etf_app.invoke(state)
        meta = out.get("extracted", {})
        meta["source"] = sources[i]
        rows.append(meta)
    df = pd.DataFrame(rows)
    # 生成对比摘要
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
        q = gr.Textbox(value="What is the expense ratio?", label="Question")
        btn = gr.Button("Run")
        table = gr.Dataframe(interactive=False)
        md = gr.Markdown()
        btn.click(run_single, inputs=[f,q], outputs=[table, md])

    with gr.Tab("Compare two PDFs"):
        f1 = gr.File(label="PDF A")
        f2 = gr.File(label="PDF B")
        btn2 = gr.Button("Compare")
        table2 = gr.Dataframe(interactive=False); md2 = gr.Markdown()
        btn2.click(run_compare, inputs=[f1,f2], outputs=[table2, md2])

app = demo

if __name__ == "__main__":
    demo.launch(share=True)

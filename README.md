设计：条款切分 → 向量化入 Pinecone（serverless）→ 元数据过滤检索 → 生成引用回答 → RAGAS 评测闭环。

索引：dimension 与度量（cosine/euclidean/dotproduct）按模型选择；Serverless 索引创建/管理与过滤、混合检索参见官方指南。

编排：LangGraph 节点式状态机；复杂场景可用 LCEL Runnable 做并行/流式。

前端：Gradio Blocks。

开放：MCP 工具化输出，便于被 IDE/桌面/其他 Agent 直接连接。
import os
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
# 1. 导入必要的工具（注意：这里用了最新的库名，避免警告）
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from env_utils import DASHSCOPE_API_KEY
from my_llm import dashscope_llm

# ==========================================
# 第一步：准备“脑子” (DeepSeek LLM)
# ==========================================
# 请确保你的 API Key 是正确的


# ==========================================
# 第二步：处理“书本” (文档加载与切分)
# ==========================================
# 加上 encoding="utf-8" 防止中文乱码
loader = TextLoader('../Book/西游记解读.txt', encoding='utf-8')
docs = loader.load()

# 切分成 500 字的小块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
print(f"✅ 文档加载成功，已切分成 {len(splits)} 个段落")

# ==========================================
# 第三步：建立“索引” (向量化与存储)
# ==========================================
# 初始化本地 Embedding 模型
embeddings_model = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=DASHSCOPE_API_KEY# 👈 填入你的通义 API Key
)

# 构建向量数据库（注意这里是 embedding=...）
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings_model)
print("✅ 向量数据库构建完成")

# ==========================================
# 第四步：构建“链” (把所有东西串起来)
# ==========================================
# 使用 from_chain_type 是最标准的方法
qa_chain = RetrievalQA.from_chain_type(
    llm=dashscope_llm,
    chain_type="stuff", # 代表把检索到的资料直接塞给 AI
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}) # 每次只找最相关的 3 段
)

# ==========================================
# 第五步：提问并获取结果
# ==========================================
query = "西游记的主线剧情是关于谁的？"

# 使用 invoke 执行，它会返回一个字典
response = qa_chain.invoke({"query": query})

print("\n" + "="*20)
print("🤖 AI 的回答：")
# 我们只打印字典里的 'result' 部分，这样就不会把原文都打印出来了
print(response["result"])
print("="*20)
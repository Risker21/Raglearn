from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from env_utils import DASHSCOPE_API_KEY
from my_llm import dashscope_llm

# 选择“翻译官” (Embedding)
embeddings_model =  DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=DASHSCOPE_API_KEY
)
# 加载文档
loader = TextLoader("../Book/西游记解读.txt", encoding='utf-8')
data = loader.load()  # 加载文档
print("文档加载加载完成")

# 文本分割器
text_spliter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
all_splits = text_spliter.split_documents(data)
print("文档分割完成")
# 向量数据库
vectorstore = Chroma.from_documents(
    documents=all_splits, # 文本块
    embedding=embeddings_model, # 选择“翻译官” (Embedding)
    persist_directory="./my_db",
)

print("向量数据库创建完成")

template = """你是一个基于文档的助手。请仅使用以下提供的【背景资料】来回答问题。
如果你不知道答案，就说不知道，不要编造。

【背景资料】：{context}

【问题】：{question}
【回答】："""

QA_PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
print("提示模板创建完成")


# 6. 构建 RAG 链并测试
# ==========================================
qa_chain = RetrievalQA.from_chain_type(
    llm=dashscope_llm, # 选择“翻译官” (LLM)
    chain_type="stuff", # 合并所有文本块
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), # 每次找最相关的 3 块
    chain_type_kwargs={"prompt": QA_PROMPT} # 自定义提示模板
)
print("RAG链创建完成")

# 开始提问！
query = "西游记中的大师兄是谁？"
response = qa_chain.invoke({"query": query})
print("RAG链调用完成")

print("\n--- RAG 回答结果 ---")
print(response["result"])
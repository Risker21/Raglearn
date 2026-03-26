from init_llm.llm_factory import DashScopeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from env_utils import DASHSCOPE_API_KEY

loader = TextLoader("../Book/西游记解读.txt", encoding='utf-8')
docs = loader.load()
# print(docs)

text_spliter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
all_text_spliter = text_spliter.split_documents(docs)

embeddings_model = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)

vectorstore =  Chroma.from_documents(
    documents=all_text_spliter,  # 文档列表
    embedding=embeddings_model,  # 向量嵌入模型
    persist_directory="./test_db",  # 持久化目录
)

template = """
你是一个基于文档的助手。请仅使用以下提供的【背景资料】来回答问题。
【文档】:{context}，
【问题】:{question}，
【回答】
"""
prompt_template = PromptTemplate(template=template, input_variables=["context","question"])

query = "西游记的主线剧情是关于谁的？"
initial_docs = vectorstore.similarity_search(query,k=10)

scored_docs = []
for doc in initial_docs:
    score = 0
    if "取经" in doc.page_content:
        score += 5
    if "师徒" in doc.page_content:
        score += 3
    scored_docs.append((doc,score))

scored_docs.sort(key=lambda x: x[1], reverse=True) # 按分数从高到低排序

final_docs = [doc for doc,score in scored_docs[:3]]
print(final_docs)
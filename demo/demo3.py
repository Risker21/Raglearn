from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from env_utils import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL
from my_llm import dashscope_llm
# 步骤 1：加载原始文档（读取文本文件）
loader = TextLoader("../Book/红楼梦.txt", encoding='utf-8')
docs = loader.load()
# print(docs)

# 步骤 2：文本分块（切分长文本）
# RecursiveCharacterTextSplitter
text_spliter =  RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)  # chunk_size: 每个文档的最大字符数，chunk_overlap: 每个文档之间的重叠字符数
all_text_spliter = text_spliter.split_documents(docs)
# print(len(all_text_spliter))

# 步骤 3：初始化向量嵌入模型
embeddings_model = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=DASHSCOPE_API_KEY,
)


# 步骤 4：创建并持久化向量数据库
vectorstore =  Chroma.from_documents(
    documents=all_text_spliter,  # 文档列表
    embedding=embeddings_model,  # 向量嵌入模型
    persist_directory="./chroma_db",  # 持久化目录
)

# 步骤 5：定义提示模板
template="""
你是一个基于文档的助手。请仅使用以下提供的【背景资料】来回答问题。
如果你不知道答案，就说不知道，不要编造。
【文档】:{context}，
【问题】:{question}，
【回答】
"""
# 步骤 6：创建提示模板实例
prompt_template = PromptTemplate(template=template, input_variables=["context","question"])
# 步骤 7：创建 RetrievalQA 链实例
qa_chain = RetrievalQA.from_chain_type(
    llm=dashscope_llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k":10}),
    chain_type_kwargs={"prompt": prompt_template},
)

# 步骤 8：执行查询并获取响应
query = "大师兄是谁以及他擅长什么"
response = qa_chain.invoke({"query": query})
print(response["result"])
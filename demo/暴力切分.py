from init_llm.llm_factory import DashScopeEmbeddings
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from env_utils import DASHSCOPE_API_KEY
from my_llm import dashscope_llm

# 1.加载数据
loader = PyPDFLoader("../Book/demo.pdf")
pages = loader.load()
# 2，切分数据
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n","\n","。",",","!","?"," ",""] # 自定义分隔符
)
docs = text_splitter.split_documents(pages)
# print(f"PDF加载完成，一共切了{len(docs)}个片段")
# print(f"第一个片段的具体内容：\n{docs[0].page_content}")
# 3，定义Embedding模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)
# 4，创建向量数据库（存入Chroma数据库）
vectorstore =  Chroma.from_documents(
    documents = docs,
    embedding = embeddings,
    persist_directory = "./chroma_db",
)
# print("数据已成功存储")
# 5，创建QA链
qa_chain = RetrievalQA.from_chain_type(
    llm = dashscope_llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
)
#
query = "这份文档主要讲了什么内容？"
response = qa_chain.invoke(query)
print(f"AI的回答：\n{response['result']}")


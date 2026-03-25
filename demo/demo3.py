from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.retrievers.kendra import RetrieveResult
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from env_utils import DASHSCOPE_API_KEY
from my_llm import dashscope_llm

embeddings_model = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=DASHSCOPE_API_KEY
)

loader = TextLoader("../Book/红楼梦.txt", encoding='utf-8')
docs = loader.load()
# print(docs)

# RecursiveCharacterTextSplitter
text_spliter =  RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)  # chunk_size: 每个文档的最大字符数，chunk_overlap: 每个文档之间的重叠字符数
all_text_spliter = text_spliter.split_documents(docs)
# print(len(all_text_spliter))

vectorstore =  Chroma.from_documents(
    documents=all_text_spliter,
    embedding=embeddings_model,
    persist_directory="./chroma_db",
)

template="""
你是一个基于文档的助手。请仅使用以下提供的【背景资料】来回答问题。
如果你不知道答案，就说不知道，不要编造。
【文档】:{context}，
【问题】:{question}，
【回答】
"""
prompt =  PromptTemplate.format_prompt(
    template=template,
    input_variables=["context","question"],
)
# RetrievalQA 链
qa_chain = RetrievalQA.from_chain_type(
    llm=dashscope_llm,
    prompt=prompt,
    retriever=vectorstore.as_retriever(search_kwargs={"k":10}),
    chain_type_kwargs={"prompt": prompt},
)

query = "红楼梦的主要人物有哪些"
response = qa_chain.invoke({"query": query})
print(response)


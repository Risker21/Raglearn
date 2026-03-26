from init_llm.llm_factory import DashScopeEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from env_utils import DASHSCOPE_API_KEY

loader = TextLoader("../Book/西游记第二十七回.txt", encoding='utf-8')
docs = loader.load()
print("加载成功+++++++++++++++")

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)

embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./xiyouji27_db",
)
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(documents=docs)
result_docs = retriever.invoke("白骨精变的第一道菜")
for doc in result_docs:
    print(doc.page_content)

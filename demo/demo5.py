from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
import dashscope

from env_utils import DASHSCOPE_API_KEY
from my_llm import dashscope_llm

dashscope.api_key = DASHSCOPE_API_KEY

# ====================== 1. 读取文档 + 分块 ======================
loader = TextLoader("../Book/西游记解读.txt", encoding='utf-8')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# ====================== 2. 构建向量库 ======================
embeddings_model = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=DASHSCOPE_API_KEY,
)

vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings_model,
    persist_directory="./xiyouji_db",
)

# ====================== 3. 向量检索 ======================
query = "西游记的主线剧情是关于谁的？"
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
retrieved_docs = retriever.invoke(query)


# ====================== 4. Rerank 重排序 ======================
def rerank_documents(query, documents, top_k=5):
    contents = [doc.page_content for doc in documents]

    # 重点：是 TextReRank ！！！中间 R 大写！！！
    resp = dashscope.TextReRank.call(
        model="gte-rerank-v2",
        query=query,
        documents=contents,
        top_n=top_k
    )

    final_docs = []
    for item in resp.output.results:
        final_docs.append(documents[item.index])
    return final_docs


# 执行重排
reranked_docs = rerank_documents(query, retrieved_docs)

# ====================== 5. LLM 生成回答 ======================
context = "\n".join([doc.page_content for doc in reranked_docs])

template = """
只根据资料回答，不知道就说不知道。
【资料】:{context}
【问题】:{question}
【回答】
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
final_prompt = prompt.format(context=context, question=query)
answer = dashscope_llm.invoke(final_prompt)

print("=" * 50)
print("问题：", query)
print("回答：", answer.content)
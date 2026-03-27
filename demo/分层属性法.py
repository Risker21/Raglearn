from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import  MarkdownHeaderTextSplitter

from env_utils import DASHSCOPE_API_KEY
from my_llm import dashscope_llm

# 1.加载数据
loader = PyPDFLoader("../Book/demo.pdf")
pages = loader.load()

import re
full_text= ""
for page in pages:
    full_text += page.page_content +"\n"

# processed_text = re.sub(r'^(\d+\.\d+\s+)',r'## \1',full_text,flags = re.MULTILINE)


# 2，切分数据(定义识别规则)
headers_to_split_on = [
    ("一、", "一级模块"),
    ("二、", "一级模块"),
    ("三、", "一级模块"),
    ("四、", "一级模块"),
    ("五、", "一级模块"),
    ("1.", "二级主题"), # 匹配 1.1, 1.2 等
    ("2.", "二级主题"),
    ("3.", "二级主题"),
    ("4.", "二级主题"),
    ("5.", "二级主题"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

full_text = "\n".join([page.page_content for page in pages])

docs = markdown_splitter.split_text(full_text)
# print(f"切分了{len(docs)}个知识点")

embedding = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY
)
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="./title_db",
)

template = """你是一个专业的助教。请根据提供的上下文（Context）回答问题。
注意：每个片段的标题信息已经包含在内。

上下文内容：
{context}

问题：{question}
回答："""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm = dashscope_llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 50}), # 增加搜索范围，多捞一点
    chain_type_kwargs={"prompt":PROMPT}
)

query = "提取所有一级标题,并做个总结"
response = qa_chain.invoke(query)
print(f"AI的回答：\n{response['result']}")





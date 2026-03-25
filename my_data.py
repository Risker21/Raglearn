# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
#
# loader = TextLoader('西游记解读.txt', encoding='utf-8')
# docs = loader.load()
# # print(docs)
#
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
# splits = text_splitter.split_documents(docs)
#
# for split in splits:
#     print(split.page_content)
#
# # print(f"文档已被切分成{len(splits)}个段")
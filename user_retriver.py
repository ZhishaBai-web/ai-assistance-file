import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def document_retriever(upload_file,user_language,API,URL):
    file_content = upload_file.read() #读取上传的内容，返回内容的二进制数据
    temp_file_path = "temp.pdf"  #建立临时文件，当前路径下的temp.pdf文件
    with open(temp_file_path, "wb") as temp_file:   #路径，写入二进制，文件的变量名
        temp_file.write(file_content)  #写入内容

    loader=PyPDFLoader(temp_file_path)
    pdf=loader.load()
    if user_language=="中文":
        separators_role=["\n\n","\n","。","！","？","，","、",""]
    else:separators_role=["\n\n","\n",".","!","?",","," ",""]

    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=separators_role
    )

    texts=text_splitter.split_documents(pdf)
    embedding_model= OpenAIEmbeddings(model="text-embedding-3-large",
                            openai_api_key=API,
                            base_url=URL,
                           )

    db=FAISS.from_documents(texts,embedding_model)
    retriever=db.as_retriever()
    return retriever
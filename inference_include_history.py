from langchain.document_loaders import PyMuPDFLoader
import os

from dotenv import load_dotenv
load_dotenv(override=True)

import glob


PDF_data = []
pdfs = glob.glob("pdf/*.pdf")

for pdfFile in glob.glob("pdf/*.pdf"):
    loader = PyMuPDFLoader(pdfFile)
    PDF_data.extend(loader.load())

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=5)
all_splits = text_splitter.split_documents(PDF_data)

from langchain.embeddings import HuggingFaceEmbeddings
model_name = os.getenv("EMBEDDING_MODEL")
model_kwargs = {'device': os.getenv("EMBEDDING_DEVICE")}
embedding = HuggingFaceEmbeddings(model_name=model_name,
                                  model_kwargs=model_kwargs)

from langchain.vectorstores import Chroma
persist_directory = 'db'
if os.path.exists(persist_directory):
    vectordb = Chroma(embedding_function=embedding, persist_directory=persist_directory)
else:
    vectordb = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory=persist_directory)

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from queue import Queue, Empty
from typing import Any
from threading import Thread

model_path = os.getenv("LLM_MODEL_PATH")
print("\n\n" + model_path + "\n\n")

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=os.getenv("N_GPU_LAYERS"),
    n_batch=512,
    n_ctx=4096,
    temperature=os.getenv("LLM_TEMPERATURE"),
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


retriever = vectordb.as_retriever()

qa_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=("<<SYS>>根據已知資訊回答問題，若無法從已知資訊中得到答案，請回覆'根據資料庫內的資訊，我無法回覆此問題'，禁止變造答案，使用中文回覆。 <</SYS>>  \n ")),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("\n\n [INST] 已知資訊:\n'''\n{knownInfo}\n```\n\n 問題: {question} [/INST]")
    ]
)

def format_docs(docs):
    print("\n\n".join(doc.page_content for doc in docs))
    return "\n\n".join(doc.page_content for doc in docs)

from operator import itemgetter
rag_chain = (
    {
        "knownInfo": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
        "chat_history":itemgetter("chat_history")
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)


if __name__=='__main__':
    import gradio as gr
    
    def format_history(history):
        arr = []
        for a in history:
            arr.extend([HumanMessage(content=a[0]), AIMessage(content=a[1])])
        return arr

    def chat(message, history):
        print(history)
        while True:
            response = rag_chain.invoke({"question":message, "chat_history":format_history(history)})
            if response != "":
                break

        yield response

    gr.ChatInterface(chat).queue().launch()
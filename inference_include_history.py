from langchain.document_loaders import PyMuPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

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


llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=os.getenv("N_GPU_LAYERS"),
    n_batch=512,
    n_ctx=4096,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage

retriever = vectordb.as_retriever()

qa_system_prompt = """
你是一名導覽員，負責回答大家的問題。 \
請使用中文回覆問題。 \
{context}
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

from operator import itemgetter
rag_chain = (
    {
        "context": itemgetter("question") | retriever,
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
        response = rag_chain.invoke({"question":message+"。", "chat_history":format_history(history)})
        yield response

    gr.ChatInterface(chat).launch()
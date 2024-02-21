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

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=("""<<SYS>> \n
                               Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.\n <</SYS>>""")),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("\n\n [INST] {question} [/INST]")
    ]
)
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
contextualize_q_chain.invoke(
    {
        "chat_history": [
            HumanMessage(content="What does LLM stand for?"),
            AIMessage(content="Large language model"),
        ],
        "question": "What is meant by large",
    }
)


retriever = vectordb.as_retriever()

qa_system_prompt = """
你是一名導覽員，使用中文回覆問題。 \
{context}
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        # ("system", qa_system_prompt),
        SystemMessage(content=("<<SYS>> \n{context} 你是一名導覽員，使用中文回覆問題。\n <</SYS>>")),
        MessagesPlaceholder(variable_name="chat_history"),
        # ("human", "{question}"),
        HumanMessagePromptTemplate.from_template("\n\n [INST] 請使用中文回覆: \n\n {question} [/INST]")
    ]
)

# from operator import itemgetter
# rag_chain = (
#     {
#         "context": itemgetter("question") | retriever,
#         "question": itemgetter("question"),
#         "chat_history":itemgetter("chat_history")
#     }
#     | qa_prompt
#     | llm
#     | StrOutputParser()
# )
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]


rag_chain = (
    RunnablePassthrough.assign(
        context=contextualized_question | retriever | format_docs
    )
    | qa_prompt
    | llm
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

    gr.ChatInterface(chat).launch()
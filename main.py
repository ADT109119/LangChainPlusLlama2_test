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
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.llms import LlamaCpp
from queue import Queue, Empty
from typing import Any
from threading import Thread

model_path = os.getenv("LLM_MODEL_PATH")
q = Queue()
job_done = object()

class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: Any) -> None:
        return self.q.empty()
callbacks = [QueueCallback(q)]

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=os.getenv("N_GPU_LAYERS"),
    n_batch=512,
    n_ctx=4096,
    f16_kv=True,
    callbacks=callbacks,
    verbose=True,
)

from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.prompts import PromptTemplate

DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> \n 你是一名導覽員，負責回答大家的問題。 \
 \n <</SYS>> \n\n [INST] 請使用中文回覆: \n\n {question} [/INST]""",
)

DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""你是一名導覽員，負責回答大家的問題。 \
 請使用中文回覆: {question}""",
)

QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=DEFAULT_SEARCH_PROMPT,
    conditionals=[(lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)],
)

prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)

from langchain.chains.retrieval_qa.base import RetrievalQA

retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

if __name__=='__main__':
    import gradio as gr

    def answer(question):
        def task():
            response = qa.invoke(question)
            q.put(job_done)
        
        t = Thread(target=task)
        t.start()

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        def res(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            question = history[-1][0]
            print("Question: ", question)
            history[-1][1] = ""
            answer(question=question)
            while True:
                try:
                    next_token = q.get(True, timeout=1)
                    if next_token is job_done:
                        break
                    history[-1][1] += next_token
                    yield history
                except Empty:
                    continue

        msg.submit(res, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    demo.launch()

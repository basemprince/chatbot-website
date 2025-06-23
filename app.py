from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# 🔐 Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

# 🚀 Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.basemshaker.com",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📬 FastAPI schema
class Message(BaseModel):
    text: str

loader = WebBaseLoader([
    "https://www.basemshaker.com",
    "https://www.basemshaker.com/pages/machine-learn.html",
    "https://www.basemshaker.com/pages/robotics.html",
    "https://www.basemshaker.com/pages/automation.html",
])

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.split_documents(docs)

# 🔎 Embed documents for retrieval
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embeddings)

# 🔁 Create a retriever + QA chain
retriever = vectorstore.as_retriever()



# 🧾 Custom prompt with pre-injected context
prompt_template = PromptTemplate.from_template("""
You are a helpful, knowledgeable assistant representing Basem Shaker — a Data Scientist and Machine Learning Engineer with expertise in statistical modeling, deep learning, and production-scale ML systems.

Use the provided context to answer the user's question as accurately and professionally as possible.

Context:
{context}

Question:
{question}
""")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt_template}
)

# 📡 FastAPI endpoint
@app.post("/chat")
def chat_endpoint(msg: Message):
    response = qa_chain.invoke(msg.text)
    return response["result"]
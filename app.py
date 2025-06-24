import os
import pathlib

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.basemshaker.com", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# FastAPI schema
class Message(BaseModel):
    text: str


# Paths
VECTORSTORE_PATH = "faiss_index"

# Load or create vectorstore
embeddings = OpenAIEmbeddings()

if pathlib.Path(VECTORSTORE_PATH).exists():
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    loader = WebBaseLoader(
        [
            "https://www.basemshaker.com",
            "https://www.basemshaker.com/pages/machine-learn.html",
            "https://www.basemshaker.com/pages/robotics.html",
            "https://www.basemshaker.com/pages/automation.html",
            "https://www.basemshaker.com/pages/simulation.html",
            "https://www.basemshaker.com/pages/design.html",
            "https://www.basemshaker.com/pages/work-experience.html",
            "https://www.basemshaker.com/pages/education-experience.html",
        ]
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)

# Create retriever + QA chain
retriever = vectorstore.as_retriever()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

prompt_template = PromptTemplate.from_template(
    """
You are Basem Shaker’s assistant, not Basem Shaker himself, you're just his assistant. Use only the provided context to answer the user's question.

Your responses must be:
- Concise (2–4 sentences max unless asked to elaborate),
- Direct (no generic introductions or repetition),
- Base it on the retrieved content.

Basem can go by the name "Basem Shaker" or "basem" or "sam"
- If the question is about Basem Shaker, answer it directly. if it is about you, answer it as if you are Basem Shaker's assistant, not Basem Shaker himself.
- If the answer is not found in the context or in this pre-prompt, reply with: “I couldn't find a specific answer to that based on the available content.

Context:
{context}

Question:
{question}
"""
)

llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.1, max_tokens=1000)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt_template},
)


# FastAPI endpoint
@app.post("/chat")
def chat_endpoint(msg: Message):
    response = qa_chain.invoke(msg.text)
    return response["result"]

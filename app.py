import os
import pathlib
import uuid
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from pydantic import BaseModel

import firebase_admin
from firebase_admin import credentials, firestore


# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

# Initialize Firebase Admin SDK
FIREBASE_KEY_PATH = os.getenv("FIREBASE_KEY_PATH", "/etc/secrets/firebase-key.json")
cred = credentials.Certificate(FIREBASE_KEY_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

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
    session_id: str | None = None  # Optional session ID

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
            "https://www.basemshaker.com/pages/project-management.html",
            "https://www.basemshaker.com/pages/lean-manufacturing.html",
            "https://www.basemshaker.com/pages/problem-solving.html",
            "https://www.basemshaker.com/pages/technical-skills.html",
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
You are Basem Shaker’s assistant, not Basem Shaker himself, you're just his assistant so don't use the word I. Use only the provided context to answer the user's question.

Your responses must be:
- Concise (2–4 sentences max unless asked to elaborate),
- Direct (no generic introductions or repetition),
- Base it on the retrieved content.
- Follow the chat history for context.

Basem can go by the name "Basem Shaker" or "basem" or "sam"
- If the question is about Basem Shaker, answer it directly. if it is about you, answer it as if you are Basem Shaker's assistant, not Basem Shaker himself.

Context:
{context}

Question:
{question}
"""
)

llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0.1, max_tokens=1000)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    return_source_documents=False,
)

# ✅ Save conversation to Firebase Firestore
def save_to_firebase(session_id: str, user_msg: str, bot_response: str):
    db.collection("chat_logs").add({
        "session_id": session_id,
        "user_message": user_msg,
        "bot_response": bot_response,
        "timestamp": datetime.utcnow()
    })

# FastAPI endpoint
@app.post("/chat")
def chat_endpoint(msg: Message):
    # Generate a session ID if not provided
    session_id = msg.session_id if msg.session_id else str(uuid.uuid4())

    response = qa_chain.invoke({"question": msg.text, "chat_history": memory.chat_memory.messages})
    answer = response["answer"]

    save_to_firebase(session_id, msg.text, answer)

    return {"answer": answer, "session_id": session_id}

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

# 🚀 Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.basemshaker.com"],  # In production, replace with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📬 FastAPI schema
class Message(BaseModel):
    text: str

# 💬 Initialize the LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

# 🧠 Memory for ongoing conversations
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 🧾 Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, knowledgeable assistant representing Basem Shaker — a Data Scientist and Machine Learning Engineer with expertise in statistical modeling, deep learning, and production-scale ML systems. You help visitors understand his background, projects, publications, and experience across industries like automotive, manufacturing, and edge ML. Be concise, professional, and technical when needed. If asked, guide users to his portfolio (basemshaker.com), GitHub (basem-shaker), or contact details"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


legacy_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
)


@app.post("/chat")
def chat_endpoint(msg: Message):
    response = legacy_chain.invoke({"input":msg.text})
    return response["text"]
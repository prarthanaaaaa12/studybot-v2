import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_DIR = "./chroma_db"
embeddings = FastEmbedEmbeddings()

AGENT_PROMPTS = {
    "study": """You are a Study Agent. Answer the student's question ONLY using the study materials below.\nContext:\n{context}\nQuestion: {question}\nAnswer:""",
    "quiz": """You are a Quiz Agent. Generate 3 MCQs from the materials.\nContext:\n{context}\nTopic: {question}\nQuiz:""",
    "summary": """You are a Summary Agent. Summarize the topic using bullet points.\nContext:\n{context}\nTopic: {question}\nSummary:""",
    "code": """You are a Code Agent. Explain code concepts from the materials.\nContext:\n{context}\nQuestion: {question}\nAnswer:"""
}

class QuestionRequest(BaseModel):
    question: str
    agent: str = "study"
    subject: str = ""
    course: str = ""
    year: str = ""

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        vectorstore.add_documents(chunks)
        os.unlink(tmp_path)
        return {"message": f"Uploaded {len(chunks)} chunks from {file.filename}"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)
        agent_type = request.agent if request.agent in AGENT_PROMPTS else "study"
        prompt = PromptTemplate(template=AGENT_PROMPTS[agent_type], input_variables=["context", "question"])
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt}
        )
        enriched_question = f"{request.question} (Subject: {request.subject}, Course: {request.course}, Year: {request.year})"
        result = qa_chain.invoke({"query": enriched_question})
        return {"answer": result["result"], "agent": agent_type}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def root():
    return {"status": "Study bot is running!"}

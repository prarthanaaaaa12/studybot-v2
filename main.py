import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # FIXED
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_community.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate  # FIXED
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
    "study": """You are a Study Agent. Answer the student's question ONLY using the study materials below. Explain clearly and in simple terms. If not found, say "I couldn't find this in your study materials."
Context:
{context}
Question: {question}
Answer:""",
    "quiz": """You are a Quiz Agent. Using ONLY the study materials below, generate 3 multiple choice questions related to the student's topic. Format each as:
Q1. [Question]
a) option b) option c) option d) option
Answer: [correct option]
Context:
{context}
Topic: {question}
Quiz:""",
    "summary": """You are a Summary Agent. Using ONLY the study materials below, provide a clear and concise summary of the requested topic. Use bullet points.
Context:
{context}
Topic to summarize: {question}
Summary:""",
    "code": """You are a Code Agent. Using ONLY the study materials below, explain any code concepts or provide code examples related to the student's question. Format code properly.
Context:
{context}
Question: {question}
Answer:"""
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
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        vectorstore.add_documents(chunks)
        os.unlink(tmp_path)
        return {"message": f"Uploaded {len(chunks)} chunks from {file.filename}"}
    except Exception as e:
        return {"error": str(e), "file": file.filename}
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=GROQ_API_KEY
        )
        agent_type = request.agent if request.agent in AGENT_PROMPTS else "study"
        prompt_template = AGENT_PROMPTS[agent_type]
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
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
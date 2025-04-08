# backend/main.py
import os
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Updated import
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

# --- Configuration & Initialization ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Configure the core google-generativeai library (used for /summarize)
genai.configure(api_key=GEMINI_API_KEY)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Page Summarizer API",
    description="API for summarizing web pages and answering questions using Gemini and LangChain.",
    version="1.2.0", # Increment version to reflect auth fix
)

# --- CORS Configuration ---
origins = [
    "chrome-extension://*",
    "http://localhost",
    "https://localhost",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request Validation ---
class TextData(BaseModel):
    text: str = Field(..., min_length=10)

class QuestionData(TextData):
    question: str = Field(..., min_length=1)

# --- Helper Function to Build LangChain Chain ---
def build_chain(text: str):
    """Builds the ConversationalRetrievalChain for Q&A."""
    try:
        logger.info(f"Building chain for text starting with: {text[:100]}...")
        documents = [Document(page_content=text)]

        # *** Explicitly pass API key to LangChain components ***
        local_api_key = os.getenv("GEMINI_API_KEY") # Get key again for clarity or use GEMINI_API_KEY directly
        if not local_api_key:
             # This should not happen if initial check passed, but good practice
             raise ValueError("API Key disappeared unexpectedly.")

        logger.info("Initializing embeddings model...")
        try:
             embeddings = GoogleGenerativeAIEmbeddings(
                 model="models/embedding-001",
                 task_type="retrieval_document",
                 google_api_key=local_api_key # <-- PASS KEY HERE
             )
        except Exception as e_emb:
             logger.warning(f"Failed to use embedding-001, falling back: {e_emb}")
             # Fallback might also need the key depending on its implementation details
             embeddings = GoogleGenerativeAIEmbeddings(
                 model="models/embedding-001",
                 google_api_key=local_api_key # <-- PASS KEY HERE (for fallback too)
             )

        logger.info("Creating vector store...")
        vectordb = FAISS.from_documents(documents, embedding=embeddings)
        retriever = vectordb.as_retriever()

        logger.info("Initializing chat model (gemini-1.5-flash-latest)...")
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.3,
            convert_system_message_to_human=True,
            google_api_key=local_api_key # <-- PASS KEY HERE
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        logger.info("Creating conversational retrieval chain...")
        chain = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=retriever,
            memory=memory,
            verbose=True
        )
        logger.info("Chain built successfully.")
        return chain
    except Exception as e:
        logger.error(f"Error building chain: {e}", exc_info=True)
        # Make the error message slightly more specific if possible
        error_detail = f"Failed to build LangChain chain: {e}"
        if "credentials" in str(e).lower():
            error_detail = f"Failed to build LangChain chain due to credential issue (ensure API key is valid and passed correctly): {e}"
        raise HTTPException(status_code=500, detail=error_detail)


# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Smart Page Summarizer API!"}

@app.post("/summarize")
async def summarize(data: TextData):
    """Summarizes the provided text using Gemini 1.5 Flash."""
    logger.info("Received request for /summarize")
    try:
        text = data.text
        if not text:
             raise HTTPException(status_code=400, detail="Text content cannot be empty.")

        logger.info("Initializing generative model for summarization (gemini-1.5-flash-latest)...")
        # This uses the genai.configure() setting implicitly
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        logger.info(f"Generating summary for text length: {len(text)}")
        prompt = f"Please provide a concise summary (1-2 paragraphs) of the following web page content:\n\n{text}\n\nSummary:"
        response = await model.generate_content_async(prompt)

        logger.info("Summary generated successfully.")
        return {"summary": response.text}
    except Exception as e:
        logger.error(f"Error during summarization: {e}", exc_info=True)
        error_message = str(e)
        if "is not found for API version" in error_message or "is not supported" in error_message:
             logger.error(f"Model 'gemini-1.5-flash-latest' might not be available or configured correctly for your API key/region.")
             raise HTTPException(status_code=404, detail=f"Model not found or not supported: {error_message}")
        elif "API key not valid" in error_message:
             raise HTTPException(status_code=401, detail=f"Invalid API Key provided for summarization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {e}")


@app.post("/ask")
async def ask(data: QuestionData):
    """Answers a question about the provided text using LangChain and Gemini 1.5 Flash."""
    logger.info(f"Received request for /ask with question: {data.question}")
    try:
        text = data.text
        question = data.question
        if not text or not question:
            raise HTTPException(status_code=400, detail="Text content and question cannot be empty.")

        # build_chain now handles passing the API key internally
        chain = build_chain(text)

        logger.info(f"Invoking chain with question: {question}")
        result = await chain.ainvoke({"question": question})

        logger.info(f"Received answer from chain: {result.get('answer')}")
        return {"answer": result.get('answer', "Sorry, I couldn't find an answer.")}
    # Catch potential exceptions from build_chain or chain.ainvoke
    except HTTPException as http_exc:
        # Re-raise HTTPExceptions (like the 500 from build_chain failure)
        raise http_exc
    except Exception as e:
        logger.error(f"Error during Q&A invocation: {e}", exc_info=True)
        # Check if it's an API key error from invoking the chain
        if "API key not valid" in str(e):
             raise HTTPException(status_code=401, detail=f"Invalid API Key used during Q&A: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get answer during invocation: {e}")


# --- Optional: Add Exception Handler for Validation Errors ---
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
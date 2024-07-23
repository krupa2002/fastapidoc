from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import fitz  
import openai
import faiss
import numpy as np
import logging
import os

app = FastAPI()

class Query(BaseModel):
    user_query: str


pdf_text = ""
pdf_embeddings = None
index = None


openai.api_key = "sk-proj-7Wugf2LijKaO9kViHzq9T3BlbkFJsM580HpLtb9kIcNU7usk"

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_text, pdf_embeddings, index
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Invalid file type")
        content = await file.read()
        pdf_text = extract_text_from_pdf(content)
        pdf_embeddings = get_embeddings(pdf_text)
        index = build_faiss_index(pdf_embeddings)
        return {"filename": file.filename, "content": pdf_text}
    except Exception as e:
        logging.error(f"Error during upload: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

def extract_text_from_pdf(content):
    try:
        pdf_document = fitz.open("pdf", content)
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        raise

def get_embeddings(text):
    try:
        
        response = openai.Embedding.create(input=[text], engine="text-embedding-ada-002")
        return np.array([data["embedding"] for data in response["data"]])
    except Exception as e:
        logging.error(f"Error getting embeddings: {e}")
        raise

def build_faiss_index(embeddings):
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
    except Exception as e:
        logging.error(f"Error building FAISS index: {e}")
        raise

@app.post("/query/")
async def query_pdf(query: Query):
    global pdf_text, index, pdf_embeddings
    try:
        if not pdf_text:
            raise HTTPException(status_code=400, detail="No PDF content available")
        query_embedding = get_embeddings(query.user_query)
        _, I = index.search(query_embedding, k=5)  # Get top 5 relevant sections
        relevant_sections = [pdf_text[i] for i in I[0]]
        combined_text = " ".join(relevant_sections)
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Answer the question based on the following text: {combined_text}\n\nQuestion: {query.user_query}\nAnswer:",
            max_tokens=200
        )
        return {"response": response.choices[0].text.strip()}
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import fitz  # PyMuPDF
import openai
import logging

app = FastAPI()

class Query(BaseModel):
    user_query: str


pdf_text = ""

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_text
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type")
    content = await file.read()
    pdf_text = extract_text_from_pdf(content)
    return {"filename": file.filename, "content": pdf_text}

def extract_text_from_pdf(content):
    pdf_document = fitz.open("pdf", content)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

@app.post("/query/")
async def query_pdf(query: Query):
    global pdf_text
    if not pdf_text:
        raise HTTPException(status_code=400, detail="No PDF content available")
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Based on the PDF content: {pdf_text}\n\nAnswer the question: {query.user_query}",
            max_tokens=50
        )
        return {"response": response.choices[0].text.strip()}
    except Exception as e:
        logging.error(f"Error querying OpenAI API: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

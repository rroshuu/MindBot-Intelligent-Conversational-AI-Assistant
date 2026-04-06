from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.rag_service import add_document_chunks
import pypdf # or your preferred PDF library

router = APIRouter()

@router.post("/document")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # 1. Read PDF content
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(file.file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # 2. Split into chunks (simple version)
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        
        # 3. Add to your new free FAISS index
        add_document_chunks(chunks, source=file.filename)
        
        return {"message": f"Successfully indexed {len(chunks)} chunks from {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
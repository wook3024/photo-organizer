from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Photo Organizer API")

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pydantic import BaseModel
from services.scanner import scan_directory
from services.tagger import get_tagger

class ScanRequest(BaseModel):
    path: str

class AnalyzeRequest(BaseModel):
    path: str

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Photo Organizer API is running with CLIP"}

@app.post("/scan")
def scan_files(request: ScanRequest):
    """
    Scans a directory for images.
    """
    try:
        images = scan_directory(request.path)
        return {"count": len(images), "images": images}
    except Exception as e:
        return {"error": str(e)}

@app.post("/analyze")
def analyze_image(request: AnalyzeRequest):
    """
    Analyzes a single image using CLIP to generate tags.
    """
    try:
        tagger = get_tagger()
        tags = tagger.generate_tags(request.path)
        return {"path": request.path, "tags": tags}
    except Exception as e:
        return {"error": str(e)}


from fastapi.responses import FileResponse
import os

@app.get("/image")
def get_image(path: str):
    """
    Serves the image file from the absolute path.
    """
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "File not found"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

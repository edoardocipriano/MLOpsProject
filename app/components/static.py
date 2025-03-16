from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
import os

router = APIRouter()

@router.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the index.html file"""
    try:
        html_file_path = os.path.join(os.path.dirname(__file__), "index.html")
        with open(html_file_path, "r") as f:
            html_content = f.read()
        return html_content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
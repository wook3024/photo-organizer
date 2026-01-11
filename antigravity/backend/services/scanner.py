import os
import mimetypes
from pathlib import Path
from typing import List, Dict

import exifread
from datetime import datetime

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.heic', '.webp', '.tiff'}

def get_exif_data(path: Path):
    """
    Extracts basic EXIF data (Date Taken) from the image.
    """
    try:
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            
            # Date Taken
            date_taken = tags.get('EXIF DateTimeOriginal') or tags.get('Image DateTime')
            date_str = str(date_taken) if date_taken else None
            
            # Simple Location (very basic check if exists)
            # GPS processing is complex, skipping strict parsing for now, 
            # just noting if it has GPS tags
            has_gps = 'GPS GPSLatitude' in tags
            
            return {
                "date_taken": date_str,
                "has_gps": has_gps
            }
    except Exception:
        return {}

def scan_directory(root_path: str) -> List[Dict]:
    """
    Recursively scans the directory for image files.
    """
    images = []
    root = Path(root_path)

    if not root.exists():
         return []

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            # Basic info
            stat = path.stat()
            exif = get_exif_data(path)
            
            images.append({
                "path": str(path.absolute()),
                "filename": path.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "date_taken": exif.get("date_taken"),
                "has_gps": exif.get("has_gps", False)
            })
    
    return images

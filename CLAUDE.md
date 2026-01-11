# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PhotoMind** - An AI-powered photo organizer application that scans local directories for images, extracts EXIF metadata, and uses OpenAI's CLIP model to generate intelligent tags for photos.

## Architecture

This is a full-stack application with two main components:

### Backend (`antigravity/backend/`)
- **Framework**: FastAPI (Python)
- **AI Model**: OpenAI CLIP (clip-vit-base-patch32) via HuggingFace Transformers
- **Purpose**:
  - Serves REST API for image scanning and analysis
  - Handles file system operations to scan directories recursively
  - Extracts EXIF metadata (date taken, GPS data) using exifread
  - Generates AI-powered tags using CLIP zero-shot classification
  - Serves image files to frontend (browsers can't access local file:// paths)

**Key Files**:
- `main.py`: FastAPI app with endpoints `/scan`, `/analyze`, `/image`
- `services/scanner.py`: Recursively scans directories for images, extracts EXIF data
- `services/tagger.py`: CLIP-based image tagging using zero-shot classification with candidate labels

**Architecture Notes**:
- CLIP model is loaded as a singleton to avoid reloading on each request
- The `/image` endpoint serves local images via FileResponse since browsers cannot access file:// paths directly
- Supports image formats: `.jpg`, `.jpeg`, `.png`, `.heic`, `.webp`, `.tiff`

### Frontend (`antigravity/frontend/`)
- **Framework**: Next.js 16 with App Router (TypeScript, React 19)
- **Styling**: Tailwind CSS 4
- **UI Components**: Custom components with lucide-react icons, framer-motion for animations
- **Purpose**:
  - Browse photos in a masonry grid layout
  - Trigger AI tagging on individual photos
  - Display EXIF metadata (date taken, GPS indicators)

**Key Structure**:
- `app/page.tsx`: Landing page / dashboard
- `app/gallery/page.tsx`: Main photo gallery view with masonry layout
- `app/layout.tsx`: Root layout with persistent sidebar navigation
- `components/Sidebar.tsx`: Fixed left sidebar with navigation
- `components/PhotoCard.tsx`: Individual photo card with hover actions and AI tagging
- `utils/api.ts`: API client for backend communication

**Architecture Notes**:
- All photo images are served via backend (`http://localhost:8000/image?path=...`) rather than direct file access
- AI tagging is lazy-loaded per photo (click to analyze)
- Uses client-side state management (no global state library yet)

## Development Commands

### Backend
```bash
cd antigravity/backend

# Run development server (with hot reload)
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# The backend runs on http://localhost:8000
# API docs available at http://localhost:8000/docs
```

**Backend Dependencies**: Install Python packages (likely using pip/conda):
- fastapi
- uvicorn
- python-multipart
- pillow
- transformers
- torch
- exifread

Note: No `requirements.txt` found yet - dependencies need to be installed manually or a requirements file should be created.

### Frontend
```bash
cd antigravity/frontend

# Install dependencies
npm install

# Run development server
npm run dev
# Frontend runs on http://localhost:3000

# Build for production
npm run build

# Run production build
npm start

# Lint code
npm run lint
```

## Key Integration Points

1. **CORS**: Backend allows requests from `http://localhost:3000` (frontend dev server)
2. **Image Serving**: Frontend fetches images via `GET /image?path=<absolute_path>` from backend
3. **API Base URL**: Configured in `frontend/utils/api.ts` as `http://localhost:8000`
4. **Default Scan Path**: Hardcoded in `gallery/page.tsx` as `/Users/shinukyi/Gallary/proto` - should be made configurable

## AI Tagging System

The tagging system uses CLIP's zero-shot classification:
- **Candidate Labels** (in `tagger.py`): screenshot, receipt, document, landscape, city, beach, forest, mountain, food, coffee, restaurant, cat, dog, pet, selfish, group photo, portrait, car, architecture, flower
- **Threshold**: Tags with >5% confidence are returned
- **Model**: Pre-trained `openai/clip-vit-base-patch32` (optimized for CPU/laptop GPU)
- **Process**: Image + text labels are encoded, similarity scores computed, softmax applied

To add more tags, edit the `candidate_labels` list in `services/tagger.py:22-29`.

## Development Workflow

1. Start backend server first (required for image serving)
2. Start frontend dev server
3. Frontend will attempt to scan `/Users/shinukyi/Gallary/proto` on page load
4. Click "AI Tag" button on any photo to trigger CLIP analysis
5. Tags are cached in component state after first analysis

## Known Issues / TODOs

- No `requirements.txt` for backend Python dependencies
- Scan path is hardcoded in frontend - needs settings UI
- No database - all data is computed on-demand
- Storage indicator in sidebar is static/fake
- Routes `/tags` and `/settings` are not implemented yet
- CLIP model loads on first analyze request (causes initial delay)

# PhotoMind

AI-powered photo organizer that scans local directories for images, extracts EXIF metadata, and uses OpenAI's CLIP model to generate intelligent tags.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 20+
- npm or yarn

### Backend Setup

```bash
cd antigravity/backend

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Edit .env with your settings
# Set ALLOWED_SCAN_DIRS to your photo directories

# Run server
python main.py
```

Backend will run on http://localhost:8000

### Frontend Setup

```bash
cd antigravity/frontend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env.local

# Run development server
npm run dev
```

Frontend will run on http://localhost:3000

## 📁 Project Structure

```
proto/
├── antigravity/
│   ├── backend/          # FastAPI backend
│   │   ├── main.py       # API endpoints
│   │   └── services/     # Core services
│   │       ├── scanner.py   # Directory scanning & EXIF
│   │       └── tagger.py    # CLIP-based AI tagging
│   └── frontend/         # Next.js frontend
│       ├── app/          # App Router pages
│       ├── components/   # React components
│       └── utils/        # API client
├── CLAUDE.md             # Development guide for Claude Code
└── CODE_IMPROVEMENTS.md  # Detailed improvement suggestions
```

## 🔑 Key Features

- 📸 **Smart Scanning**: Recursively scans directories for images
- 🏷️ **AI Tagging**: Uses CLIP for zero-shot image classification
- 📅 **EXIF Metadata**: Extracts date taken, GPS data
- 🖼️ **Modern UI**: Responsive masonry grid layout
- ⚡ **Fast**: Lazy loading and efficient image serving

## 🛠️ Development

See [CLAUDE.md](./CLAUDE.md) for detailed development instructions.

See [CODE_IMPROVEMENTS.md](./CODE_IMPROVEMENTS.md) for improvement suggestions and best practices.

## 📝 API Documentation

Once the backend is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## ⚠️ Security Notice

**Important**: Before deploying to production, please review and implement the security improvements outlined in `CODE_IMPROVEMENTS.md`, especially:
- Path traversal protection
- CORS configuration
- Input validation

## 📄 License

MIT

# PhotoMind 코드 개선 제안서

## 📋 목차
1. [보안 취약점 (Critical)](#1-보안-취약점-critical)
2. [백엔드 개선사항](#2-백엔드-개선사항)
3. [프론트엔드 개선사항](#3-프론트엔드-개선사항)
4. [아키텍처 개선사항](#4-아키텍처-개선사항)
5. [성능 최적화](#5-성능-최적화)
6. [개발 경험 개선](#6-개발-경험-개선)
7. [테스트 및 품질 보증](#7-테스트-및-품질-보증)

---

## 1. 보안 취약점 (Critical)

### 🚨 Path Traversal 취약점
**현재 문제:**
```python
# backend/main.py:57-64
@app.get("/image")
def get_image(path: str):
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "File not found"}
```

**위험도:** ⚠️ **CRITICAL**
- 사용자가 임의의 파일 경로를 제공할 수 있음
- `/etc/passwd`, `/etc/shadow` 등 시스템 파일 접근 가능
- 소스 코드, 환경 변수 파일 노출 위험

**개선 방안:**
```python
from pathlib import Path
import os

ALLOWED_IMAGE_DIRS = [
    Path("/Users/shinukyi/Gallary/proto"),
    # 설정에서 관리되는 안전한 디렉토리 목록
]

@app.get("/image")
def get_image(path: str):
    try:
        requested_path = Path(path).resolve()

        # 1. 허용된 디렉토리 내부인지 확인
        if not any(requested_path.is_relative_to(allowed_dir)
                   for allowed_dir in ALLOWED_IMAGE_DIRS):
            raise HTTPException(status_code=403, detail="Access denied")

        # 2. 파일 존재 여부 확인
        if not requested_path.exists() or not requested_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        # 3. 이미지 파일 확장자 검증
        if requested_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Invalid file type")

        return FileResponse(requested_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid path")
```

### 🚨 CORS 설정 문제
**현재 문제:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**문제점:**
- `allow_methods=["*"]`, `allow_headers=["*"]` - 너무 광범위
- 프로덕션 환경 설정이 없음

**개선 방안:**
```python
from fastapi.middleware.cors import CORSMiddleware
import os

# 환경 변수로 관리
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

if ENVIRONMENT == "development":
    origins = ["http://localhost:3000"]
else:
    origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # 필요한 메소드만 명시
    allow_headers=["Content-Type", "Authorization"],
    max_age=3600,
)
```

### 🚨 입력 검증 부족
**현재 문제:**
```python
@app.post("/scan")
def scan_files(request: ScanRequest):
    try:
        images = scan_directory(request.path)
        return {"count": len(images), "images": images}
    except Exception as e:
        return {"error": str(e)}  # 스택 트레이스 노출 가능
```

**개선 방안:**
```python
from pydantic import BaseModel, validator
import logging

logger = logging.getLogger(__name__)

class ScanRequest(BaseModel):
    path: str

    @validator('path')
    def validate_path(cls, v):
        path = Path(v).resolve()
        if not path.exists():
            raise ValueError("Path does not exist")
        if not path.is_dir():
            raise ValueError("Path must be a directory")
        # 허용된 디렉토리 검증
        if not any(path.is_relative_to(allowed) for allowed in ALLOWED_IMAGE_DIRS):
            raise ValueError("Access denied to this directory")
        return str(path)

@app.post("/scan")
def scan_files(request: ScanRequest):
    try:
        images = scan_directory(request.path)
        return {"count": len(images), "images": images}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Scan error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
```

---

## 2. 백엔드 개선사항

### 2.1 의존성 관리

**문제:** `requirements.txt`가 없음

**해결 방안:** `requirements.txt` 생성
```txt
# backend/requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.32.0
python-multipart==0.0.12
pydantic==2.10.0
pillow==11.0.0
transformers==4.47.0
torch==2.5.1
exifread==3.0.0
python-dotenv==1.0.1
```

**더 나은 방법:** `pyproject.toml` 사용 (modern Python)
```toml
# backend/pyproject.toml
[project]
name = "photomind-backend"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "python-multipart>=0.0.12",
    "pydantic>=2.10.0",
    "pillow>=11.0.0",
    "transformers>=4.47.0",
    "torch>=2.5.1",
    "exifread>=3.0.0",
    "python-dotenv>=1.0.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "httpx>=0.27.0",
    "black>=24.0.0",
    "ruff>=0.8.0",
]
```

### 2.2 환경 설정 관리

**문제:** 하드코딩된 설정값들

**해결:** 설정 파일 도입
```python
# backend/config.py
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List

class Settings(BaseSettings):
    # API 설정
    api_title: str = "PhotoMind API"
    api_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS 설정
    environment: str = "development"
    allowed_origins: List[str] = ["http://localhost:3000"]

    # 파일 시스템 설정
    allowed_scan_dirs: List[Path] = [Path.home() / "Pictures"]
    max_file_size: int = 50 * 1024 * 1024  # 50MB

    # CLIP 모델 설정
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_confidence_threshold: float = 0.05

    # 로깅
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

**`.env` 파일:**
```bash
# backend/.env
ENVIRONMENT=development
ALLOWED_ORIGINS=http://localhost:3000
ALLOWED_SCAN_DIRS=/Users/shinukyi/Pictures,/Users/shinukyi/Gallary/proto
CLIP_MODEL_NAME=openai/clip-vit-base-patch32
LOG_LEVEL=INFO
```

### 2.3 에러 처리 개선

**현재 문제:** 일관성 없는 에러 처리
```python
# services/scanner.py
except Exception:
    return {}  # 조용히 실패
```

**개선:**
```python
# backend/exceptions.py
class PhotoMindException(Exception):
    """Base exception for PhotoMind"""
    pass

class ImageProcessingError(PhotoMindException):
    """Image processing failed"""
    pass

class ScanError(PhotoMindException):
    """Directory scanning failed"""
    pass

# services/scanner.py
import logging
logger = logging.getLogger(__name__)

def get_exif_data(path: Path) -> dict:
    try:
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            return {
                "date_taken": str(tags.get('EXIF DateTimeOriginal') or tags.get('Image DateTime') or ''),
                "has_gps": 'GPS GPSLatitude' in tags
            }
    except FileNotFoundError:
        logger.warning(f"File not found: {path}")
        raise ImageProcessingError(f"File not found: {path}")
    except PermissionError:
        logger.error(f"Permission denied: {path}")
        raise ImageProcessingError(f"Permission denied: {path}")
    except Exception as e:
        logger.error(f"Failed to read EXIF from {path}: {e}")
        return {"date_taken": None, "has_gps": False}
```

### 2.4 구조화 개선 (프로젝트 구조)

**현재 구조:**
```
backend/
├── main.py (모든 로직이 한 파일에)
└── services/
    ├── scanner.py
    └── tagger.py
```

**개선된 구조:**
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 앱 설정
│   ├── config.py               # 설정 관리
│   ├── exceptions.py           # 커스텀 예외
│   ├── dependencies.py         # 의존성 주입
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── images.py       # 이미지 관련 엔드포인트
│   │   │   ├── scan.py         # 스캔 관련 엔드포인트
│   │   │   └── analyze.py      # 분석 관련 엔드포인트
│   │   └── models/
│   │       ├── requests.py     # Request 모델
│   │       └── responses.py    # Response 모델
│   ├── services/
│   │   ├── scanner.py
│   │   ├── tagger.py
│   │   └── cache.py            # 캐싱 서비스
│   └── utils/
│       ├── logging.py
│       └── validators.py
├── tests/
│   ├── test_scanner.py
│   ├── test_tagger.py
│   └── test_api.py
├── requirements.txt
├── .env.example
└── README.md
```

### 2.5 데이터베이스 도입

**문제:** 스캔 결과와 태그를 매번 재계산

**개선:** SQLite/PostgreSQL로 메타데이터 저장
```python
# app/models/database.py
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True)
    path = Column(String, unique=True, index=True)
    filename = Column(String)
    size = Column(Integer)
    modified = Column(Float)
    date_taken = Column(String, nullable=True)
    has_gps = Column(Boolean, default=False)

    # 스캔 정보
    scanned_at = Column(DateTime, default=datetime.utcnow)
    last_analyzed = Column(DateTime, nullable=True)

    # AI 태그 (JSON으로 저장)
    tags = Column(JSON, nullable=True)  # [{"label": "cat", "confidence": 0.85}, ...]

    # 캐시 무효화
    file_hash = Column(String, nullable=True)  # MD5 해시로 변경 감지

# CRUD 함수들
def get_or_create_image(db, path: str, metadata: dict) -> Image:
    image = db.query(Image).filter(Image.path == path).first()
    if image:
        # 파일이 수정되었는지 확인
        if image.modified != metadata['modified']:
            # 업데이트
            for key, value in metadata.items():
                setattr(image, key, value)
            image.last_analyzed = None  # 재분석 필요
            db.commit()
    else:
        image = Image(**metadata)
        db.add(image)
        db.commit()
    return image
```

### 2.6 CLIP 모델 최적화

**현재 문제:**
- 매 요청마다 이미지 전체 로딩
- 후보 레이블이 고정됨
- 배치 처리 없음

**개선:**
```python
# services/tagger.py
from functools import lru_cache
from PIL import Image
import hashlib

class ClipTagger:
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or settings.clip_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading CLIP model on {self.device}...")
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model.eval()  # 평가 모드
        print("CLIP model loaded.")

    @lru_cache(maxsize=1000)
    def _get_text_embeddings(self, labels_tuple: tuple):
        """텍스트 임베딩 캐싱 - 동일한 레이블 세트는 재사용"""
        inputs = self.processor(text=list(labels_tuple), return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features

    def generate_tags(
        self,
        image_path: str,
        candidate_labels: List[str] = None,
        top_k: int = 5
    ) -> List[tuple]:
        if candidate_labels is None:
            candidate_labels = self._get_default_labels()

        try:
            # 이미지 로드 및 전처리
            image = Image.open(image_path).convert("RGB")
            image_inputs = self.processor(images=image, return_tensors="pt")
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

            # 이미지 임베딩
            with torch.no_grad():
                image_features = self.model.get_image_features(**image_inputs)

            # 텍스트 임베딩 (캐시됨)
            text_features = self._get_text_embeddings(tuple(candidate_labels))

            # 유사도 계산
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).squeeze(0)

            probs = similarity.softmax(dim=0).cpu().numpy()

            # 결과 정렬 및 필터링
            results = [(label, float(prob)) for label, prob in zip(candidate_labels, probs)]
            results.sort(key=lambda x: x[1], reverse=True)

            # top_k 또는 threshold 이상만 반환
            threshold = settings.clip_confidence_threshold
            return [r for r in results[:top_k] if r[1] > threshold]

        except Exception as e:
            logger.error(f"Error tagging {image_path}: {e}")
            raise ImageProcessingError(f"Failed to tag image: {str(e)}")

    def generate_tags_batch(self, image_paths: List[str], candidate_labels: List[str] = None):
        """배치 처리로 여러 이미지 동시 분석"""
        # 구현...
        pass
```

---

## 3. 프론트엔드 개선사항

### 3.1 환경 변수 관리

**문제:** API URL이 하드코딩됨
```typescript
// utils/api.ts
const API_BASE_URL = 'http://localhost:8000';
```

**해결:**
```typescript
// utils/api.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000

// .env.production
NEXT_PUBLIC_API_URL=https://api.photomind.com
```

### 3.2 상태 관리 개선

**문제:** 모든 상태가 컴포넌트 로컬에 분산됨

**해결:** Zustand/Jotai로 전역 상태 관리
```typescript
// stores/photoStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface PhotoStore {
  photos: Photo[];
  scanPath: string;
  isLoading: boolean;
  error: string | null;

  // Actions
  setPhotos: (photos: Photo[]) => void;
  setScanPath: (path: string) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  // Computed
  photoCount: () => number;
}

export const usePhotoStore = create<PhotoStore>()(
  persist(
    (set, get) => ({
      photos: [],
      scanPath: '/Users/shinukyi/Gallary/proto',
      isLoading: false,
      error: null,

      setPhotos: (photos) => set({ photos }),
      setScanPath: (path) => set({ scanPath: path }),
      setLoading: (loading) => set({ isLoading: loading }),
      setError: (error) => set({ error }),

      photoCount: () => get().photos.length,
    }),
    {
      name: 'photo-storage',
      partialize: (state) => ({ scanPath: state.scanPath }), // scanPath만 persist
    }
  )
);
```

### 3.3 에러 처리 및 사용자 피드백

**문제:** 에러가 console.error로만 처리됨

**해결:** Toast 알림 시스템 도입
```bash
npm install sonner
```

```typescript
// app/layout.tsx
import { Toaster } from 'sonner';

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body>
        <Sidebar />
        <main className="pl-64 min-h-screen">{children}</main>
        <Toaster position="top-right" richColors />
      </body>
    </html>
  );
}

// app/gallery/page.tsx
import { toast } from 'sonner';

const loadPhotos = async () => {
  setLoading(true);
  try {
    const res = await scanDirectory(targetPath);
    setPhotos(res.images);
    toast.success(`${res.count}개의 사진을 발견했습니다`);
  } catch (err) {
    toast.error('사진을 불러오는 데 실패했습니다', {
      description: err instanceof Error ? err.message : '알 수 없는 오류',
    });
  } finally {
    setLoading(false);
  }
};
```

### 3.4 이미지 로딩 최적화

**문제:**
- 모든 이미지를 한번에 로드
- Lazy loading 없음
- 썸네일 없음

**해결 1: React Virtualization**
```bash
npm install react-window react-window-infinite-loader
```

```typescript
// components/VirtualizedGallery.tsx
import { FixedSizeGrid } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';

export function VirtualizedGallery({ photos }: { photos: Photo[] }) {
  const COLUMN_COUNT = 4;
  const GUTTER_SIZE = 16;

  const Cell = ({ columnIndex, rowIndex, style }: any) => {
    const index = rowIndex * COLUMN_COUNT + columnIndex;
    if (index >= photos.length) return null;

    return (
      <div style={style}>
        <PhotoCard photo={photos[index]} />
      </div>
    );
  };

  return (
    <AutoSizer>
      {({ height, width }) => (
        <FixedSizeGrid
          columnCount={COLUMN_COUNT}
          columnWidth={(width - GUTTER_SIZE * (COLUMN_COUNT - 1)) / COLUMN_COUNT}
          height={height}
          rowCount={Math.ceil(photos.length / COLUMN_COUNT)}
          rowHeight={400}
          width={width}
        >
          {Cell}
        </FixedSizeGrid>
      )}
    </AutoSizer>
  );
}
```

**해결 2: 백엔드에서 썸네일 생성**
```python
# backend/services/thumbnail.py
from PIL import Image
from pathlib import Path
import hashlib

THUMBNAIL_DIR = Path("./thumbnails")
THUMBNAIL_DIR.mkdir(exist_ok=True)

def generate_thumbnail(image_path: str, size: tuple = (400, 400)) -> Path:
    """썸네일 생성 및 캐싱"""
    # 파일 해시로 썸네일 경로 생성
    file_hash = hashlib.md5(image_path.encode()).hexdigest()
    thumb_path = THUMBNAIL_DIR / f"{file_hash}.webp"

    if thumb_path.exists():
        return thumb_path

    # 썸네일 생성
    with Image.open(image_path) as img:
        img.thumbnail(size, Image.Resampling.LANCZOS)
        img.save(thumb_path, "WEBP", quality=85, optimize=True)

    return thumb_path

# main.py
@app.get("/image/thumbnail")
def get_thumbnail(path: str):
    # 보안 검증...
    thumb_path = generate_thumbnail(path)
    return FileResponse(thumb_path)
```

### 3.5 타입 안전성 강화

**개선:** API 응답 타입을 백엔드와 공유
```typescript
// types/api.ts
export interface Photo {
  path: string;
  filename: string;
  size: number;
  modified: number;
  date_taken?: string | null;
  has_gps?: boolean;
}

export interface Tag {
  label: string;
  confidence: number;
}

export interface ScanResponse {
  count: number;
  images: Photo[];
  error?: string;
}

export interface AnalyzeResponse {
  path: string;
  tags: Tag[];
  error?: string;
}

// API 클라이언트에 타입 가드 추가
export async function scanDirectory(path: string): Promise<ScanResponse> {
  const res = await fetch(`${API_BASE_URL}/scan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  });

  if (!res.ok) {
    throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  }

  const data = await res.json();

  if (data.error) {
    throw new Error(data.error);
  }

  return data;
}
```

### 3.6 접근성 개선

**문제:** 키보드 내비게이션, 스크린 리더 지원 부족

**개선:**
```typescript
// components/PhotoCard.tsx
export function PhotoCard({ photo }: PhotoCardProps) {
  return (
    <div
      className="group relative..."
      role="article"
      aria-label={`사진: ${photo.filename}`}
    >
      <div className="aspect-[3/4]...">
        <img
          src={imageUrl}
          alt={photo.filename}
          loading="lazy"
          onError={(e) => {
            (e.target as HTMLImageElement).src = '/placeholder.png';
          }}
        />

        <div className="absolute inset-0...">
          <button
            onClick={handleAnalyze}
            disabled={analyzing}
            className="bg-white/90..."
            aria-label={tags.length > 0 ? '태그 보기' : 'AI 태그 생성'}
            aria-busy={analyzing}
          >
            {/* ... */}
          </button>
        </div>
      </div>

      {showTags && tags.length > 0 && (
        <div
          className="mt-3..."
          role="list"
          aria-label="이미지 태그"
        >
          {tags.slice(0, 3).map(([tag, score]) => (
            <span
              key={tag}
              className="text-[10px]..."
              role="listitem"
            >
              {tag} {Math.round(score * 100)}%
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
```

### 3.7 성능 모니터링

**추가:** Web Vitals 측정
```typescript
// app/layout.tsx
import { SpeedInsights } from "@vercel/speed-insights/next";
import { Analytics } from "@vercel/analytics/react";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body>
        {children}
        <SpeedInsights />
        <Analytics />
      </body>
    </html>
  );
}
```

---

## 4. 아키텍처 개선사항

### 4.1 마이크로서비스 분리 (선택적)

**현재:** 모놀리식 구조
**개선:** 서비스 분리 고려

```
Services:
├── API Gateway (FastAPI)
├── Image Scanner Service (독립 워커)
├── AI Tagging Service (GPU 서버)
├── Thumbnail Generator Service
└── Database (PostgreSQL)
```

### 4.2 캐싱 전략

**Redis 도입:**
```python
# backend/services/cache.py
import redis
import json
from typing import Optional

redis_client = redis.Redis(
    host='localhost',
    port=6379,
    decode_responses=True
)

def cache_tags(image_path: str, tags: list, ttl: int = 86400):
    """태그 결과 캐싱 (24시간)"""
    key = f"tags:{image_path}"
    redis_client.setex(key, ttl, json.dumps(tags))

def get_cached_tags(image_path: str) -> Optional[list]:
    """캐시된 태그 가져오기"""
    key = f"tags:{image_path}"
    data = redis_client.get(key)
    return json.loads(data) if data else None

# main.py
@app.post("/analyze")
def analyze_image(request: AnalyzeRequest):
    # 캐시 확인
    cached_tags = get_cached_tags(request.path)
    if cached_tags:
        return {"path": request.path, "tags": cached_tags, "cached": True}

    # AI 분석
    tagger = get_tagger()
    tags = tagger.generate_tags(request.path)

    # 캐시 저장
    cache_tags(request.path, tags)

    return {"path": request.path, "tags": tags, "cached": False}
```

### 4.3 백그라운드 작업 큐

**Celery 도입:**
```python
# backend/tasks.py
from celery import Celery

celery_app = Celery('photomind', broker='redis://localhost:6379')

@celery_app.task
def analyze_image_async(image_path: str):
    """백그라운드에서 이미지 분석"""
    tagger = get_tagger()
    tags = tagger.generate_tags(image_path)

    # DB에 저장
    save_tags_to_db(image_path, tags)

    return tags

@celery_app.task
def scan_directory_async(directory: str):
    """대용량 디렉토리 스캔"""
    images = scan_directory(directory)

    # 각 이미지를 개별 태스크로 분석
    for image in images:
        analyze_image_async.delay(image['path'])

    return len(images)

# main.py
@app.post("/scan/async")
def scan_async(request: ScanRequest):
    task = scan_directory_async.delay(request.path)
    return {"task_id": task.id, "status": "processing"}

@app.get("/task/{task_id}")
def get_task_status(task_id: str):
    task = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task.status,
        "result": task.result if task.ready() else None
    }
```

### 4.4 WebSocket으로 실시간 업데이트

```python
# backend/main.py
from fastapi import WebSocket
from typing import List

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        manager.disconnect(websocket)

# 스캔 진행 상황 브로드캐스트
async def scan_with_progress(path: str):
    images = []
    for idx, image in enumerate(scan_directory_generator(path)):
        images.append(image)
        await manager.broadcast({
            "type": "scan_progress",
            "current": idx + 1,
            "image": image
        })
    return images
```

---

## 5. 성능 최적화

### 5.1 이미지 로딩 최적화

**Next.js Image 컴포넌트 사용:**
```typescript
// components/PhotoCard.tsx
import Image from 'next/image';

export function PhotoCard({ photo }: PhotoCardProps) {
  return (
    <div className="relative">
      <Image
        src={imageUrl}
        alt={photo.filename}
        width={400}
        height={600}
        loading="lazy"
        placeholder="blur"
        blurDataURL="data:image/svg+xml;base64,..." // Low quality placeholder
        className="w-full h-full object-cover"
      />
    </div>
  );
}
```

**백엔드에서 이미지 최적화:**
```python
# next.config.ts에서 이미지 도메인 허용
const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
        pathname: '/image/**',
      },
    ],
  },
};
```

### 5.2 데이터베이스 쿼리 최적화

```python
# 인덱스 추가
class Image(Base):
    __tablename__ = "images"

    path = Column(String, unique=True, index=True)
    filename = Column(String, index=True)  # 파일명으로 검색
    date_taken = Column(String, index=True)  # 날짜로 정렬
    modified = Column(Float, index=True)

# 페이지네이션
@app.get("/images")
def list_images(skip: int = 0, limit: int = 50, sort_by: str = "date_taken"):
    images = db.query(Image)\
        .order_by(getattr(Image, sort_by).desc())\
        .offset(skip)\
        .limit(limit)\
        .all()

    return {"images": images, "skip": skip, "limit": limit}
```

### 5.3 CLIP 모델 배치 처리

```python
def generate_tags_batch(self, image_paths: List[str], batch_size: int = 8):
    """배치 처리로 성능 향상"""
    results = []

    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]

        # 이미지들을 배치로 로드
        images = [Image.open(p).convert("RGB") for p in batch]

        # 배치 인코딩
        image_inputs = self.processor(images=images, return_tensors="pt", padding=True)
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**image_inputs)
            # ... 나머지 처리

        results.extend(batch_results)

    return results
```

---

## 6. 개발 경험 개선

### 6.1 Docker 컨테이너화

```dockerfile
# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# frontend/Dockerfile
FROM node:20-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app

COPY --from=builder /app/next.config.ts ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

CMD ["npm", "start"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./antigravity/backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/photomind
      - REDIS_URL=redis://redis:6379
    volumes:
      - ${HOME}/Pictures:/data/pictures:ro
    depends_on:
      - db
      - redis

  frontend:
    build: ./antigravity/frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: photomind
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

volumes:
  pgdata:
```

### 6.2 개발 도구 설정

**Linting & Formatting:**
```python
# backend/.pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.0
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.8.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
```

```json
// frontend/.eslintrc.json
{
  "extends": ["next/core-web-vitals", "prettier"],
  "rules": {
    "@typescript-eslint/no-unused-vars": "error",
    "@typescript-eslint/no-explicit-any": "warn"
  }
}
```

### 6.3 API 문서 자동화

```python
# backend/main.py
app = FastAPI(
    title="PhotoMind API",
    description="AI-Powered Photo Organizer API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 응답 모델 정의
from pydantic import BaseModel, Field

class ImageResponse(BaseModel):
    path: str = Field(..., description="Absolute path to the image")
    filename: str = Field(..., description="Image filename")
    size: int = Field(..., description="File size in bytes")
    date_taken: Optional[str] = Field(None, description="EXIF date taken")
    has_gps: bool = Field(False, description="Whether GPS data is present")

    class Config:
        json_schema_extra = {
            "example": {
                "path": "/Users/user/Pictures/photo.jpg",
                "filename": "photo.jpg",
                "size": 2048000,
                "date_taken": "2024:01:15 14:30:00",
                "has_gps": True
            }
        }

@app.post("/scan", response_model=ScanResponse)
def scan_files(request: ScanRequest):
    """
    디렉토리를 스캔하여 모든 이미지 파일을 찾습니다.

    - **path**: 스캔할 디렉토리의 절대 경로

    Returns:
        - count: 발견된 이미지 개수
        - images: 이미지 메타데이터 목록
    """
    # ...
```

---

## 7. 테스트 및 품질 보증

### 7.1 백엔드 테스트

```python
# backend/tests/test_scanner.py
import pytest
from pathlib import Path
from services.scanner import scan_directory, get_exif_data

@pytest.fixture
def test_image_dir(tmp_path):
    """테스트용 이미지 디렉토리 생성"""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    # 테스트 이미지 생성...
    return img_dir

def test_scan_directory(test_image_dir):
    images = scan_directory(str(test_image_dir))
    assert len(images) > 0
    assert all('path' in img for img in images)

def test_get_exif_data():
    # Mock EXIF 데이터 테스트
    pass

# backend/tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_scan_invalid_path():
    response = client.post("/scan", json={"path": "/nonexistent"})
    assert response.status_code == 400
```

### 7.2 프론트엔드 테스트

```bash
npm install -D @testing-library/react @testing-library/jest-dom vitest
```

```typescript
// components/__tests__/PhotoCard.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { PhotoCard } from '../PhotoCard';

describe('PhotoCard', () => {
  const mockPhoto = {
    path: '/test/photo.jpg',
    filename: 'photo.jpg',
    size: 1024,
    modified: Date.now(),
    date_taken: '2024:01:15 14:30:00',
    has_gps: true,
  };

  it('renders photo information', () => {
    render(<PhotoCard photo={mockPhoto} />);
    expect(screen.getByText('photo.jpg')).toBeInTheDocument();
  });

  it('triggers AI tagging on button click', async () => {
    render(<PhotoCard photo={mockPhoto} />);
    const button = screen.getByRole('button', { name: /AI 태그/i });
    fireEvent.click(button);
    // ... 비동기 테스트
  });
});
```

### 7.3 E2E 테스트

```bash
npm install -D @playwright/test
```

```typescript
// e2e/gallery.spec.ts
import { test, expect } from '@playwright/test';

test('should load and display photos', async ({ page }) => {
  await page.goto('http://localhost:3000/gallery');

  // 스캔 완료 대기
  await page.waitForSelector('[data-testid="photo-card"]');

  const photos = await page.locator('[data-testid="photo-card"]').count();
  expect(photos).toBeGreaterThan(0);
});

test('should analyze photo with AI', async ({ page }) => {
  await page.goto('http://localhost:3000/gallery');

  // 첫 번째 사진의 AI 태그 버튼 클릭
  await page.locator('[data-testid="ai-tag-button"]').first().click();

  // 태그 표시 대기
  await page.waitForSelector('[role="list"][aria-label="이미지 태그"]');

  const tags = await page.locator('[role="listitem"]').count();
  expect(tags).toBeGreaterThan(0);
});
```

---

## 📊 우선순위 요약

### 🔴 High Priority (즉시 수정 필요)
1. **Path Traversal 보안 취약점** - 파일 접근 제한
2. **CORS 설정 개선** - 프로덕션 환경 대비
3. **입력 검증 강화** - Pydantic validators
4. **requirements.txt 생성** - 의존성 관리
5. **환경 변수 도입** - 하드코딩 제거

### 🟡 Medium Priority (단기 개선)
1. **데이터베이스 도입** - 메타데이터 영속성
2. **에러 처리 개선** - 일관된 예외 처리
3. **썸네일 생성** - 성능 최적화
4. **상태 관리 라이브러리** - Zustand/Jotai
5. **Toast 알림** - 사용자 피드백

### 🟢 Low Priority (장기 개선)
1. **캐싱 시스템** - Redis 도입
2. **백그라운드 작업** - Celery 큐
3. **WebSocket** - 실시간 업데이트
4. **Docker 컨테이너화** - 배포 간소화
5. **E2E 테스트** - Playwright

---

## 🎯 Quick Wins (빠르게 적용 가능한 개선)

1. **`.env` 파일 추가** (5분)
2. **requirements.txt 생성** (10분)
3. **Toast 알림 추가** (30분)
4. **이미지 lazy loading** (20분)
5. **API 에러 핸들링 개선** (1시간)

이 문서를 참고하여 단계적으로 개선을 진행하시면 됩니다!

const API_BASE_URL = 'http://localhost:8000';

export interface Photo {
    path: string;
    filename: string;
    size: number;
    modified: number;
    date_taken?: string;
    has_gps?: boolean;
}

export interface ScanResponse {
    count: number;
    images: Photo[];
}

export interface AnalyzeResponse {
    path: string;
    tags: Array<[string, number]>;
}

export async function scanDirectory(path: string): Promise<ScanResponse> {
    const res = await fetch(`${API_BASE_URL}/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
    });
    if (!res.ok) throw new Error('Failed to scan');
    return res.json();
}

export async function analyzeImage(path: string): Promise<AnalyzeResponse> {
    const res = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
    });
    if (!res.ok) throw new Error('Failed to analyze');
    return res.json();
}

"use client";

import { useState } from 'react';
import { Photo, analyzeImage } from '@/utils/api';
import { Sparkles, MapPin, Calendar, Loader2 } from 'lucide-react';

interface PhotoCardProps {
    photo: Photo;
}

export function PhotoCard({ photo }: PhotoCardProps) {
    const [tags, setTags] = useState<Array<[string, number]>>([]);
    const [analyzing, setAnalyzing] = useState(false);
    const [showTags, setShowTags] = useState(false);

    // Convert absolute path to a local file URL or serve via backend?
    // Browsers block local file access (file://). 
    // We cannot display local files mainly in browser without a server serving them.
    // We need to serve images via Backend! I forgot this critical part.
    // I'll need to update Backend to serve static files or an /image endpoint.
    // For now, I will assume I can fix backend to serve `/api/image?path=...`
    // Temp URL: using a placeholder for now to proceed, but will FIX backend next.

    const imageUrl = `http://localhost:8000/image?path=${encodeURIComponent(photo.path)}`;

    const handleAnalyze = async (e: React.MouseEvent) => {
        e.stopPropagation();
        if (tags.length > 0) {
            setShowTags(!showTags);
            return;
        }

        setAnalyzing(true);
        try {
            const result = await analyzeImage(photo.path);
            setTags(result.tags);
            setShowTags(true);
        } catch (err) {
            console.error(err);
        } finally {
            setAnalyzing(false);
        }
    };

    return (
        <div className="group relative break-inside-avoid mb-4 rounded-xl overflow-hidden bg-white shadow-sm border border-zinc-100 transition-all hover:shadow-md">
            {/* Image */}
            <div className="aspect-[3/4] overflow-hidden bg-zinc-100 relative">
                <img
                    src={imageUrl}
                    alt={photo.filename}
                    className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                    onError={(e) => {
                        // Fallback if image fails to load
                        (e.target as HTMLImageElement).src = 'https://placehold.co/400x600?text=Load+Error';
                    }}
                />

                {/* Overlay Actions */}
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors flex items-end justify-between p-3 opacity-0 group-hover:opacity-100">
                    <button
                        onClick={handleAnalyze}
                        disabled={analyzing}
                        className="bg-white/90 backdrop-blur text-xs font-semibold px-3 py-1.5 rounded-full shadow-sm flex items-center gap-1.5 hover:bg-white transition-colors"
                    >
                        {analyzing ? <Loader2 className="w-3 h-3 animate-spin" /> : <Sparkles className="w-3 h-3 text-indigo-500" />}
                        {tags.length > 0 ? 'View Tags' : 'AI Tag'}
                    </button>
                </div>
            </div>

            {/* Info */}
            <div className="p-3">
                <p className="text-sm font-medium text-zinc-900 truncate">{photo.filename}</p>
                <div className="flex items-center gap-3 mt-2 text-xs text-zinc-500">
                    {photo.date_taken && (
                        <div className="flex items-center gap-1">
                            <Calendar className="w-3 h-3" />
                            <span>{photo.date_taken.split(' ')[0]}</span>
                        </div>
                    )}
                    {photo.has_gps && (
                        <div className="flex items-center gap-1">
                            <MapPin className="w-3 h-3" />
                            <span>Loc</span>
                        </div>
                    )}
                </div>

                {/* Tags Display */}
                {showTags && tags.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-1">
                        {tags.slice(0, 3).map(([tag, score]) => (
                            <span key={tag} className="text-[10px] bg-indigo-50 text-indigo-600 px-1.5 py-0.5 rounded-md border border-indigo-100">
                                {tag} {Math.round(score * 100)}%
                            </span>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}

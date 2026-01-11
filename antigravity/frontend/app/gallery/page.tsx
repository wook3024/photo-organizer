"use client";

import { useEffect, useState } from 'react';
import { Photo, scanDirectory } from '@/utils/api';
import { PhotoCard } from '@/components/PhotoCard';
import { RefreshCw, Folder } from 'lucide-react';

export default function GalleryPage() {
    const [photos, setPhotos] = useState<Photo[]>([]);
    const [loading, setLoading] = useState(true);
    const [targetPath, setTargetPath] = useState('/Users/shinukyi/Gallary/proto'); // Default scan path

    const loadPhotos = async () => {
        setLoading(true);
        try {
            const res = await scanDirectory(targetPath);
            setPhotos(res.images);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadPhotos();
    }, []);

    return (
        <div className="p-8 max-w-7xl mx-auto">
            <header className="flex items-center justify-between mb-8">
                <div>
                    <h2 className="text-2xl font-bold tracking-tight text-zinc-900">All Photos</h2>
                    <p className="text-zinc-500 mt-1 flex items-center gap-2 text-sm">
                        <Folder className="w-4 h-4" />
                        Scanning: {targetPath}
                    </p>
                </div>
                <button
                    onClick={loadPhotos}
                    className="flex items-center gap-2 px-4 py-2 bg-white border border-zinc-200 rounded-lg text-sm font-medium hover:bg-zinc-50 transition-colors"
                >
                    <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                    Refresh
                </button>
            </header>

            {loading && photos.length === 0 ? (
                <div className="flex justify-center py-20">
                    <RefreshCw className="w-6 h-6 animate-spin text-zinc-300" />
                </div>
            ) : (
                <div className="columns-2 md:columns-3 lg:columns-4 gap-4 space-y-4">
                    {photos.map((photo) => (
                        <PhotoCard key={photo.path} photo={photo} />
                    ))}
                </div>
            )}

            {!loading && photos.length === 0 && (
                <div className="text-center py-20 text-zinc-400">
                    No photos found in this directory.
                </div>
            )}
        </div>
    );
}

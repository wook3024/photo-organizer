"use client";

import Link from "next/link";
import { ArrowRight, Image as ImageIcon } from "lucide-react";

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center p-8 min-h-[80vh]">
      <div className="max-w-2xl text-center space-y-6">
        <h1 className="text-4xl font-bold tracking-tighter sm:text-5xl bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
          Welcome to PhotoMind
        </h1>
        <p className="text-lg text-zinc-500">
          Your intelligent personal photo organizer. Connect your folder,
          extract metadata, and let AI tag your memories automatically.
        </p>

        <div className="flex gap-4 justify-center">
          <Link
            href="/gallery"
            className="inline-flex h-12 items-center justify-center rounded-lg bg-zinc-900 px-8 text-sm font-medium text-zinc-50 shadow transition-colors hover:bg-zinc-900/90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-zinc-950"
          >
            Go to Gallery <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
          <button className="inline-flex h-12 items-center justify-center rounded-lg border border-zinc-200 bg-white px-8 text-sm font-medium shadow-sm transition-colors hover:bg-zinc-100 hover:text-zinc-900 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-zinc-950">
            Settings
          </button>
        </div>
      </div>
    </div>
  );
}

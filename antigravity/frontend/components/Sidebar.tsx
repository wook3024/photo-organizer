"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { LayoutGrid, Image as ImageIcon, Settings, Hash } from 'lucide-react';

const NAV_ITEMS = [
    { name: 'Dashboard', href: '/', icon: LayoutGrid },
    { name: 'All Photos', href: '/gallery', icon: ImageIcon },
    { name: 'Smart Tags', href: '/tags', icon: Hash },
    { name: 'Settings', href: '/settings', icon: Settings },
];

export function Sidebar() {
    const pathname = usePathname();

    return (
        <div className="w-64 h-screen bg-zinc-50 border-r border-zinc-200 flex flex-col fixed left-0 top-0">
            <div className="p-6">
                <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                    PhotoMind
                </h1>
                <p className="text-xs text-zinc-500 mt-1">AI-Powered Organizer</p>
            </div>

            <nav className="flex-1 px-4 space-y-1">
                {NAV_ITEMS.map((item) => {
                    const Icon = item.icon;
                    const isActive = pathname === item.href;
                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={`flex items-center px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${isActive
                                    ? 'bg-blue-50 text-blue-700'
                                    : 'text-zinc-600 hover:bg-zinc-100'
                                }`}
                        >
                            <Icon className="w-5 h-5 mr-3" />
                            {item.name}
                        </Link>
                    );
                })}
            </nav>

            <div className="p-4 border-t border-zinc-200">
                <div className="bg-zinc-100 rounded-lg p-3">
                    <p className="text-xs font-medium text-zinc-500">Storage Used</p>
                    <div className="w-full bg-zinc-200 h-1.5 rounded-full mt-2">
                        <div className="bg-blue-500 h-1.5 rounded-full w-[45%]"></div>
                    </div>
                </div>
            </div>
        </div>
    );
}

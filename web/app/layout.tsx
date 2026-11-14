import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Screw It",
  description:
    "Scan a broken thing, understand it, and get it fixed or printed.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div className="flex min-h-screen flex-col">
          <header className="border-b border-slate-800 bg-panel/80 backdrop-blur">
            <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
              <Link href="/" className="text-lg font-semibold tracking-tight">
                <span className="text-accent">Screw</span> It
              </Link>
              <nav className="flex gap-4 text-sm text-slate-300">
                <Link href="/" className="hover:text-white">
                  Gallery
                </Link>
                <Link href="/scan/new" className="hover:text-white">
                  New scan
                </Link>
              </nav>
            </div>
          </header>
          <main className="flex-1">{children}</main>
        </div>
      </body>
    </html>
  );
}

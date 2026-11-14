import type { Metadata } from "next";
import { SiteHeader } from "@/components/SiteHeader";
import { AccountProvider } from "@/components/AccountProvider";
import "./globals.css";

export const metadata: Metadata = {
  title: "Screw It",
  description: "Scan an object. Explore its parts. Let's build.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <a href="#main-content" className="skip-link">Skip to content</a>
        <AccountProvider>
          <SiteHeader />
          <main id="main-content">{children}</main>
        </AccountProvider>
      </body>
    </html>
  );
}

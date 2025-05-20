import "./globals.css";
import type { Metadata } from "next";
import { Outfit, JetBrains_Mono } from "next/font/google";

const outfit = Outfit({ subsets: ["latin"], variable: "--font-outfit" });
const jetbrains = JetBrains_Mono({ subsets: ["latin"], variable: "--font-code" });

export const metadata: Metadata = {
  title: "NeuroBlock | Visual AI Builder",
  description: "Design, train, and deploy neural networks visually with NeuroBlock.",
  openGraph: {
    title: "NeuroBlock",
    description: "Visual AI Builder with drag-and-drop architecture and real-time training.",
    url: "https://neuroblock.co",
    siteName: "NeuroBlock",
  },
  twitter: {
    card: "summary_large_image",
    title: "NeuroBlock",
    description: "Visual AI Builder with drag-and-drop architecture and real-time training.",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${outfit.variable} ${jetbrains.variable}`}>
      <body className="bg-[#0a0a0a] text-white font-sans min-h-screen antialiased">
        {children}
      </body>
    </html>
  );
}
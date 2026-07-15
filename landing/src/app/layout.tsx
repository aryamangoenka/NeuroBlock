import type React from "react";
import "./globals.css";
import { Bricolage_Grotesque, IBM_Plex_Mono } from "next/font/google";

const bricolage = Bricolage_Grotesque({
  subsets: ["latin"],
  variable: "--f-display",
  weight: ["400", "600", "700"],
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  variable: "--f-mono",
  weight: ["400", "500"],
});

const title = "NeuroBlock — Build neural networks visually";
const description =
  "Design, train, and export real neural networks by connecting blocks on a canvas. Live training, real code out. Made for high school students.";

export const metadata = {
  metadataBase: new URL("https://neuroblock-app.vercel.app"),
  title,
  description,
  openGraph: {
    title,
    description,
    type: "website",
    images: ["/builder.png"],
  },
  twitter: {
    card: "summary_large_image",
    title,
    description,
    images: ["/builder.png"],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={`${bricolage.variable} ${plexMono.variable}`}>
      <head>
        <link
          href="https://api.fontshare.com/v2/css?f[]=general-sans@400,500,600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}

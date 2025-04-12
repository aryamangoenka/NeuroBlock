/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        main: 'var(--main)',
        bg: 'var(--bg)',
        bw: 'var(--bw)',
        border: 'var(--border)',
        text: 'var(--text)',
        mtext: 'var(--mtext)',
      },
      boxShadow: {
        'custom': 'var(--shadow)',
      },
      borderRadius: {
        DEFAULT: 'var(--border-radius)',
      }
    },
  },
  plugins: [],
} 
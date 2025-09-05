// tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/**/*.{js,ts,jsx,tsx}',
    './pages/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
    './app/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      screens: {
        'xs': '320px',
        '3xl': '1920px',
        '4xl': '2560px',
      },
      backgroundColor: {
        'white-90': 'rgba(255, 255, 255, 0.9)',
        'black-30': 'rgba(0, 0, 0, 0.3)',
      },
      fontFamily: {
        'bb-mono': ['BB Manual Mono', 'monospace'],
        'formula': ['Formula Condensed', 'sans-serif'],
        'formula-bold': ['Formula Condensed', 'sans-serif'],
        'formula-light': ['Formula Condensed', 'sans-serif'],
        'helvetica-now-text': ['Helvetica Now Text', 'sans-serif'],
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem',
      },
      maxWidth: {
        'xs': '20rem',
        '8xl': '88rem',
        '9xl': '96rem',
      },
      fontSize: {
        '10xl': '10rem',
        '11xl': '12rem',
        '12xl': '14rem',
      },
    },
  },
};

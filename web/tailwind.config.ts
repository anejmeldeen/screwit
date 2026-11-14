import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#0b1120",
        panel: "#111827",
        accent: "#38bdf8",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};

export default config;

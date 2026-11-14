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
        ink: "#343123",
        paper: "#f1d45d",
        muted: "#615736",
        line: "#b4a04b",
        wash: "#e5ca65",
        panel: "#f8e8a5",
        accent: "#343123",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};

export default config;

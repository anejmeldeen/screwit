/** @type {import('next').NextConfig} */
// /api/* is proxied at request time by app/api/[...path]/route.ts, which reads SCREWIT_API_ORIGIN
// on every request. (A rewrite here would bake the API address into the build.)
// A static export (GitHub Pages) cannot include that dynamic route, and `output: 'export'` made
// every server build fail, run.sh included, so the export is opt-in: NEXT_OUTPUT=export npm run build.
const nextConfig = {
  ...(process.env.NEXT_OUTPUT === 'export' ? { output: 'export' } : {}),
  images: {
    unoptimized: true,
  },
};

export default nextConfig;

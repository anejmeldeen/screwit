/** @type {import('next').NextConfig} */
// /api/* is proxied at request time by app/api/[...path]/route.ts, which reads SCREWIT_API_ORIGIN
// on every request. (A rewrite here would bake the API address into the build.)
const nextConfig = {
  ...(process.env.SCREWIT_STATIC_EXPORT === '1' ? { output: 'export' } : {}),
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
/** @type {import('next').NextConfig} */
// /api/* is proxied at request time by app/api/[...path]/route.ts, which reads SCREWIT_API_ORIGIN
// on every request. (A rewrite here would bake the API address into the build.)
const nextConfig = {
  reactStrictMode: true,
};

export default nextConfig;

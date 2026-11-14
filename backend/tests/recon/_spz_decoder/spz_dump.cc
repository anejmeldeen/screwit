// Independent decoder: load a .spz with Niantic's own loadSpz and print the first N gaussians.
#include "load-spz.h"
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: spz_dump <file.spz> [n]\n"); return 2; }
  int n = (argc >= 3) ? std::atoi(argv[2]) : 1;
  spz::UnpackOptions opts;  // default: UNSPECIFIED, no conversion
  spz::GaussianCloud g = spz::loadSpz(std::string(argv[1]), opts);
  if (g.numPoints == 0) { std::fprintf(stderr, "decode failed / empty\n"); return 1; }
  std::printf("numPoints %d shDegree %d\n", g.numPoints, g.shDegree);
  for (int i = 0; i < n && i < g.numPoints; i++) {
    std::printf("g%d pos %.6f %.6f %.6f scale %.6f %.6f %.6f rot %.6f %.6f %.6f %.6f alpha %.6f color %.6f %.6f %.6f\n",
      i,
      g.positions[i*3+0], g.positions[i*3+1], g.positions[i*3+2],
      g.scales[i*3+0], g.scales[i*3+1], g.scales[i*3+2],
      g.rotations[i*4+0], g.rotations[i*4+1], g.rotations[i*4+2], g.rotations[i*4+3],
      g.alphas[i],
      g.colors[i*3+0], g.colors[i*3+1], g.colors[i*3+2]);
  }
  return 0;
}

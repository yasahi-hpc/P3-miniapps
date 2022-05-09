#ifndef __CONFIG_HPP__
#define __CONFIG_HPP__

#include "Types.hpp"

struct Config {
  // Local grids
  int nx, ny, nz;
  int gnx, gny, gnz;

  // Parallelization
  int px, py, pz;
  int nbiter;

  // Grid size
  const float64 Lx = 1.0;
  const float64 Ly = 1.0;
  const float64 Lz = 1.0;
  float64 dx, dy, dz;

  // Physical constants
  const float64 umax = 1.0;
  const float64 Kappa = 1.0;
  float64 dt;

  Config() = delete;
  Config(int _nx, int _ny, int _nz, int _px, int _py, int _pz, int _nbiter)
    : nx(_nx), ny(_ny), nz(_nz), px(_px), py(_py), pz(_pz), nbiter(_nbiter) {
    gnx = nx * px;
    gny = ny * py;
    gnz = nz * pz;

    dx = Lx / static_cast<float64>(gnx);
    dy = Ly / static_cast<float64>(gny);
    dz = Lz / static_cast<float64>(gnz);
    dt = 0.1 * dx * dx / Kappa;
  }
};

#endif

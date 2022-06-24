#ifndef __CONFIG_HPP__
#define __CONFIG_HPP__

#include "types.hpp"

struct Config {
  // grids
  int nx, ny, nz;

  int nbiter;
  int freq_diag;

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
  Config(int _nx, int _ny, int _nz, int _nbiter, int _freq_diag)
    : nx(_nx), ny(_ny), nz(_nz), nbiter(_nbiter), freq_diag(_freq_diag) {

    dx = Lx / static_cast<float64>(nx);
    dy = Ly / static_cast<float64>(ny);
    dz = Lz / static_cast<float64>(nz);
    dt = 0.1 * dx * dx / Kappa;
  }
};

#endif

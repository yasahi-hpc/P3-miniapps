#ifndef __TIMESTEP_HPP__
#define __TIMESTEP_HPP__

#include "Config.hpp"
#include "../Timer.hpp"
#include "tiles.h"

template <class View3DType>
void step(Config &conf, View3DType &u, View3DType &un, std::vector<Timer*> &timers) {
  const int nx = conf.nx, ny = conf.ny, nz = conf.nz;
  const float64 coef = conf.Kappa * conf.dt / (conf.dx*conf.dx);
  
  timers[Heat]->begin();
  MDPolicy<3> heat_policy3d({{0, 0, 0}},
                            {{nx, ny, nz}},
                            {{TILE_SIZE0, TILE_SIZE1, TILE_SIZE2}}
                           );

  auto heat_eq = KOKKOS_LAMBDA (const int ix, const int iy, const int iz) {
    const int ixp1 = (ix + nx + 1) % nx;
    const int ixm1 = (ix + nx - 1) % nx;
    const int iyp1 = (iy + ny + 1) % ny;
    const int iym1 = (iy + ny - 1) % ny;
    const int izp1 = (iz + nz + 1) % nz;
    const int izm1 = (iz + nz - 1) % nz;

    un(ix, iy, iz) = u(ix, iy, iz)
                   + coef * ( u(ixp1, iy, iz) + u(ixm1, iy, iz)
                            + u(ix, iyp1, iz) + u(ix, iym1, iz)
                            + u(ix, iy, izp1) + u(ix, iy, izm1)
                            - 6. * u(ix, iy, iz) );
  };

  Kokkos::parallel_for("heat", heat_policy3d, heat_eq);
  Kokkos::fence();
  timers[Heat]->end();
}

#endif

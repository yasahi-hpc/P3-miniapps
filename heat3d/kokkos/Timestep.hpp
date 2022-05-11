#ifndef __TIMESTEP_HPP__
#define __TIMESTEP_HPP__

#include "Config.hpp"
#include "../Timer.hpp"
#include "boundary.hpp"
#include "tiles.h"

template <class View3DType>
void step(Config &conf, Boundary &boundary, View3DType &u, View3DType &un, std::vector<Timer*> &timers) {
  boundary.exchangeHalos(u, timers);

  const int nx = conf.nx, ny = conf.ny, nz = conf.nz;
  const float64 coef = conf.Kappa * conf.dt / (conf.dx*conf.dx);
  
  timers[Heat]->begin();
  MDPolicy<3> heat_policy3d({{0, 0, 0}},
                            {{nx, ny, nz}},
                            {{TILE_SIZE0, TILE_SIZE1, TILE_SIZE2}}
                           );

  auto heat_eq = KOKKOS_LAMBDA (const int ix, const int iy, const int iz) {
    un(ix, iy, iz) = u(ix, iy, iz)
      + coef * ( u(ix+1, iy, iz) + u(ix-1, iy, iz)
               + u(ix, iy+1, iz) + u(ix, iy-1, iz)
               + u(ix, iy, iz+1) + u(ix, iy, iz-1)
               - 6. * u(ix, iy, iz) );
  };

  Kokkos::parallel_for("heat", heat_policy3d, heat_eq);
  Kokkos::fence();
  timers[Heat]->end();
}

#endif

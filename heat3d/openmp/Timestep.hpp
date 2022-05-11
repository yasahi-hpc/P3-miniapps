#ifndef __TIMESTEP_HPP__
#define __TIMESTEP_HPP__

#include "Config.hpp"
#include "../Timer.hpp"
#include "boundary.hpp"

void step(Config &conf, Boundary &boundary, RealView3D &u, RealView3D &un, std::vector<Timer*> &timers);

void step(Config &conf, Boundary &boundary, RealView3D &u, RealView3D &un, std::vector<Timer*> &timers) {
  boundary.exchangeHalos(conf, u, timers);

  const int nx = conf.nx, ny = conf.ny, nz = conf.nz;
  const float64 coef = conf.Kappa * conf.dt / (conf.dx*conf.dx);
  
  timers[Heat]->begin();
  #if defined(ENABLE_OPENMP_OFFLOAD)
    #pragma omp target teams distribute parallel for simd collapse(3)
  #else
    #pragma omp parallel for schedule(static) collapse(2)
  #endif
  for(int iz=0; iz<nz; iz++) {
    for(int iy=0; iy<ny; iy++) {
      LOOP_SIMD
      for(int ix=0; ix<nx; ix++) {
        un(ix, iy, iz) = u(ix, iy, iz)
          + coef * ( u(ix+1, iy, iz) + u(ix-1, iy, iz)
                   + u(ix, iy+1, iz) + u(ix, iy-1, iz)
                   + u(ix, iy, iz+1) + u(ix, iy, iz-1)
                   - 6. * u(ix, iy, iz) );
      }
    }
  }
  timers[Heat]->end();
}

#endif

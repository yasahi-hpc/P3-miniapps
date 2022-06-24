#ifndef __TIMESTEP_HPP__
#define __TIMESTEP_HPP__

#include "Config.hpp"
#include "../Timer.hpp"

void step(Config &conf, RealView3D &u, RealView3D &un, std::vector<Timer*> &timers);

void step(Config &conf, RealView3D &u, RealView3D &un, std::vector<Timer*> &timers) {
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
      }
    }
  }
  timers[Heat]->end();
}

#endif

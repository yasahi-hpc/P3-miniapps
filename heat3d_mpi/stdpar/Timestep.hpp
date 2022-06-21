#ifndef __TIMESTEP_HPP__
#define __TIMESTEP_HPP__

#include "Config.hpp"
#include "../Timer.hpp"
#include "Heat_kernel.hpp"
#include "MPI_Comm.hpp"
#include "Parallel_For.hpp"

void step(Config &conf, Comm &comm, RealView3D &u, RealView3D &un, std::vector<Timer*> &timers);

void step(Config &conf, Comm &comm, RealView3D &u, RealView3D &un, std::vector<Timer*> &timers) {
  comm.exchangeHalos(u, timers);
  
  #if defined(ACCESS_VIA_RAW_POINTERS)
    auto heat3d_kernel = [&]() {
      Iterate_policy<1> policy1d(conf.nx*conf.ny*conf.nz);
      Impl::for_each(policy1d, heat_1d_functor<default_iterate_layout>(conf, u, un));
      };
    exec_with_timer(heat3d_kernel, timers[Heat]);
  #else
    auto heat3d_kernel = [&]() {
      Iterate_policy<3> policy3d({0, 0, 0}, {conf.nx, conf.ny, conf.nz});
      Impl::for_each(policy3d, heat_functor(conf, u, un));
      };
    exec_with_timer(heat3d_kernel, timers[Heat]);
  #endif
}

#endif

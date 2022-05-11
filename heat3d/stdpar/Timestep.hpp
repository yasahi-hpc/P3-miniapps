#ifndef __TIMESTEP_HPP__
#define __TIMESTEP_HPP__

#include "Config.hpp"
#include "../Timer.hpp"
#include "Heat_kernel.hpp"
#include "boundary.hpp"
#include "Parallel_For.hpp"
#include "utils.hpp"

void step(Config &conf, Boundary &boundary, RealView3D &u, RealView3D &un, std::vector<Timer*> &timers);

void step(Config &conf, Boundary &boundary, RealView3D &u, RealView3D &un, std::vector<Timer*> &timers) {
  boundary.exchangeHalos(u, timers);
  
  auto heat3d_kernel = [&]() {
    Iterate_policy<3> policy3d({0, 0, 0}, {conf.nx, conf.ny, conf.nz});
    Impl::for_each(policy3d, heat_functor(conf, u, un));
  };
  exec_with_timer(heat3d_kernel, timers[Heat]);
}

#endif

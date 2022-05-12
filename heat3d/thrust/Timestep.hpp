#ifndef __TIMESTEP_HPP__
#define __TIMESTEP_HPP__

#include "Config.hpp"
#include "../Timer.hpp"
#include "Heat_kernel.hpp"
#include "Parallel_For.hpp"
#include "utils.hpp"

void step(Config &conf, RealView3D &u, RealView3D &un, std::vector<Timer*> &timers);

void step(Config &conf, RealView3D &u, RealView3D &un, std::vector<Timer*> &timers) {
  auto heat3d_kernel = [&]() {
    const int3 begin = make_int3(0, 0, 0);
    const int3 end   = make_int3(conf.nx, conf.ny, conf.nz);
    Impl::for_each<default_iterate_layout>(begin, end, heat_functor(conf, u, un));
    synchronize();
  };
  exec_with_timer(heat3d_kernel, timers[Heat]);
}

#endif

#include <Kokkos_Core.hpp>
#include <iostream>
#include <chrono>
#include <array>
#include "types.hpp"
#include "Config.hpp"
#include "../Parser.hpp"
#include "Init.hpp"
#include "Timestep.hpp"
#include "Math.hpp"

int main(int argc, char *argv[]) {
  Parser parser(argc, argv);
  auto shape = parser.shape_;
  int nbiter = parser.nbiter_;
  int nx = shape[0], ny = shape[1], nz = shape[2];

  std::vector<Timer*> timers;
  defineTimers(timers);

  // When initializing Kokkos, you may pass in command-line arguments,
  // just like with MPI_Init().  Kokkos reserves the right to remove
  // arguments from the list that start with '--kokkos-'.
  Kokkos::InitArguments args_kokkos;
  args_kokkos.num_threads = parser.num_threads_;
  args_kokkos.num_numa    = parser.teams_;
  args_kokkos.device_id   = parser.device_;

  Kokkos::initialize (args_kokkos);
  {
    RealView1D x, y, z;
    RealOffsetView3D u, un;

    Config conf(nx, ny, nz, nbiter);

    initialize(conf, x, y, z, u, un);

    // Main loop
    timers[Total]->begin();
    for(int i=0; i<conf.nbiter; i++) {
      timers[MainLoop]->begin();
      step(conf, u, un, timers);
      Impl::swap(u, un);
      timers[MainLoop]->end();
    }
    timers[Total]->end();

    using real_type = RealView3D::value_type;
    real_type time = conf.dt * conf.nbiter;
    finalize(conf, time, x, y, z, u, un);
 
    // Measure performance
    performance(conf, timers[Total]->seconds());
  }
  Kokkos::finalize();
  printTimers(timers);
  freeTimers(timers);

  return 0;
}

#include <Kokkos_Core.hpp>
#include <iostream>
#include <chrono>
#include <array>
#include "Types.hpp"
#include "MPI_Comm.hpp"
#include "Config.hpp"
#include "../Parser.hpp"
#include "Init.hpp"
#include "IO.hpp"
#include "Timestep.hpp"

int main(int argc, char *argv[]) {
  Parser parser(argc, argv);
  auto shape = parser.shape_;
  auto topology = parser.topology_;
  int nbiter = parser.nbiter_;
  int freq_diag = parser.freq_diag_;
  bool enable_diag = freq_diag > 0;
  int nx = shape[0], ny = shape[1], nz = shape[2];
  int px = topology[0], py = topology[1], pz = topology[2];
  Config conf(nx, ny, nz, px, py, pz, nbiter, freq_diag);
  Comm comm(argc, argv, shape, topology);

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

    initialize(conf, comm, x, y, z, u, un);

    // Main loop
    timers[Total]->begin();
    for(int i=0; i<conf.nbiter; i++) {
      timers[MainLoop]->begin();
      if(enable_diag) to_csv(conf, comm, u, i, timers);
      step(conf, comm, u, un, timers);
      Impl::swap(u, un);
      timers[MainLoop]->end();
    }
    timers[Total]->end();

    // Check results
    using real_type = RealOffsetView3D::value_type;
    float64 time = conf.dt * conf.nbiter;
    finalize(conf, comm, time, x, y, z, u, un);

    if(comm.is_master()) {
      // Measure performance
      performance(conf, comm, timers[Total]->seconds());
      printTimers(timers);
    }
    comm.cleanup();
  }
  Kokkos::finalize();
  freeTimers(timers);
  comm.finalize();
  return 0;
}

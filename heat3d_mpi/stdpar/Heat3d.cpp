#include <iostream>
#include <chrono>
#include <array>
#include "Types.hpp"
#include "MPI_Comm.hpp"
#include "Config.hpp"
#include "../Parser.hpp"
#include "Init.hpp"
#include "Timestep.hpp"

int main(int argc, char *argv[]) {
  Parser parser(argc, argv);
  auto shape = parser.shape_;
  auto topology = parser.topology_;
  int nbiter = parser.nbiter_;
  int nx = shape[0], ny = shape[1], nz = shape[2];
  int px = topology[0], py = topology[1], pz = topology[2];
  Config conf(nx, ny, nz, px, py, pz, nbiter);
  Comm comm(argc, argv, shape, topology);

  std::vector<Timer*> timers;
  defineTimers(timers);

  RealView1D x, y, z;
  RealView3D u, un;

  initialize(conf, comm, x, y, z, u, un);

  // Main loop
  timers[Total]->begin();
  for(int i=0; i<conf.nbiter; i++) {
    timers[MainLoop]->begin();
    step(conf, comm, u, un, timers);
    u.swap(un);
    timers[MainLoop]->end();
  }
  timers[Total]->end();

  // 1 load, 1 store (assuming the input array u is on cache)
  using real_type = RealView3D::value_type;
  const int n = conf.nx * conf.ny * conf.nz;
  const double size = static_cast<double>( comm.size() );
  double memory_GB = static_cast<double>(n) * size * static_cast<double>(conf.nbiter) * 2 * sizeof(real_type) / 1.e9;

  // 9 Flop per iteration
  double GFlops = static_cast<double>(n) * size * static_cast<double>(conf.nbiter) * 9 / 1.e9;

  // Check results
  Real time = conf.dt * conf.nbiter;
  finalize(conf, comm, time, x, y, z, u, un);

  if(comm.is_master()) {
    // Measure performance
    performance(conf, comm, timers[Total]->seconds());
    printTimers(timers);
  }
  freeTimers(timers);
  comm.cleanup();
  comm.finalize();
}

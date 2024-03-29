#include <iostream>
#include <chrono>
#include <array>
#include "types.hpp"
#include "Config.hpp"
#include "../Parser.hpp"
#include "Init.hpp"
#include "IO.hpp"
#include "Timestep.hpp"

int main(int argc, char *argv[]) {
  Parser parser(argc, argv);
  auto shape = parser.shape_;
  int nbiter = parser.nbiter_;
  int freq_diag = parser.freq_diag_;
  bool enable_diag = freq_diag > 0;
  int nx = shape[0], ny = shape[1], nz = shape[2];

  Config conf(nx, ny, nz, nbiter, freq_diag);

  std::vector<Timer*> timers;
  defineTimers(timers);

  RealView1D x, y, z;
  RealView3D u, un;

  initialize(conf, x, y, z, u, un);

  // Main loop
  timers[Total]->begin();
  for(int i=0; i<conf.nbiter; i++) {
    timers[MainLoop]->begin();
    if(enable_diag) to_csv(conf, u, i, timers);
    step(conf, u, un, timers);
    u.swap(un);
    timers[MainLoop]->end();
  }
  timers[Total]->end();

  using real_type = RealView3D::value_type;
  real_type time = conf.dt * conf.nbiter;
  finalize(conf, time, x, y, z, u, un);

  // Measure performance
  performance(conf, timers[Total]->seconds());
  printTimers(timers);
  freeTimers(timers);

  return 0;
}

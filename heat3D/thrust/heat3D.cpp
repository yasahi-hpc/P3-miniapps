#include <iostream>
#include <chrono>
#include <array>
#include "heat3D.hpp"
#include "types.hpp"
#include "utils.hpp"

int main(int argc, char *argv[]) {
  View1D<Real> x("x", Config::nx), y("y", Config::ny), z("z", Config::nz);
  View3D<Real> u("u", Config::nx, Config::ny, Config::nz), un("un", Config::nx, Config::ny, Config::nz);

  initialize(x, y, z, u, un);

  // Main loop
  auto start = std::chrono::high_resolution_clock::now();
  for(int i=0; i<Config::nbiter; i++) {
    step(u, un);
    u.swap(un);
  }
  synchronize();
  auto end = std::chrono::high_resolution_clock::now();
  double seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();

  // 1 load, 1 store (assuming the input array u is on cache)
  const int n = Config::nx * Config::ny * Config::nz;
  double memory_GB = static_cast<double>(n) * static_cast<double>(Config::nbiter) * 2 * sizeof(Real) / 1.e9;

  // 9 Flop per iteration
  double GFlops = static_cast<double>(n) * static_cast<double>(Config::nbiter) * 9 / 1.e9;

  // Check results
  Real time = Config::dt * Config::nbiter;
  finalize(time, x, y, z, u, un);

  // Measure performance
  performance(GFlops, memory_GB, seconds);
}

#ifndef __HEAT3D_HPP__
#define __HEAT3D_HPP__

#include <iostream>
#include "types.hpp"
#include "tiles.hpp"

namespace Config {
  constexpr int nx = 512;
  constexpr int ny = 512;
  constexpr int nz = 512;
  constexpr int nbiter = 1000;

  // Resolution
  constexpr Real Lx = 1.0;
  constexpr Real Ly = 1.0;
  constexpr Real Lz = 1.0;
  constexpr Real dx = Lx / static_cast<Real>(nx);
  constexpr Real dy = Ly / static_cast<Real>(ny);
  constexpr Real dz = Lz / static_cast<Real>(nz);

  // Physical constants
  constexpr Real umax = 1.0;
  constexpr Real Kappa = 1.0;
  constexpr Real dt = 0.1 * dx * dx / Kappa;
};

// Helper
void performance(double GFlops, double memory_GB, double seconds);

void performance(double GFlops, double memory_GB, double seconds) {
  double bandwidth = memory_GB/seconds;
  std::cout << "Elapsed time " << seconds << " [s]" << std::endl;
  std::cout << "Bandwidth " << bandwidth << " [GB/s]" << std::endl; 
  std::cout << GFlops / seconds << " [GFlops]" << std::endl; 
}

// Main functions
template <class ScalarType>
void initialize(View1D<ScalarType> &x, 
                View1D<ScalarType> &y,
                View1D<ScalarType> &z,
                View3D<ScalarType> &u,
                View3D<ScalarType> &un) {

  auto h_x  = Kokkos::create_mirror_view(x);
  auto h_y  = Kokkos::create_mirror_view(y);
  auto h_z  = Kokkos::create_mirror_view(z);
  auto h_u  = Kokkos::create_mirror_view(u);
  auto h_un = Kokkos::create_mirror_view(un);

  // Initialize in host
  for(int iz=0; iz<Config::nz; iz++) {
    const ScalarType ztmp = static_cast<ScalarType>(iz - Config::nz/2) * Config::dz;
    h_z(iz) = ztmp;
    for(int iy=0; iy<Config::ny; iy++) {
      const ScalarType ytmp = static_cast<ScalarType>(iy - Config::ny/2) * Config::dy;
      h_y(iy) = ytmp;
      for(int ix=0; ix<Config::nx; ix++) {
        const ScalarType xtmp = static_cast<ScalarType>(ix - Config::nx/2) * Config::dx;
        h_x(ix) = xtmp;
        h_u(ix, iy, iz) = Config::umax 
                  * cos(xtmp / Config::Lx * 2.0 * M_PI + ytmp / Config::Ly * 2.0 * M_PI + ztmp / Config::Lz * 2.0 * M_PI);
      }
    }
  }
  Kokkos::deep_copy(x, h_x);
  Kokkos::deep_copy(y, h_y);
  Kokkos::deep_copy(z, h_z);
  Kokkos::deep_copy(u, h_u);
}

template <typename ScalarType>
void finalize(ScalarType time, 
              View1D<ScalarType> &x,
              View1D<ScalarType> &y,
              View1D<ScalarType> &z,
              View3D<ScalarType> &u,
              View3D<ScalarType> &un) {

  auto h_x  = Kokkos::create_mirror_view(x);
  auto h_y  = Kokkos::create_mirror_view(y);
  auto h_z  = Kokkos::create_mirror_view(z);
  auto h_u  = Kokkos::create_mirror_view(u);
  auto h_un = Kokkos::create_mirror_view(un);

  // Analytical solution stored to h_un
  for(int iz=0; iz<Config::nz; iz++) {
    const ScalarType ztmp = h_z(iz);
    for(int iy=0; iy<Config::ny; iy++) {
      const ScalarType ytmp = h_y(iy);
      for(int ix=0; ix<Config::nx; ix++) {
        const ScalarType xtmp = h_x(ix);
        h_un(ix, iy, iz) = Config::umax * cos(xtmp / Config::Lx * 2.0 * M_PI + ytmp / Config::Ly * 2.0 * M_PI + ztmp / Config::Lz * 2.0 * M_PI)
                   * exp(- Config::Kappa * (pow((2.0 * M_PI / Config::Lx), 2) + pow((2.0 * M_PI / Config::Ly), 2) + pow((2.0 * M_PI / Config::Lz), 2) ) * time);
      }
    }
  }
  Kokkos::deep_copy(un, h_un);

  // Check error
  const int nx = Config::nx, ny = Config::ny, nz = Config::nz;
  auto L2norm_kernel = KOKKOS_LAMBDA (const unsigned int i, const unsigned int j, const unsigned int k, ScalarType &sum) {
    auto diff = un(i, j, k) - u(i, j, k);
    sum += diff * diff;
  };

  MDPolicy<3> reduce_policy3d({0, 0, 0},
                              {nx, ny, nz},
                              {32, 8, 1}
                             );
  ScalarType sum = 0;
  Kokkos::parallel_reduce("check_error", reduce_policy3d, L2norm_kernel, sum);
  ScalarType L2_norm = sqrt(sum);
  std::cout << "L2_norm: " << L2_norm << std::endl;
}

template <class ScalarType>
void step(const View3D<ScalarType> &u, View3D<ScalarType> &un) {
  const Real Kappa = Config::Kappa * Config::dt / (Config::dx * Config::dx);
  const int nx = Config::nx, ny = Config::ny, nz = Config::nz;

  auto heat_eq = KOKKOS_LAMBDA (const unsigned int i, const unsigned int j, const unsigned int k) {
    const int ip1 = (i + nx + 1) % nx;
    const int im1 = (i + nx - 1) % nx;
    const int jp1 = (j + ny + 1) % ny;
    const int jm1 = (j + ny - 1) % ny;
    const int kp1 = (k + nz + 1) % nz;
    const int km1 = (k + nz - 1) % nz;

    un(i, j, k) = u(i, j, k) 
      + Kappa * (  u(ip1, j, k) + u(im1, j, k) 
                 + u(i, jp1, k) + u(i, jm1, k) 
                 + u(i, j, kp1) + u(i, j, km1) - 6 * u(i, j, k));
  };

  MDPolicy<3> heat_policy3d({0, 0, 0},
                            {nx, ny, nz},
                            {TILE_SIZE0, TILE_SIZE1, TILE_SIZE2}
                           );
  Kokkos::parallel_for("heat3d", heat_policy3d, heat_eq);
}

#endif

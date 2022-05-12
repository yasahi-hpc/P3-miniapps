#ifndef __INIT_HPP__
#define __INIT_HPP__

#include "Config.hpp"
#include "types.hpp"
#include "tiles.h"

// Prototypes
void initialize(Config &conf,
                RealView1D &x, RealView1D &y, RealView1D &z,
                RealOffsetView3D &u, RealOffsetView3D &un
               );

void finalize(Config &conf, float64 time,
              RealView1D &x, RealView1D &y, RealView1D &z,
              RealOffsetView3D &u, RealOffsetView3D &un
             );

void performance(Config &conf, double seconds);

void initialize(Config &conf,
                RealView1D &x, RealView1D &y, RealView1D &z,
                RealOffsetView3D &u, RealOffsetView3D &un) {
  using real_type = RealView1D::value_type;
  x = RealView1D("x", conf.nx);
  y = RealView1D("y", conf.ny);
  z = RealView1D("z", conf.nz);
 
  u  = RealView3D("u", conf.nx, conf.ny, conf.nz);
  un = RealView3D("un", conf.nx, conf.ny, conf.nz);

  // Print information
  std::cout << "(nx, ny, nz) = " << conf.nx << ", " << conf.ny << ", " << conf.nz << "\n" << std::endl;

  // Initialize at host
  auto h_x  = Kokkos::create_mirror_view(x);
  auto h_y  = Kokkos::create_mirror_view(y);
  auto h_z  = Kokkos::create_mirror_view(z);
  auto h_u  = Kokkos::create_mirror_view(u);
  auto h_un = Kokkos::create_mirror_view(un);

  for(int iz=0; iz<conf.nz; iz++) {
    for(int iy=0; iy<conf.ny; iy++) {
      for(int ix=0; ix<conf.nx; ix++) {
        const real_type xtmp = static_cast<real_type>(ix - conf.nx/2) * conf.dx;
        const real_type ytmp = static_cast<real_type>(iy - conf.ny/2) * conf.dy;
        const real_type ztmp = static_cast<real_type>(iz - conf.nz/2) * conf.dz;

        h_x(ix) = xtmp;
        h_y(iy) = ytmp;
        h_z(iz) = ztmp;
        h_u(ix, iy, iz) = conf.umax
          * cos(xtmp / conf.Lx * 2.0 * M_PI + ytmp / conf.Ly * 2.0 * M_PI + ztmp / conf.Lz * 2.0 * M_PI); 
      }
    }
  }
  Kokkos::deep_copy(x, h_x);
  Kokkos::deep_copy(y, h_y);
  Kokkos::deep_copy(z, h_z);
  Kokkos::deep_copy(u, h_u);
}

void finalize(Config &conf, float64 time,
              RealView1D &x, RealView1D &y, RealView1D &z,
              RealOffsetView3D &u, RealOffsetView3D &un) {
  using real_type = RealOffsetView3D::value_type;
  const int nx = conf.nx;
  const int ny = conf.ny;
  const int nz = conf.nz;
  
  auto h_x  = Kokkos::create_mirror_view(x);
  auto h_y  = Kokkos::create_mirror_view(y);
  auto h_z  = Kokkos::create_mirror_view(z);
  auto h_un = Kokkos::create_mirror_view(un);

  Kokkos::deep_copy(h_x, x);
  Kokkos::deep_copy(h_y, y);
  Kokkos::deep_copy(h_z, z);

  for(int iz=0; iz<nz; iz++) {
    for(int iy=0; iy<ny; iy++) {
      for(int ix=0; ix<nx; ix++) {
        const real_type xtmp = h_x(ix);
        const real_type ytmp = h_y(iy);
        const real_type ztmp = h_z(iz);

        const real_type u_init = conf.umax
          * cos(xtmp / conf.Lx * 2.0 * M_PI + ytmp / conf.Ly * 2.0 * M_PI + ztmp / conf.Lz * 2.0 * M_PI); 
        
        h_un(ix, iy, iz) = u_init * exp(-conf.Kappa * (pow((2.0 * M_PI / conf.Lx), 2) + pow((2.0 * M_PI / conf.Ly), 2) + pow((2.0 * M_PI / conf.Lz), 2) ) * time);
      }
    }
  }
  Kokkos::deep_copy(un, h_un); 

  real_type L2_norm = 0.0;
  MDPolicy<3> reduce_policy3d({0, 0, 0},
                              {nx, ny, nz},
                              {TILE_SIZE0, TILE_SIZE1, TILE_SIZE2}
                             );

  auto L2norm_kernel = KOKKOS_LAMBDA (const int ix, const int iy, const int iz, real_type &sum) {
    auto diff = un(ix, iy, iz) - u(ix, iy, iz);
    sum += diff * diff;
  };
  Kokkos::parallel_reduce("check_error", reduce_policy3d, L2norm_kernel, L2_norm);
  Kokkos::fence();

  std::cout << "L2_norm: " << sqrt(L2_norm) << std::endl;
}

void performance(Config &conf, double seconds) {
  using real_type = RealView3D::value_type;
  const int n = conf.nx * conf.ny * conf.nz;
  double GBytes = static_cast<double>(n) * static_cast<double>(conf.nbiter) * 2 * sizeof(real_type) / 1.e9;

  // 9 Flop per iteration
  double GFlops = static_cast<double>(n) * static_cast<double>(conf.nbiter) * 9 / 1.e9;

  #if defined( KOKKOS_ENABLE_CUDA )
    std::string backend = "CUDA";
  #elif defined( KOKKOS_ENABLE_HIP )
    std::string backend = "HIP";
  #else
    std::string backend = "OPENMP";
  #endif

  std::cout << "Backend: " + backend << std::endl;
  std::cout << "Elapsed time: " << seconds << " [s]" << std::endl;
  std::cout << "Bandwidth: " << GBytes / seconds << " [GB/s]" << std::endl;
  std::cout << "Flops: " << GFlops / seconds << " [GFlops]\n" << std::endl;
}

#endif

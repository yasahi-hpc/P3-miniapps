#ifndef __INIT_HPP__
#define __INIT_HPP__

#include "Config.hpp"
#include "MPI_Comm.hpp"
#include "Types.hpp"
#include "tiles.h"

// Prototypes
void initialize(Config &conf, Comm &comm,
                RealView1D &x, RealView1D &y, RealView1D &z,
                RealOffsetView3D &u, RealOffsetView3D &un
               );

void finalize(Config &conf, Comm &comm, float64 time,
              RealView1D &x, RealView1D &y, RealView1D &z,
              RealOffsetView3D &u, RealOffsetView3D &un
             );

void performance(Config &conf, Comm &comm, float64 seconds);

template <class View3DType>
void testComm(Config &conf, Comm &comm, View3DType &u, View3DType &un) {
  auto cart_rank = comm.cart_rank();
  auto topology  = comm.topology();

  auto h_u  = Kokkos::create_mirror_view(u);
  auto h_un = Kokkos::create_mirror_view(un);

  // fill u and un
  for(int iz=0; iz<conf.nz; iz++) {
    for(int iy=0; iy<conf.ny; iy++) {
      for(int ix=0; ix<conf.nx; ix++) {
        int gix = ix + conf.nx * cart_rank.at(0);
        int giy = iy + conf.ny * cart_rank.at(1);
        int giz = iz + conf.nz * cart_rank.at(2);

        h_u(ix, iy, iz)  = ((double)giz * conf.gny + (double)giy) * conf.gnx + gix;
        h_un(ix, iy, iz) = ((double)giz * conf.gny + (double)giy) * conf.gnx + gix;
      }
    }
  }

  Kokkos::deep_copy(u, h_u);
  Kokkos::deep_copy(un, h_un);

  // Boundary conditions
  std::vector<Timer*> timers;
  comm.exchangeHalos(u, timers);

  Kokkos::deep_copy(h_u, u);

  // Capturing Views causes an error and thus just access to values directly
  auto print_error = [&] (float64 _u, float64 _un, int ix, int iy, int iz, int gix, int giy, int giz) {
    auto diff = _un - _u;
    if (fabs(diff) > .1) {
      printf("Pb at rank %d (%d, %d, %d) u(%d, %d, %d): %lf, un(%d, %d, %d): %lf, error: %lf\n",
             comm.rank(), cart_rank.at(0), cart_rank.at(1), cart_rank.at(2), ix, iy, iz, _u, gix, giy, giz, _un, diff);
    }
  };

  // Fill halos manually
  for(int iz=0; iz<conf.nz; iz++) {
    for(int iy=0; iy<conf.ny; iy++) {
      int gix_left  = 0 + conf.nx * ( ( cart_rank.at(0) + topology.at(0) + 1) % topology.at(0) );
      int gix_right = conf.nx-1 + conf.nx * ( ( cart_rank.at(0) + topology.at(0) - 1 ) % topology.at(0) );
      int giy = iy + conf.ny * cart_rank.at(1);
      int giz = iz + conf.nz * cart_rank.at(2);

      h_un(-1, iy, iz)      = ((double)giz * conf.gny + (double)giy) * conf.gnx + gix_right;
      h_un(conf.nx, iy, iz) = ((double)giz * conf.gny + (double)giy) * conf.gnx + gix_left;
      
      print_error(h_u(-1, iy, iz), h_un(-1, iy, iz), -1, iy, iz, gix_right, giy, giz);
      print_error(h_u(conf.nx, iy, iz), h_un(conf.nx, iy, iz), conf.nx, iy, iz, gix_left, giy, giz);
    }
  }

  for(int iz=0; iz<conf.nz; iz++) {
    for(int ix=0; ix<conf.nx; ix++) {
      int giy_left  = 0 + conf.ny * ( ( cart_rank.at(1) + topology.at(1) + 1 ) % topology.at(1) );
      int giy_right = conf.ny-1 + conf.ny * ( ( cart_rank.at(1) + topology.at(1) - 1 ) % topology.at(1) );
      int gix = ix + conf.nx * cart_rank.at(0);
      int giz = iz + conf.nz * cart_rank.at(2);

      h_un(ix, -1, iz)      = ((double)giz * conf.gny + (double)giy_right) * conf.gnx + gix;
      h_un(ix, conf.ny, iz) = ((double)giz * conf.gny + (double)giy_left) * conf.gnx + gix;
      print_error(h_u(ix, -1, iz), h_un(ix, -1, iz), ix, -1, iz, gix, giy_right, giz);
      print_error(h_u(ix, conf.ny, iz), h_un(ix, conf.ny, iz), ix, conf.ny, iz, gix, giy_left, giz);
    }
  }

  for(int iy=0; iy<conf.ny; iy++) {
    for(int ix=0; ix<conf.nx; ix++) {
      int giz_left  = 0 + conf.nz * ( ( cart_rank.at(2) + topology.at(2) + 1 ) % topology.at(2) );
      int giz_right = conf.nz-1 + conf.nz * ( ( cart_rank.at(2) + topology.at(2) - 1 ) % topology.at(2) );
      int gix = ix + conf.nx * cart_rank.at(0);
      int giy = iy + conf.ny * cart_rank.at(1);

      h_un(ix, iy, -1)      = ((double)giz_right * conf.gny + (double)giy) * conf.gnx + gix;
      h_un(ix, iy, conf.nz) = ((double)giz_left * conf.gny + (double)giy) * conf.gnx + gix;
      print_error(h_u(ix, iy, -1), h_un(ix, iy, -1), ix, iy, -1, gix, giy, giz_right);
      print_error(h_u(ix, iy, conf.nz), h_un(ix, iy, conf.nz), ix, iy, conf.nz, gix, giy, giz_left);
    }
  }
}

void initialize(Config &conf, Comm &comm,
                RealView1D &x, RealView1D &y, RealView1D &z,
                RealOffsetView3D &u, RealOffsetView3D &un) {
  using real_type = RealView1D::value_type;
  x = RealView1D("x", conf.nx);
  y = RealView1D("y", conf.ny);
  z = RealView1D("z", conf.nz);
 
  u  = RealOffsetView3D("u", {-1, conf.nx}, {-1, conf.ny}, {-1, conf.nz});
  un = RealOffsetView3D("u", {-1, conf.nx}, {-1, conf.ny}, {-1, conf.nz});

  comm.setTopology();

  auto cart_rank = comm.cart_rank();
  auto topology  = comm.topology();

  // Print information
  if(comm.is_master()) {
    std::cout << "Parallelization (px, py, pz) = " << topology.at(0) << ", " << topology.at(1) << ", " << topology.at(2) << std::endl;
    std::cout << "Local (nx, ny, nz) = " << conf.nx << ", " << conf.ny << ", " << conf.nz << std::endl;
    std::cout << "Global (nx, ny, nz) = " << conf.gnx << ", " << conf.gny << ", " << conf.gnz << "\n" << std::endl;
  }

  // Test for communication
  testComm(conf, comm, u, un);

  // Initialize at host
  auto h_x  = Kokkos::create_mirror_view(x);
  auto h_y  = Kokkos::create_mirror_view(y);
  auto h_z  = Kokkos::create_mirror_view(z);
  auto h_u  = Kokkos::create_mirror_view(u);
  auto h_un = Kokkos::create_mirror_view(un);

  for(int iz=0; iz<conf.nz; iz++) {
    for(int iy=0; iy<conf.ny; iy++) {
      for(int ix=0; ix<conf.nx; ix++) {
        const real_type xtmp = static_cast<real_type>(ix + conf.nx * cart_rank.at(0) - conf.gnx/2) * conf.dx;
        const real_type ytmp = static_cast<real_type>(iy + conf.ny * cart_rank.at(1) - conf.gny/2) * conf.dy;
        const real_type ztmp = static_cast<real_type>(iz + conf.nz * cart_rank.at(2) - conf.gnz/2) * conf.dz;

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

void finalize(Config &conf, Comm &comm, float64 time,
              RealView1D &x, RealView1D &y, RealView1D &z,
              RealOffsetView3D &u, RealOffsetView3D &un) {
  using real_type = RealOffsetView3D::value_type;
  const int nx = conf.nx;
  const int ny = conf.ny;
  const int nz = conf.nz;
  
  auto topology  = comm.topology();
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

  real_type l2loc = 0.0, l2glob = 0.0;
  MDPolicy<3> reduce_policy3d({0, 0, 0},
                              {nx, ny, nz},
                              {TILE_SIZE0, TILE_SIZE1, TILE_SIZE2}
                             );

  auto L2norm_kernel = KOKKOS_LAMBDA (const int ix, const int iy, const int iz, real_type &sum) {
    auto diff = un(ix, iy, iz) - u(ix, iy, iz);
    sum += diff * diff;
  };
  Kokkos::parallel_reduce("check_error", reduce_policy3d, L2norm_kernel, l2loc);
  Kokkos::fence();

  MPI_Reduce(&l2loc, &l2glob, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if(comm.is_master()) {
    std::cout << "L2_norm: " << sqrt(l2glob) << std::endl;
  }
}

void performance(Config &conf, Comm &comm, double seconds) {
  using real_type = RealView3D::value_type;
  const int n = conf.nx * conf.ny * conf.nz;
  const double size = static_cast<double>( comm.size() );
  double GBytes = static_cast<double>(n) * size * static_cast<double>(conf.nbiter) * 2 * sizeof(real_type) / 1.e9;

  // 9 Flop per iteration
  double GFlops = static_cast<double>(n) * size * static_cast<double>(conf.nbiter) * 9 / 1.e9;

  #if defined( KOKKOS_ENABLE_CUDA ) || defined( KOKKOS_ENABLE_HIP )
    std::string arch = "GPU";
  #else
    std::string arch = "CPU";
  #endif

  std::cout << "Elapsed time: " << seconds << " [s]" << std::endl;
  std::cout << "Bandwidth/" + arch + ": " << GBytes / seconds / comm.size() << " [GB/s]" << std::endl;
  std::cout << "Flops/" + arch + ": " << GFlops / seconds / comm.size() << " [GFlops]\n" << std::endl;
}

#endif

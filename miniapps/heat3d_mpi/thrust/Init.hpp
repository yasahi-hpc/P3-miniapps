#ifndef __INIT_HPP__
#define __INIT_HPP__

#include <cstdio>
#include "Config.hpp"
#include "MPI_Comm.hpp"
#include "Parallel_For.hpp"
#include "Parallel_Reduce.hpp"

// Prototypes
void testComm(Config &conf, Comm &comm, RealView3D &u, RealView3D &un);
void initialize(Config &conf, Comm &comm,
                RealView1D &x, RealView1D &y, RealView1D &z,
                RealView3D &u, RealView3D &un
               );

void finalize(Config &conf, Comm &comm, float64 time,
              RealView1D &x, RealView1D &y, RealView1D &z,
              RealView3D &u, RealView3D &un
             );

void performance(Config &conf, Comm &comm, float64 seconds);

void testComm(Config &conf, Comm &comm, RealView3D &u, RealView3D &un) {
  auto cart_rank = comm.cart_rank();
  auto topology  = comm.topology();

  // fill u and un
  for(int iz=0; iz<conf.nz; iz++) {
    for(int iy=0; iy<conf.ny; iy++) {
      for(int ix=0; ix<conf.nx; ix++) {
        int gix = ix + conf.nx * cart_rank.at(0);
        int giy = iy + conf.ny * cart_rank.at(1);
        int giz = iz + conf.nz * cart_rank.at(2);

        u(ix, iy, iz)  = ((double)giz * conf.gny + (double)giy) * conf.gnx + gix;
        un(ix, iy, iz) = ((double)giz * conf.gny + (double)giy) * conf.gnx + gix;
      }
    }
  }
  u.updateDevice();

  // Boundary conditions
  std::vector<Timer*> timers;
  comm.exchangeHalos(u, timers);
  u.updateSelf();

  auto print_error = [&](int ix, int iy, int iz, int gix, int giy, int giz) {
    auto diff = un(ix, iy, iz) - u(ix, iy, iz);
    if (fabs(diff) > .1) { 
      printf("Pb at rank %d (%d, %d, %d) u(%d, %d, %d): %lf, un(%d, %d, %d): %lf, error: %lf\n",
             comm.rank(), cart_rank.at(0), cart_rank.at(1), cart_rank.at(2), ix, iy, iz, u(ix, iy, iz), gix, giy, giz, un(ix, iy, iz), diff);
    }
  };

  // Fill halos manually
  for(int iz=0; iz<conf.nz; iz++) {
    for(int iy=0; iy<conf.ny; iy++) {
      int gix_left  = 0 + conf.nx * ( ( cart_rank.at(0) + topology.at(0) + 1) % topology.at(0) );
      int gix_right = conf.nx-1 + conf.nx * ( ( cart_rank.at(0) + topology.at(0) - 1 ) % topology.at(0) );
      int giy = iy + conf.ny * cart_rank.at(1);
      int giz = iz + conf.nz * cart_rank.at(2);

      un(-1, iy, iz)      = ((double)giz * conf.gny + (double)giy) * conf.gnx + gix_right;
      un(conf.nx, iy, iz) = ((double)giz * conf.gny + (double)giy) * conf.gnx + gix_left;

      print_error(-1, iy, iz, gix_right, giy, giz);
      print_error(conf.nx, iy, iz, gix_left, giy, giz);
    }
  }

  for(int iz=0; iz<conf.nz; iz++) {
    for(int ix=0; ix<conf.nx; ix++) {
      int giy_left  = 0 + conf.ny * ( ( cart_rank.at(1) + topology.at(1) + 1 ) % topology.at(1) );
      int giy_right = conf.ny-1 + conf.ny * ( ( cart_rank.at(1) + topology.at(1) - 1 ) % topology.at(1) );
      int gix = ix + conf.nx * cart_rank.at(0);
      int giz = iz + conf.nz * cart_rank.at(2);

      un(ix, -1, iz)      = ((double)giz * conf.gny + (double)giy_right) * conf.gnx + gix;
      un(ix, conf.ny, iz) = ((double)giz * conf.gny + (double)giy_left) * conf.gnx + gix;
      print_error(ix, -1, iz, gix, giy_right, giz);
      print_error(ix, conf.ny, iz, gix, giy_left, giz);
    }
  }

  for(int iy=0; iy<conf.ny; iy++) {
    for(int ix=0; ix<conf.nx; ix++) {
      int giz_left  = 0 + conf.nz * ( ( cart_rank.at(2) + topology.at(2) + 1 ) % topology.at(2) );
      int giz_right = conf.nz-1 + conf.nz * ( ( cart_rank.at(2) + topology.at(2) - 1 ) % topology.at(2) );
      int gix = ix + conf.nx * cart_rank.at(0);
      int giy = iy + conf.ny * cart_rank.at(1);

      un(ix, iy, -1)      = ((double)giz_right * conf.gny + (double)giy) * conf.gnx + gix;
      un(ix, iy, conf.nz) = ((double)giz_left * conf.gny + (double)giy) * conf.gnx + gix;
      print_error(ix, iy, -1, gix, giy, giz_right);
      print_error(ix, iy, conf.nz, gix, giy, giz_left);
    }
  }
}

void initialize(Config &conf, Comm &comm,
                RealView1D &x, RealView1D &y, RealView1D &z,
                RealView3D &u, RealView3D &un
               ) {
  using real_type = RealView1D::value_type;
  x = RealView1D("x", conf.nx);
  y = RealView1D("y", conf.ny);
  z = RealView1D("z", conf.nz);
 
  const size_type nx_halo = conf.nx+2;
  const size_type ny_halo = conf.ny+2;
  const size_type nz_halo = conf.nz+2;

  using shape_3d = std::array<size_type, 3>;
  using range_3d = std::array<int, 3>;

  u  = RealView3D("u",  shape_3d{nx_halo, ny_halo, nz_halo}, range_3d{-1, -1, -1});
  un = RealView3D("un", shape_3d{nx_halo, ny_halo, nz_halo}, range_3d{-1, -1, -1});

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
  for(int iz=0; iz<conf.nz; iz++) {
    for(int iy=0; iy<conf.ny; iy++) {
      for(int ix=0; ix<conf.nx; ix++) {
        const real_type xtmp = static_cast<real_type>(ix + conf.nx * cart_rank.at(0) - conf.gnx/2) * conf.dx;
        const real_type ytmp = static_cast<real_type>(iy + conf.ny * cart_rank.at(1) - conf.gny/2) * conf.dy;
        const real_type ztmp = static_cast<real_type>(iz + conf.nz * cart_rank.at(2) - conf.gnz/2) * conf.dz;

        x(ix) = xtmp;
        y(iy) = ytmp;
        z(iz) = ztmp;
        u(ix, iy, iz) = conf.umax
          * cos(xtmp / conf.Lx * 2.0 * M_PI + ytmp / conf.Ly * 2.0 * M_PI + ztmp / conf.Lz * 2.0 * M_PI); 
      }
    }
  }

  // Deep copy to device
  x.updateDevice();
  y.updateDevice();
  z.updateDevice();
  u.updateDevice();
}

void finalize(Config &conf, Comm &comm, float64 time,
              RealView1D &x, RealView1D &y, RealView1D &z,
              RealView3D &u, RealView3D &un
             ) {
  using real_type = RealView1D::value_type;
  const int nx = conf.nx;
  const int ny = conf.ny;
  const int nz = conf.nz;

  x.updateSelf();
  y.updateSelf();
  z.updateSelf();
  
  auto topology  = comm.topology();
  for(int iz=0; iz<nz; iz++) {
    for(int iy=0; iy<ny; iy++) {
      for(int ix=0; ix<nx; ix++) {
        const real_type xtmp = x(ix);
        const real_type ytmp = y(iy);
        const real_type ztmp = z(iz);

        const real_type u_init = conf.umax
          * cos(xtmp / conf.Lx * 2.0 * M_PI + ytmp / conf.Ly * 2.0 * M_PI + ztmp / conf.Lz * 2.0 * M_PI); 
        
        un(ix, iy, iz) = u_init * exp(-conf.Kappa * (pow((2.0 * M_PI / conf.Lx), 2) + pow((2.0 * M_PI / conf.Ly), 2) + pow((2.0 * M_PI / conf.Lz), 2) ) * time);
      }
    }
  }
  un.updateDevice();

  float64 l2loc = 0.0;
  auto _u  = u.mdspan();
  auto _un = un.mdspan();

  int3 begin = make_int3(0, 0, 0);
  int3 end   = make_int3(conf.nx, conf.ny, conf.nz);

  auto L2norm_kernel = [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix, const int iy, const int iz) {
    auto diff = _un(ix, iy, iz) - _u(ix, iy, iz);
    return diff * diff;
  };

  Impl::transform_reduce<default_iterate_layout>(begin, end, thrust::plus<float64>(), L2norm_kernel, l2loc);
  synchronize();

  float64 l2glob = 0.;
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

  #if defined( ENABLE_CUDA ) || defined( ENABLE_HIP )
    std::string arch = "GPU";
  #else
    std::string arch = "CPU";
  #endif

  #if defined( ENABLE_CUDA ) && ! defined( ENABLE_THRUST )
    std::string backend = "CUDA";
  #elif defined( ENABLE_HIP ) && ! defined( ENABLE_THRUST )
    std::string backend = "HIP";
  #elif defined( ENABLE_OPENMP ) && ! defined( ENABLE_THRUST )
    std::string backend = "OPENMP";
  #else
    std::string backend = "THRUST";
  #endif

  #if defined(RANGE_POLICY_1D)
    std::cout << "1D Range policy" << std::endl;
    #if defined(ACCESS_VIA_RAW_POINTERS)
      std::cout << "Access via raw pointers" << std::endl;
    #endif
  #else
    std::cout << "3D Range policy" << std::endl;
  #endif

  std::cout << backend + " backend" << std::endl;
  std::cout << "Elapsed time: " << seconds << " [s]" << std::endl;
  std::cout << "Bandwidth/" + arch + ": " << GBytes / seconds / comm.size() << " [GB/s]" << std::endl;
  std::cout << "Flops/" + arch + ": " << GFlops / seconds / comm.size() << " [GFlops]\n" << std::endl;
}

#endif

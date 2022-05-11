#ifndef __INIT_HPP__
#define __INIT_HPP__

#include <cstdio>
#include "Config.hpp"
#include "boundary.hpp"
#include "Parallel_For.hpp"
#include "Parallel_Reduce.hpp"

// Prototypes
void initialize(Config &conf, Boundary &boundary,
                RealView1D &x, RealView1D &y, RealView1D &z,
                RealView3D &u, RealView3D &un
               );

void finalize(Config &conf, Boundary &boundary, float64 time,
              RealView1D &x, RealView1D &y, RealView1D &z,
              RealView3D &u, RealView3D &un
             );

void performance(Config &conf, float64 seconds);

void initialize(Config &conf, Boundary &boundary,
                RealView1D &x, RealView1D &y, RealView1D &z,
                RealView3D &u, RealView3D &un
               ) {
  using real_type = RealView1D::value_type;
  x = RealView1D("x", conf.nx);
  y = RealView1D("y", conf.ny);
  z = RealView1D("z", conf.nz);
 
  const size_t nx_halo = conf.nx+2;
  const size_t ny_halo = conf.ny+2;
  const size_t nz_halo = conf.nz+2;

  using shape_3d = std::array<size_t, 3>;
  using range_3d = std::array<int, 3>;

  u  = RealView3D("u",  shape_3d{nx_halo, ny_halo, nz_halo}, range_3d{-1, -1, -1});
  un = RealView3D("un", shape_3d{nx_halo, ny_halo, nz_halo}, range_3d{-1, -1, -1});

  // Print information
  std::cout << "(nx, ny, nz) = " << conf.nx << ", " << conf.ny << ", " << conf.nz << "\n" << std::endl;

  // Initialize at host
  for(int iz=0; iz<conf.nz; iz++) {
    for(int iy=0; iy<conf.ny; iy++) {
      for(int ix=0; ix<conf.nx; ix++) {
        const real_type xtmp = static_cast<real_type>(ix - conf.nx/2) * conf.dx;
        const real_type ytmp = static_cast<real_type>(iy - conf.ny/2) * conf.dy;
        const real_type ztmp = static_cast<real_type>(iz - conf.nz/2) * conf.dz;

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

void finalize(Config &conf, Boundary &boundary, float64 time,
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

  float64 L2_norm = 0.0;
  auto _u  = u.mdspan();
  auto _un = un.mdspan();

  int3 begin = make_int3(0, 0, 0);
  int3 end   = make_int3(conf.nx, conf.ny, conf.nz);

  auto L2norm_kernel = [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix, const int iy, const int iz) {
    auto diff = _un(ix, iy, iz) - _u(ix, iy, iz);
    return diff * diff;
  };

  Impl::transform_reduce<default_iterate_layout>(begin, end, thrust::plus<float64>(), L2norm_kernel, L2_norm);
  synchronize();

  std::cout << "L2_norm: " << sqrt(L2_norm) << std::endl;
}

void performance(Config &conf, double seconds) {
  using real_type = RealView3D::value_type;
  const int n = conf.nx * conf.ny * conf.nz;
  double GBytes = static_cast<double>(n) * static_cast<double>(conf.nbiter) * 2 * sizeof(real_type) / 1.e9;

  // 9 Flop per iteration
  double GFlops = static_cast<double>(n) * static_cast<double>(conf.nbiter) * 9 / 1.e9;

  #if defined( ENABLE_CUDA ) && ! defined( ENABLE_THRUST )
    std::string backend = "CUDA";
  #elif defined( ENABLE_HIP ) && ! defined( ENABLE_THRUST )
    std::string backend = "HIP";
  #elif defined( ENABLE_OPENMP ) && ! defined( ENABLE_THRUST )
    std::string backend = "OPENMP";
  #else
    std::string backend = "THRUST";
  #endif

  std::cout << "Backend: " + backend << std::endl;
  std::cout << "Elapsed time: " << seconds << " [s]" << std::endl;
  std::cout << "Bandwidth: " << GBytes / seconds << " [GB/s]" << std::endl;
  std::cout << "Flops: " << GFlops / seconds << " [GFlops]\n" << std::endl;
}

#endif

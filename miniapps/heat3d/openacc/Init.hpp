#ifndef __INIT_HPP__
#define __INIT_HPP__

#include "Config.hpp"
#include "types.hpp"

// Prototypes
void initialize(Config &conf,
                RealView1D &x, RealView1D &y, RealView1D &z,
                RealView3D &u, RealView3D &un
               );

void finalize(Config &conf, float64 time,
              RealView1D &x, RealView1D &y, RealView1D &z,
              RealView3D &u, RealView3D &un
             );

void performance(Config &conf, double seconds);

void initialize(Config &conf,
                RealView1D &x, RealView1D &y, RealView1D &z,
                RealView3D &u, RealView3D &un
               ) {
  using real_type = RealView1D::value_type;
  x = RealView1D("x", conf.nx);
  y = RealView1D("y", conf.ny);
  z = RealView1D("z", conf.nz);

  u  = RealView3D("u",  conf.nx, conf.ny, conf.nz);
  un = RealView3D("un", conf.nx, conf.ny, conf.nz);

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

  x.updateDevice();
  y.updateDevice();
  z.updateDevice();
  u.updateDevice();
}

void finalize(Config &conf, float64 time,
              RealView1D &x, RealView1D &y, RealView1D &z,
              RealView3D &u, RealView3D &un
             ) {
  using real_type = RealView1D::value_type;
  const int nx = conf.nx;
  const int ny = conf.ny;
  const int nz = conf.nz;
  
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
  #if defined(ENABLE_OPENACC)
    #pragma acc parallel loop independent collapse(2) reduction(+:L2_norm)
  #else
    #pragma omp parallel for collapse(2) reduction(+:L2_norm)
  #endif
  for(int k=0; k<nz; k++) {
    for(int j=0; j<ny; j++) {
      LOOP_SIMD
      for(int i=0; i<nx; i++) {
        auto diff = un(i, j, k) - u(i, j, k);
        L2_norm += diff * diff;
      }
    }
  }
  std::cout << "L2_norm: " << sqrt(L2_norm) << std::endl;
}

void performance(Config &conf, double seconds) {
  using real_type = RealView3D::value_type;
  const int n = conf.nx * conf.ny * conf.nz;
  double GBytes = static_cast<double>(n) * static_cast<double>(conf.nbiter) * 2 * sizeof(real_type) / 1.e9;

  // 9 Flop per iteration
  double GFlops = static_cast<double>(n) * static_cast<double>(conf.nbiter) * 9 / 1.e9;

  #if defined(ENABLE_OPENACC)
    std::string backend = "OPENACC";
  #elif defined(ENABLE_OPENMP)
    std::string backend = "OPENMP";
  #else
    std::string backend = "OPENMP";
  #endif

  std::cout << "Backend: " + backend << std::endl;
  std::cout << "Elapsed time: " << seconds << " [s]" << std::endl;
  std::cout << "Bandwidth: " << GBytes / seconds << " [GB/s]" << std::endl;
  std::cout << "Flops: " << GFlops / seconds << " [GFlops]" << std::endl;
}

#endif

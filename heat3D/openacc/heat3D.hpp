#ifndef __HEAT3D_HPP__
#define __HEAT3D_HPP__

#include <iostream>
#include "types.hpp"

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

  // Initialize in host
  for(int iz=0; iz<Config::nz; iz++) {
    const ScalarType ztmp = static_cast<ScalarType>(iz - Config::nz/2) * Config::dz;
    z(iz) = ztmp;
    for(int iy=0; iy<Config::ny; iy++) {
      const ScalarType ytmp = static_cast<ScalarType>(iy - Config::ny/2) * Config::dy;
      y(iy) = ytmp;
      for(int ix=0; ix<Config::nx; ix++) {
        const ScalarType xtmp = static_cast<ScalarType>(ix - Config::nx/2) * Config::dx;
        x(ix) = xtmp;
        u(ix, iy, iz) = Config::umax 
                  * cos(xtmp / Config::Lx * 2.0 * M_PI + ytmp / Config::Ly * 2.0 * M_PI + ztmp / Config::Lz * 2.0 * M_PI);
      }
    }
  }
  x.updateDevice();
  y.updateDevice();
  z.updateDevice();
  u.updateDevice();
}

template <typename ScalarType>
void finalize(ScalarType time, 
              View1D<ScalarType> &x,
              View1D<ScalarType> &y,
              View1D<ScalarType> &z,
              View3D<ScalarType> &u,
              View3D<ScalarType> &un) {

  // Analytical solution stored to h_un
  for(int iz=0; iz<Config::nz; iz++) {
    const ScalarType ztmp = z(iz);
    for(int iy=0; iy<Config::ny; iy++) {
      const ScalarType ytmp = y(iy);
      for(int ix=0; ix<Config::nx; ix++) {
        const ScalarType xtmp = x(ix);
        un(ix, iy, iz) = Config::umax * cos(xtmp / Config::Lx * 2.0 * M_PI + ytmp / Config::Ly * 2.0 * M_PI + ztmp / Config::Lz * 2.0 * M_PI)
                   * exp(- Config::Kappa * (pow((2.0 * M_PI / Config::Lx), 2) + pow((2.0 * M_PI / Config::Ly), 2) + pow((2.0 * M_PI / Config::Lz), 2) ) * time);
      }
    }
  }
  un.updateDevice();

  // Check error
  ScalarType sum = 0;
  #if defined( ENABLE_OPENACC )
    #pragma acc data present(u, un)
    #pragma acc parallel loop collapse(2) reduction(+:sum)
  #else
    #pragma omp parallel for collapse(2) reduction(+:sum)
  #endif
  for(int k=0; k<Config::nz; k++) {
    for(int j=0; j<Config::ny; j++) {
      LOOP_SIMD
      for(int i=0; i<Config::nx; i++) {
        auto diff = un(i, j, k) - u(i, j, k);
        sum += diff * diff;
      }
    }
  }

  ScalarType L2_norm = sqrt(sum);
  std::cout << "L2_norm: " << L2_norm << std::endl;
}

template <class ScalarType>
void step(const View3D<ScalarType> &u, View3D<ScalarType> &un) {
  const Real Kappa = Config::Kappa * Config::dt / (Config::dx * Config::dx);
  const int nx = Config::nx, ny = Config::ny, nz = Config::nz;

  #if defined(ENABLE_LOOP_3D)
    #if defined( ENABLE_OPENACC )
      #pragma acc data present(u, un)
      #pragma acc parallel loop collapse(2)
    #else
      #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for(int k=0; k<nz; k++) {
      for(int j=0; j<ny; j++) {
        LOOP_SIMD
        for(int i=0; i<nx; i++) {
          const int ip1 = (i + nx + 1) % nx;
          const int im1 = (i + nx - 1) % nx;
          const int jp1 = (j + ny + 1) % ny;
          const int jm1 = (j + ny - 1) % ny;
          const int kp1 = (k + nz + 1) % nz;
          const int km1 = (k + nz - 1) % nz;

          un(i, j, k) = u(i, j, k) 
            + Kappa * ( u(ip1, j, k) + u(im1, j, k) 
                      + u(i, jp1, k) + u(i, jm1, k) 
                      + u(i, j, kp1) + u(i, j, km1) - 6 * u(i, j, k));
        }
      }
    }

  #else
    const ScalarType *_u = u.data();
    ScalarType *_un = un.data();
    const int n = nx*ny*nz;
    #if defined( ENABLE_OPENACC )
      #pragma acc data present(_u, _un)
      #pragma acc parallel loop
    #else
      #pragma omp parallel for
    #endif
    for(int idx=0; idx<n; idx++) {
      const int ix  = idx % nx;
      const int iyz = idx / nx;
      const int iy  = iyz % ny;
      const int iz  = iyz / ny;

      const int ip1 = (ix + nx + 1) % nx;
      const int im1 = (ix + nx - 1) % nx;
      const int jp1 = (iy + ny + 1) % ny;
      const int jm1 = (iy + ny - 1) % ny;
      const int kp1 = (iz + nz + 1) % nz;
      const int km1 = (iz + nz - 1) % nz;

      _un[idx] = _u[idx]
        + Kappa * (  _u[ip1 + iy*nx  + iz*nx*ny]  + _u[im1 + iy*nx  + iz*nx*ny]
                   + _u[ix  + jp1*nx + iz*nx*ny]  + _u[ix  + jm1*nx + iz*nx*ny]
                   + _u[ix  + iy*nx  + kp1*nx*ny] + _u[ix  + iy*nx  + km1*nx*ny] - 6 * _u[idx]);
    }
  #endif
}

#endif

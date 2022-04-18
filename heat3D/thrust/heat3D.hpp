#ifndef __HEAT3D_HPP__
#define __HEAT3D_HPP__

#include <iostream>
#include <numeric>
#include <algorithm>
#include "types.hpp"
#include "thrust_parallel_for.hpp"

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
  auto u_ptr  = u.device_data();
  auto un_ptr = un.device_data();
  const int n = u.size();
  ScalarType L2_norm = sqrt( thrust::transform_reduce(thrust::device,
                                                      counting_iterator(0),
                                                      counting_iterator(0) + n,
                                                      [=] MDSPAN_FORCE_INLINE_FUNCTION (const unsigned int i) {
                                                        auto diff = u_ptr[i] - un_ptr[i];
                                                        return diff*diff;
                                                      },
                                                      0.0,
                                                      thrust::plus<ScalarType>()
                                                     ));
  std::cout << "L2_norm: " << L2_norm << std::endl;
}

template <class ScalarType>
void step(const View3D<ScalarType> &u, View3D<ScalarType> &un) {
  const Real Kappa = Config::Kappa * Config::dt / (Config::dx * Config::dx);
  const int nx = Config::nx, ny = Config::ny, nz = Config::nz;

  #if defined(ENABLE_MDSPAN)
    // Access to the data through mdspans
    auto _u = u.device_mdspan();
    auto _un = un.device_mdspan();

    auto heat_eq = [=] MDSPAN_FORCE_INLINE_FUNCTION (const int i, const int j, const int k) {
      const int ip1 = (i + nx + 1) % nx;
      const int im1 = (i + nx - 1) % nx;
      const int jp1 = (j + ny + 1) % ny;
      const int jm1 = (j + ny - 1) % ny;
      const int kp1 = (k + nz + 1) % nz;
      const int km1 = (k + nz - 1) % nz;

      _un(i, j, k) = _u(i, j, k) 
        + Kappa * (  _u(ip1, j, k) + _u(im1, j, k) 
                   + _u(i, jp1, k) + _u(i, jm1, k) 
                   + _u(i, j, kp1) + _u(i, j, km1) - 6 * _u(i, j, k));
    };

    //Iterate_policy<3> policy3d({0, 0, 0}, {nx, ny, nz});
    //Impl::for_each(policy3d, heat_eq);
    const int3 begin = make_int3(0, 0, 0);
    const int3 end   = make_int3(nx, ny, nz);
    Impl::for_each(begin, end, heat_eq);

  #else
    // Access to the data through raw pointers
    const ScalarType *_u = u.device_data();
    ScalarType *_un = un.device_data();

    auto heat_eq = [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
      const int ix = idx % nx;
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
    };

    //Iterate_policy<1> policy1d(nx*ny*nz);
    //Impl::for_each(policy1d, heat_eq);
    const int1 begin = make_int1(0);
    const int1 end   = make_int1(nx*ny*nz);
    Impl::for_each(begin, end, heat_eq);
  #endif
}

#endif

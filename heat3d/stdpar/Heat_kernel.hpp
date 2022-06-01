#ifndef __HEAT_KERNEL_HPP__
#define __HEAT_KERNEL_HPP__

#include "Config.hpp"
#include "types.hpp"

template < class LayoutPolicy >
struct heat_1d_functor {
  using value_type = RealView3D::value_type;
  using layout_type = LayoutPolicy;
  value_type* u_;
  value_type* un_;
  float64 coef_;
  int nx_, ny_, nz_;

  heat_1d_functor(Config &conf, RealView3D &u, RealView3D &un) {
    u_  = u.data();
    un_ = un.data();

    coef_ = conf.Kappa * conf.dt / (conf.dx*conf.dx);
    nx_ = conf.nx;
    ny_ = conf.ny;
    nz_ = conf.nz;
  }

  template <class L=layout_type, std::enable_if_t<std::is_same_v<L, stdex::layout_left>, std::nullptr_t> = nullptr>
  void operator()(const int idx) const {
    const int ix  = idx % nx_;
    const int iyz = idx / nx_;
    const int iy  = iyz % ny_;
    const int iz  = iyz / ny_;

    const int ixp1 = (ix + nx_ + 1) % nx_;
    const int ixm1 = (ix + nx_ - 1) % nx_;
    const int iyp1 = (iy + ny_ + 1) % ny_;
    const int iym1 = (iy + ny_ - 1) % ny_;
    const int izp1 = (iz + nz_ + 1) % nz_;
    const int izm1 = (iz + nz_ - 1) % nz_;

    un_[idx] = u_[idx]
             + coef_ * ( u_[ixp1 + iy*nx_ + iz*nx_*ny_] + u_[ixm1 + iy*nx_ + iz*nx_*ny_]
                       + u_[ix + iyp1*nx_ + iz*nx_*ny_] + u_[ix + iym1*nx_ + iz*nx_*ny_]
                       + u_[ix + iy*nx_ + izp1*nx_*ny_] + u_[ix + iy*nx_ + izm1*nx_*ny_]
                       - 6. * u_[idx] );
  }

  template <class L=layout_type, std::enable_if_t<std::is_same_v<L, stdex::layout_right>, std::nullptr_t> = nullptr>
  void operator()(const int idx) const {
    const int iz  = idx % nz_;
    const int ixy = idx / nz_;
    const int iy  = ixy % ny_;
    const int ix  = ixy / ny_;

    const int ixp1 = (ix + nx_ + 1) % nx_;
    const int ixm1 = (ix + nx_ - 1) % nx_;
    const int iyp1 = (iy + ny_ + 1) % ny_;
    const int iym1 = (iy + ny_ - 1) % ny_;
    const int izp1 = (iz + nz_ + 1) % nz_;
    const int izm1 = (iz + nz_ - 1) % nz_;

    un_[idx] = u_[idx]
             + coef_ * ( u_[izp1 + iy*nz_ + ix*nz_*ny_] + u_[izm1 + iy*nz_ + ix*nz_*ny_]
                       + u_[iz + iyp1*nz_ + ix*nz_*ny_] + u_[iz + iym1*nz_ + ix*nz_*ny_]
                       + u_[iz + iy*nz_ + ixp1*nz_*ny_] + u_[iz + iy*nz_ + ixm1*nz_*ny_]
                       - 6. * u_[idx] );
  }
};

struct heat_functor {
  using mdspan3d_type = RealView3D::mdspan_type;
  mdspan3d_type u_, un_;
  float64 coef_;
  int nx_, ny_, nz_;
 
  heat_functor(Config &conf, RealView3D &u, RealView3D &un) {
    u_  = u.mdspan();
    un_ = un.mdspan();
 
    coef_ = conf.Kappa * conf.dt / (conf.dx*conf.dx);
    nx_ = conf.nx;
    ny_ = conf.ny;
    nz_ = conf.nz;
  }

  void operator()(const int ix, const int iy, const int iz) const {
    const int ixp1 = (ix + nx_ + 1) % nx_;
    const int ixm1 = (ix + nx_ - 1) % nx_;
    const int iyp1 = (iy + ny_ + 1) % ny_;
    const int iym1 = (iy + ny_ - 1) % ny_;
    const int izp1 = (iz + nz_ + 1) % nz_;
    const int izm1 = (iz + nz_ - 1) % nz_;
 
    un_(ix, iy, iz) = u_(ix, iy, iz)
                    + coef_ * ( u_(ixp1, iy, iz) + u_(ixm1, iy, iz)
                              + u_(ix, iyp1, iz) + u_(ix, iym1, iz)
                              + u_(ix, iy, izp1) + u_(ix, iy, izm1)
                              - 6. * u_(ix, iy, iz) );
  }
};

#endif

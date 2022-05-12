#ifndef __HEAT_KERNEL_HPP__
#define __HEAT_KERNEL_HPP__

#include "Config.hpp"
#include "types.hpp"

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

  MDSPAN_FORCE_INLINE_FUNCTION
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

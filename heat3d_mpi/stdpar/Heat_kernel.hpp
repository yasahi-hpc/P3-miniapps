#ifndef __HEAT_KERNEL_HPP__
#define __HEAT_KERNEL_HPP__

#include "Config.hpp"
#include "Types.hpp"

struct heat_functor {
  using mdspan3d_type = RealView3D::mdspan_type;
  mdspan3d_type u_, un_;
  float64 coef_;
 
  heat_functor(Config &conf, RealView3D &u, RealView3D &un) {
    u_  = u.mdspan();
    un_ = un.mdspan();
 
    coef_ = conf.Kappa * conf.dt / (conf.dx*conf.dx);
  }

  void operator()(const int ix, const int iy, const int iz) const {
    un_(ix, iy, iz) = u_(ix, iy, iz)
                    + coef_ * ( u_(ix+1, iy, iz) + u_(ix-1, iy, iz)
                              + u_(ix, iy+1, iz) + u_(ix, iy-1, iz)
                              + u_(ix, iy, iz+1) + u_(ix, iy, iz-1)
                              - 6. * u_(ix, iy, iz) );
  }
};

#endif

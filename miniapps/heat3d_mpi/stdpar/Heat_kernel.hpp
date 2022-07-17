#ifndef __HEAT_KERNEL_HPP__
#define __HEAT_KERNEL_HPP__

#include "Config.hpp"
#include "Types.hpp"

template < class LayoutPolicy >
struct heat_1d_functor {
  using mdspan3d_type = RealView3D::mdspan_type;
  using value_type = RealView3D::value_type;
  using layout_type = LayoutPolicy;

  #if defined(ACCESS_VIA_RAW_POINTERS)
    value_type* u_;
    value_type* un_;
  #else
    mdspan3d_type u_, un_;
  #endif

  float64 coef_;
  int nx_, ny_, nz_;
  int nx_halo_, ny_halo_, nz_halo_;

  heat_1d_functor(Config &conf, RealView3D &u, RealView3D &un) {
    #if defined(ACCESS_VIA_RAW_POINTERS)
      u_  = u.data();
      un_ = un.data();
    #else
      u_  = u.mdspan();
      un_ = un.mdspan();
    #endif

    coef_ = conf.Kappa * conf.dt / (conf.dx*conf.dx);
    nx_ = conf.nx;
    ny_ = conf.ny;
    nz_ = conf.nz;
    nx_halo_ = nx_ + 2;
    ny_halo_ = ny_ + 2;
    nz_halo_ = nz_ + 2;
  }

  template <class L=layout_type, std::enable_if_t<std::is_same_v<L, stdex::layout_left>, std::nullptr_t> = nullptr>
  void operator()(const int idx) const {
    #if defined(ACCESS_VIA_RAW_POINTERS)
      const int ix  = idx % nx_ + 1;
      const int iyz = idx / nx_;
      const int iy  = iyz % ny_ + 1;
      const int iz  = iyz / ny_ + 1;
      const int _idx = ix + iy*nx_halo_ + iz*nx_halo_*ny_halo_;

      un_[_idx] = u_[_idx]
               + coef_ * ( u_[ix+1 + iy*nx_halo_ + iz*nx_halo_*ny_halo_] + u_[ix-1 + iy*nx_halo_ + iz*nx_halo_*ny_halo_]
                         + u_[ix + (iy+1)*nx_halo_ + iz*nx_halo_*ny_halo_] + u_[ix + (iy-1)*nx_halo_ + iz*nx_halo_*ny_halo_]
                         + u_[ix + iy*nx_halo_ + (iz+1)*nx_halo_*ny_halo_] + u_[ix + iy*nx_halo_ + (iz-1)*nx_halo_*ny_halo_]
                         - 6. * u_[_idx] );
    #else
      const int ix  = idx % nx_;
      const int iyz = idx / nx_;
      const int iy  = iyz % ny_;
      const int iz  = iyz / ny_;

      un_(ix, iy, iz) = u_(ix, iy, iz)
                      + coef_ * ( u_(ix+1, iy, iz) + u_(ix-1, iy, iz)
                                + u_(ix, iy+1, iz) + u_(ix, iy-1, iz)
                                + u_(ix, iy, iz+1) + u_(ix, iy, iz-1)
                                - 6. * u_(ix, iy, iz) );
    #endif
  }

  template <class L=layout_type, std::enable_if_t<std::is_same_v<L, stdex::layout_right>, std::nullptr_t> = nullptr>
  void operator()(const int idx) const {
    #if defined(ACCESS_VIA_RAW_POINTERS)
      const int iz  = idx % nz_ + 1;
      const int ixy = idx / nz_;
      const int iy  = ixy % ny_ + 1;
      const int ix  = ixy / ny_ + 1;
      const int _idx = iz + iy*nz_halo_ + ix*nz_halo_*ny_halo_;

      un_[_idx] = u_[_idx]
               + coef_ * ( u_[iz + iy*nz_halo_ + (ix+1)*nz_halo_*ny_halo_] + u_[iz + iy*nz_halo_ + (ix-1)*nz_halo_*ny_halo_]
                         + u_[iz + (iy+1)*nz_halo_ + ix*nz_halo_*ny_halo_] + u_[iz + (iy-1)*nz_halo_ + ix*nz_halo_*ny_halo_]
                         + u_[iz+1 + iy*nz_halo_ + ix*nz_halo_*ny_halo_]   + u_[iz-1 + iy*nz_halo_ + ix*nz_halo_*ny_halo_]
                         - 6. * u_[_idx] );
    #else
      const int iz  = idx % nz_;
      const int ixy = idx / nz_;
      const int iy  = ixy % ny_;
      const int ix  = ixy / ny_;
      un_(ix, iy, iz) = u_(ix, iy, iz)
                      + coef_ * ( u_(ix+1, iy, iz) + u_(ix-1, iy, iz)
                                + u_(ix, iy+1, iz) + u_(ix, iy-1, iz)
                                + u_(ix, iy, iz+1) + u_(ix, iy, iz-1)
                                - 6. * u_(ix, iy, iz) );
    #endif
  }
};

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

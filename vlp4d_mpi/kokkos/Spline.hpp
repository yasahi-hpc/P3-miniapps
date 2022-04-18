#ifndef __SPLINE_HPP__
#define __SPLINE_HPP__

#include "Config.hpp"
#include "Types.hpp"
#include "Communication.hpp"
#include "tiles.h"
#include "Transpose.hpp"
#include "Math.hpp"

struct Spline {
  using array_layout = typename RealOffsetView4D::array_layout;
  Impl::Transpose<float64, array_layout> *transpose_;

private:
  RealOffsetView4D fn_trans_;
  int nx_, ny_, nvx_, nvy_;

public:
  Spline(Config *conf) {
    Domain* dom = &conf->dom_;
    nx_  = dom->local_nx_[0] + HALO_PTS * 2;
    ny_  = dom->local_nx_[1] + HALO_PTS * 2;
    nvx_ = dom->local_nx_[2] + HALO_PTS * 2;
    nvy_ = dom->local_nx_[3] + HALO_PTS * 2;

    transpose_ = new Impl::Transpose<float64, array_layout>(nx_*ny_, nvx_*nvy_);
    int nx_min = dom->local_nxmin_[0], ny_min = dom->local_nxmin_[1], nvx_min = dom->local_nxmin_[2], nvy_min = dom->local_nxmin_[3];
    int nx_max = dom->local_nxmax_[0], ny_max = dom->local_nxmax_[1], nvx_max = dom->local_nxmax_[2], nvy_max = dom->local_nxmax_[3];
    fn_trans_ = RealOffsetView4D("fn_trans",
                                 {nvx_min-HALO_PTS, nvx_max+HALO_PTS},
                                 {nvy_min-HALO_PTS, nvy_max+HALO_PTS},
                                 {nx_min-HALO_PTS, nx_max+HALO_PTS},
                                 {ny_min-HALO_PTS, ny_max+HALO_PTS}
                                );
  }

  ~Spline() {
    if(transpose_ != nullptr) delete transpose_;
  };

  void computeCoeff_xy(Config *conf, RealOffsetView4D fn, 
                       const std::vector<int> &tiles={TILE_SIZE0, TILE_SIZE1});
  void computeCoeff_vxvy(Config *conf, RealOffsetView4D fn, 
                         const std::vector<int> &tiles={TILE_SIZE0, TILE_SIZE1});

public:
  // Kokkos functions and functors
  struct spline_coef_2d {
    Config *conf_;
    RealOffsetView4D fn_;    // transposed to (ivx, ivy, ix, iy) for xy and (ix, iy, ivx, ivy) for vxvy
    float64 sqrt3_;
    float64 alpha_, beta_;
    int check_ = 1;
    int n0_min_;
    int n1_min_;
    int n2_min_;
    int n3_min_;
  
    spline_coef_2d(Config *conf, RealOffsetView4D fn)
      : conf_(conf), fn_(fn) { 
      sqrt3_ = sqrt(3);
      alpha_ = sqrt(3) - 2;
      beta_ = sqrt3_ * (1 - alpha_ * alpha_);
      n0_min_ = fn_.begin(0);
      n1_min_ = fn_.begin(1);
      n2_min_ = fn_.begin(2);
      n3_min_ = fn_.begin(3);
    }
  
    ~spline_coef_2d() { }

    // Internal functions
    /* 
     * @brief
     * @param[in]  tmp1d(-HALO_PTS:n0+HALO_PTS+1)
     * @param[out] fn1d(-HALO_PTS:n0+HALO_PTS+1)
     */
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION
    void getSplineCoeff1D(ViewType &fn1d) const {
      const int istart = fn1d.begin(0) + HALO_PTS - 2;
      const int iend   = fn1d.end(0)   - HALO_PTS + 1;
      float64 tmp1d[LONG_BUFFER_WIDTH];
    
      // fn[istart-1] stores the precomputed left sum
      tmp1d[0] = fn1d(istart-1) + fn1d(istart);
      for(int nn = 1; istart + nn <= iend; nn++) {
        tmp1d[nn] = fn1d(istart + nn) + alpha_ * tmp1d[nn - 1];
      }
       
      // fn[iend+1] stores the precomputed right sum
      float64 fnend = fn1d(iend + 1) + fn1d(iend);
      float64 alpha_k = alpha_;
      for(int nn = 1; istart <= iend - nn; nn++) {
        fnend += fn1d(iend - nn) * alpha_k; //STDALGO
        alpha_k *= alpha_;
      }
      
      fn1d(iend) = fnend * sqrt3_;
      for(int nn = iend - 1; nn >= istart; nn--) {
        fn1d(nn) = beta_ * tmp1d[nn-istart] + alpha_ * fn1d(nn + 1);
      }
    }

    /* 
     * @brief
     * @param[in]  tmp2d(-HALO_PTS:n0+HALO_PTS+1, -HALO_PTS:n1+HALO_PTS+1)
     * @param[out] fn2d(-HALO_PTS:n0+HALO_PTS+1, -HALO_PTS:n1+HALO_PTS+1)
     * 
     */
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION
    void getSplineCoeff2D(ViewType &fn2d, int check) const {
      const int ixstart = fn2d.begin(0) + HALO_PTS - 2;
      const int ixend   = fn2d.end(0)   - HALO_PTS + 1;
      const int iystart = fn2d.begin(1) + HALO_PTS - 2;
      const int iyend   = fn2d.end(1)   - HALO_PTS + 1;
    
      // Precomputed Right and left coefficients on each column and row
      // are already start in (*, xstart-1), (*, xend+1) locations
      // All these calculations are done in halo_fill_boundary_cond:communication.cpp
      
      // Compute spline coefficients using precomputed parts
      for(int iy = iystart - 1; iy <= iyend + 1; iy++) {
        // row updated
        auto row = Kokkos::Experimental::subview(fn2d, Kokkos::ALL, iy);
        getSplineCoeff1D(row);
      }
    
      for(int ix = ixstart; ix <= ixend; ix++) {
        // col updated
        auto col = Kokkos::Experimental::subview(fn2d, ix, Kokkos::ALL);
        getSplineCoeff1D(col);
      }
    }
  
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i0, const int i1) const {
      #if defined ( LAYOUT_LEFT )
        // For LayoutLeft specialization for CPU
        auto sub_fn = Kokkos::Experimental::subview(fn_, Kokkos::ALL, Kokkos::ALL, i0+n2_min_, i1+n3_min_);
      #else
        auto sub_fn = Kokkos::Experimental::subview(fn_, i0+n0_min_, i1+n1_min_, Kokkos::ALL, Kokkos::ALL);
      #endif
  
      // 2D Spline interpolation
      getSplineCoeff2D(sub_fn, check_);
    }
  };

}; 

#endif

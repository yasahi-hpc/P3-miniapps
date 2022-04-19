#ifndef __SPLINE_HPP__
#define __SPLINE_HPP__

#include <numeric>
#include <execution>
#include <algorithm>
#include "Config.hpp"
#include "Types.hpp"
#include "Transpose.hpp"

struct Spline {
  Impl::Transpose<float64, default_layout> *transpose_;

private:
  RealView4D fn_trans_, fn_tmp_, fn_trans_tmp_;
  size_t nx_, ny_, nvx_, nvy_;
  int nx_min_, ny_min_, nvx_min_, nvy_min_;

public:
  Spline(Config *conf) {
    Domain *dom = &(conf->dom_);

    nx_min_  = dom->local_nxmin_[0] - HALO_PTS; 
    ny_min_  = dom->local_nxmin_[1] - HALO_PTS;
    nvx_min_ = dom->local_nxmin_[2] - HALO_PTS;
    nvy_min_ = dom->local_nxmin_[3] - HALO_PTS;
    const int nx_max  = dom->local_nxmax_[0] + HALO_PTS + 1;
    const int ny_max  = dom->local_nxmax_[1] + HALO_PTS + 1;
    const int nvx_max = dom->local_nxmax_[2] + HALO_PTS + 1;
    const int nvy_max = dom->local_nxmax_[3] + HALO_PTS + 1;

    nx_  = static_cast<size_t>(nx_max  - nx_min_);
    ny_  = static_cast<size_t>(ny_max  - ny_min_);
    nvx_ = static_cast<size_t>(nvx_max - nvx_min_);
    nvy_ = static_cast<size_t>(nvy_max - nvy_min_);

    // Something is wrong with Transpose kernel
    // Device Synchronization mandatory
    fn_trans_     = RealView4D("fn_trans", shape_nd<4>{nvx_,nvy_,nx_,ny_}, shape_nd<4>{nvx_min_,nvy_min_,nx_min_,ny_min_});
    fn_trans_tmp_ = RealView4D("fn_trans_tmp", shape_nd<4>{nvx_,nvy_,nx_,ny_}, shape_nd<4>{nvx_min_,nvy_min_,nx_min_,ny_min_});
    fn_tmp_       = RealView4D("fn_tmp", shape_nd<4>{nx_,ny_,nvx_,nvy_}, shape_nd<4>{nx_min_,ny_min_,nvx_min_,nvy_min_});
    fn_trans_.fill(0);
    fn_trans_tmp_.fill(0);
    fn_tmp_.fill(0);
    fn_trans_.updateDevice();
    fn_trans_tmp_.updateDevice();
    fn_tmp_.updateDevice();

    transpose_ = new Impl::Transpose<float64, default_layout>(nx_*ny_, nvx_*nvy_);
  };

  ~Spline() {
    if(transpose_ != nullptr) delete transpose_;
  }
  // Internal functions
  void computeCoeff_xy(RealView4D &fn) {
    using layout_type = RealView4D::layout_type;

    if(std::is_same_v<layout_type, stdex::layout_left>) {
      #if defined( _NVHPC_STDPAR_GPU )
        transpose_->forward(fn.data(), fn_trans_.data());
        computeCoeffCore_parallel_xy(fn_trans_, fn_trans_tmp_);
        transpose_->backward(fn_trans_.data(), fn.data());
      #else
        computeCoeffCore_parallel_vxvy(fn, fn_tmp_);
      #endif
    } else {
      #if defined( _NVHPC_STDPAR_GPU )
        computeCoeffCore_parallel_vxvy(fn, fn_tmp_);
      #else
        transpose_->forward(fn.data(), fn_trans_.data());
        computeCoeffCore_parallel_xy(fn_trans_, fn_trans_tmp_);
        transpose_->backward(fn_trans_.data(), fn.data());
      #endif
    }
  }

  void computeCoeff_vxvy(RealView4D &fn) {
    using layout_type = RealView4D::layout_type;
    if(std::is_same_v<layout_type, stdex::layout_left>) {
      #if defined( _NVHPC_STDPAR_GPU )
        computeCoeffCore_parallel_xy(fn, fn_tmp_);
      #else
        transpose_->forward(fn.data(), fn_trans_.data());
        computeCoeffCore_parallel_vxvy(fn_trans_, fn_trans_tmp_);
        transpose_->backward(fn_trans_.data(), fn.data());
      #endif
    } else {
      #if defined( _NVHPC_STDPAR_GPU )
        transpose_->forward(fn.data(), fn_trans_.data());
        computeCoeffCore_parallel_vxvy(fn_trans_, fn_tmp_);
        transpose_->backward(fn_trans_.data(), fn.data());
      #else
        computeCoeffCore_parallel_xy(fn, fn_tmp_);
      #endif
    }
  }

private:
  inline void computeCoeffCore_parallel_xy(RealView4D &fn, RealView4D &fn_tmp) {
    const float64 sqrt3 = sqrt(3);
    const int n0_min = fn.begin(0), n0_max = fn.end(0);
    const int n1_min = fn.begin(1), n1_max = fn.end(1);
    const int n2_min = fn.begin(2), n2_max = fn.end(2);
    const int n3_min = fn.begin(3), n3_max = fn.end(3);

    const int i2start = n2_min + HALO_PTS - 2;
    const int i2end   = n2_max - HALO_PTS + 1;
    const int i3start = n3_min + HALO_PTS - 2;
    const int i3end   = n3_max - HALO_PTS + 1;

    auto _fn = fn.mdspan();
    auto _fn_tmp = fn_tmp.mdspan();

    auto spline_2d = [=](const int i0, const int i1) {
      const float64 alpha = sqrt3 - 2;
      const float64 beta  = sqrt3 * (1 - alpha * alpha);

      #if defined( LONG_ENOUGH_BUFFER )
        float64 tmp1d[LONG_WIDTH];
      #endif
      // row update
      for(int i3 = i3start-1; i3 <= i3end + 1; i3++) {

        // fn[istart-1] stores the precomputed left sum
        #if defined( LONG_ENOUGH_BUFFER )
          tmp1d[0] = _fn(i0, i1, i2start-1, i3) + _fn(i0, i1, i2start, i3);
          for(int nn = 1; i2start + nn <= i2end; nn++) {
            tmp1d[nn]= _fn(i0, i1, i2start+nn, i3) + alpha * tmp1d[nn - 1];
          }
        #else
          _fn_tmp(i0, i1, i2start, i3) = _fn(i0, i1, i2start-1, i3) + _fn(i0, i1, i2start, i3);
          for(int nn = 1; i2start + nn <= i2end; nn++) {
            _fn_tmp(i0, i1, i2start + nn, i3) = _fn(i0, i1, i2start + nn, i3) + alpha * _fn_tmp(i0, i1, i2start + nn - 1, i3);
          }
        #endif

        // fn[iend+1] stores the precomputed right sum
        float64 fnend = _fn(i0, i1, i2end+1, i3) + _fn(i0, i1, i2end, i3);
        float64 alpha_k = alpha;
        for(int nn = 1; i2start <= i2end - nn; nn++) {
          fnend += _fn(i0, i1, i2end-nn, i3) * alpha_k; //STDALGO
          alpha_k *= alpha;
        }

        _fn(i0, i1, i2end, i3) = fnend * sqrt3;
        #if defined( LONG_ENOUGH_BUFFER )
          for(int nn = i2end - 1; nn >= i2start; nn--) {
            _fn(i0, i1, nn, i3) = beta * tmp1d[nn-i2start] + alpha * _fn(i0, i1, nn + 1, i3);
          }
        #else
          for(int nn = i2end - 1; nn >= i2start; nn--) {
            _fn(i0, i1, nn, i3) = beta * _fn_tmp(i0, i1, nn, i3) + alpha * _fn(i0, i1, nn + 1, i3);
          }
        #endif
      }

      // col update
      for(int i2 = i2start; i2 <= i2end; i2++) {
        // fn[istart-1] stores the precomputed left sum
        #if defined( LONG_ENOUGH_BUFFER )
          tmp1d[0] = _fn(i0, i1, i2, i3start-1) + _fn(i0, i1, i2, i3start);
          for(int nn = 1; i3start + nn <= i3end; nn++) {
            tmp1d[nn] = _fn(i0, i1, i2, i3start + nn) + alpha * tmp1d[nn - 1];
          }
        #else
          _fn_tmp(i0, i1, i2, i3start) = _fn(i0, i1, i2, i3start-1) + _fn(i0, i1, i2, i3start);
          for(int nn = 1; i3start + nn <= i3end; nn++) {
            _fn_tmp(i0, i1, i2, i3start + nn) = _fn(i0, i1, i2, i3start + nn) + alpha * _fn_tmp(i0, i1, i2, i3start + nn - 1);
          }
        #endif

        // fn[iend+1] stores the precomputed right sum
        float64 fnend = _fn(i0, i1, i2, i3end + 1) + _fn(i0, i1, i2, i3end);
        float64 alpha_k = alpha;
        for(int nn = 1; i3start <= i3end - nn; nn++) {
          fnend += _fn(i0, i1, i2, i3end - nn) * alpha_k; //STDALGO
          alpha_k *= alpha;
        }

        _fn(i0, i1, i2, i3end) = fnend * sqrt3;
        #if defined( LONG_ENOUGH_BUFFER )
          for(int nn = i3end - 1; nn >= i3start; nn--) {
            _fn(i0, i1, i2, nn) = beta * tmp1d[nn-i3start] + alpha * _fn(i0, i1, i2, nn + 1);
          }
        #else
          for(int nn = i3end - 1; nn >= i3start; nn--) {
            _fn(i0, i1, i2, nn) = beta * _fn_tmp(i0, i1, i2, nn) + alpha * _fn(i0, i1, i2, nn + 1);
          }
        #endif
      }
    };

    Iterate_policy<2> policy2d({n0_min, n1_min}, {n0_max, n1_max});
    Impl::for_each(policy2d, spline_2d);
  }

  inline void computeCoeffCore_parallel_vxvy(RealView4D &fn, RealView4D &fn_tmp) {
    const float64 sqrt3 = sqrt(3);
    const int n0_min = fn.begin(0), n0_max = fn.end(0);
    const int n1_min = fn.begin(1), n1_max = fn.end(1);
    const int n2_min = fn.begin(2), n2_max = fn.end(2);
    const int n3_min = fn.begin(3), n3_max = fn.end(3);

    const int i0start = n0_min + HALO_PTS - 2;
    const int i0end   = n0_max - HALO_PTS + 1;
    const int i1start = n1_min + HALO_PTS - 2;
    const int i1end   = n1_max - HALO_PTS + 1;

    auto _fn = fn.mdspan();
    auto _fn_tmp = fn_tmp.mdspan();

    auto spline_2d = [=](const int i2, const int i3) {
      const float64 alpha = sqrt3 - 2;
      const float64 beta  = sqrt3 * (1 - alpha * alpha);
      #if defined( LONG_ENOUGH_BUFFER )
        float64 tmp1d[LONG_WIDTH];
      #endif
      // row update
      for(int i1 = i1start-1; i1 <= i1end + 1; i1++) {
        // fn[istart-1] stores the precomputed left sum
        #if defined( LONG_ENOUGH_BUFFER )
          tmp1d[0] = _fn(i0start-1, i1, i2, i3) + _fn(i0start, i1, i2, i3);
          for(int nn = 1; i0start + nn <= i0end; nn++) {
            tmp1d[nn] = _fn(i0start + nn, i1, i2, i3) + alpha * tmp1d[nn - 1];
          }
        #else
          _fn_tmp(i0start, i1, i2, i3) = _fn(i0start-1, i1, i2, i3) + _fn(i0start, i1, i2, i3);
          for(int nn = 1; i0start + nn <= i0end; nn++) {
            _fn_tmp(i0start+nn, i1, i2, i3) = _fn(i0start + nn, i1, i2, i3) + alpha * _fn_tmp(i0start + nn - 1, i1, i2, i3);
          }
        #endif
         
        // fn[iend+1] stores the precomputed right sum
        float64 fnend = _fn(i0end+1, i1, i2, i3) + _fn(i0end, i1, i2, i3);
        float64 alpha_k = alpha;
        for(int nn = 1; i0start <= i0end - nn; nn++) {
          fnend += _fn(i0end - nn, i1, i2, i3) * alpha_k; //STDALGO
          alpha_k *= alpha;
        }
        
        _fn(i0end, i1, i2, i3) = fnend * sqrt3;
        #if defined( LONG_ENOUGH_BUFFER )
          for(int nn = i0end - 1; nn >= i0start; nn--) {
            _fn(nn, i1, i2, i3) = beta * tmp1d[nn-i0start] + alpha * _fn(nn + 1, i1, i2, i3);
          }
        #else
          for(int nn = i0end - 1; nn >= i0start; nn--) {
            _fn(nn, i1, i2, i3) = beta * _fn_tmp(nn, i1, i2, i3) + alpha * _fn(nn + 1, i1, i2, i3);
          }
        #endif
      }

      // col update
      for(int i0 = i0start; i0 <= i0end; i0++) {
        // fn[istart-1] stores the precomputed left sum
        #if defined( LONG_ENOUGH_BUFFER )
          tmp1d[0] = _fn(i0, i1start-1, i2, i3) + _fn(i0, i1start, i2, i3);
          for(int nn = 1; i1start + nn <= i1end; nn++) {
            tmp1d[nn] = _fn(i0, i1start + nn, i2, i3) + alpha * tmp1d[nn - 1];
          }
        #else
          _fn_tmp(i0, i1start, i2, i3) = _fn(i0, i1start-1, i2, i3) + _fn(i0, i1start, i2, i3);
          for(int nn = 1; i1start + nn <= i1end; nn++) {
            _fn_tmp(i0, i1start + nn, i2, i3) = _fn(i0, i1start + nn, i2, i3) + alpha * _fn_tmp(i0, i1start + nn - 1, i2, i3);
          }
        #endif
         
        // fn[iend+1] stores the precomputed right sum
        float64 fnend = _fn(i0, i1end+1, i2, i3) + _fn(i0, i1end, i2, i3);
        float64 alpha_k = alpha;
        for(int nn = 1; i1start <= i1end - nn; nn++) {
          fnend += _fn(i0, i1end - nn, i2, i3) * alpha_k; //STDALGO
          alpha_k *= alpha;
        }
         
        _fn(i0, i1end, i2, i3) = fnend * sqrt3;
        #if defined( LONG_ENOUGH_BUFFER )
          for(int nn = i1end - 1; nn >= i1start; nn--) {
            _fn(i0, nn, i2, i3) = beta * tmp1d[nn-i1start] + alpha * _fn(i0, nn + 1, i2, i3);
          }
        #else
          for(int nn = i1end - 1; nn >= i1start; nn--) {
            _fn(i0, nn, i2, i3) = beta * _fn_tmp(i0, nn, i2, i3) + alpha * _fn(i0, nn + 1, i2, i3);
          }
        #endif
      }
    };

    Iterate_policy<2> policy2d({n2_min, n3_min}, {n2_max, n3_max});
    Impl::for_each(policy2d, spline_2d);
  }
};

#endif

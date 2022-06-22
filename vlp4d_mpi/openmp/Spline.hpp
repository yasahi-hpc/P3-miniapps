#ifndef __SPLINE_HPP__
#define __SPLINE_HPP__

#include "Config.hpp"
#include "Types.hpp"
#include "Transpose.hpp"

namespace Spline {
  // prototypes
  void computeCoeff_xy(Config *conf, Impl::Transpose<float64, default_layout> *transpose, RealView4D &fn);
  void computeCoeff_vxvy(Config *conf, Impl::Transpose<float64, default_layout> *transpose, RealView4D &fn);

  // Internal functions

  // Common interface for OpenMP4.5/OpenMP, fn_tmp is used as a buffer
  // Layout Left
  template <class LayoutPolicy,
            typename std::enable_if_t< std::is_same_v<LayoutPolicy, stdex::layout_left>, std::nullptr_t> = nullptr
  >
  void computeCoeff(RealView4D &fn, RealView4D &fn_tmp) {
    const float64 sqrt3 = sqrt(3);
    const int n0_min = fn.begin(0), n0_max = fn.end(0);
    const int n1_min = fn.begin(1), n1_max = fn.end(1);
    const int n2_min = fn.begin(2), n2_max = fn.end(2);
    const int n3_min = fn.begin(3), n3_max = fn.end(3);

    #if defined( ENABLE_OPENMP_OFFLOAD )
      const int i2start = n2_min + HALO_PTS - 2;
      const int i2end   = n2_max - HALO_PTS + 1;
      const int i3start = n3_min + HALO_PTS - 2;
      const int i3end   = n3_max - HALO_PTS + 1;

      #pragma omp target teams distribute parallel for simd collapse(2)
      for(int i1=n1_min; i1 < n1_max; i1++) {
        for(int i0=n0_min; i0 < n0_max; i0++) {
          const float64 alpha = sqrt3 - 2;
          const float64 beta  = sqrt3 * (1 - alpha * alpha);
          #if defined( LONG_ENOUGH_BUFFER )
            float64 tmp1d[LONG_BUFFER_WIDTH];
          #endif
          // row update
          for(int i3 = i3start-1; i3 <= i3end + 1; i3++) {

            // fn[istart-1] stores the precomputed left sum
            #if defined( LONG_ENOUGH_BUFFER )
              tmp1d[0] = (fn(i0, i1, i2start-1, i3) + fn(i0, i1, i2start, i3));
              for(int nn = 1; i2start + nn <= i2end; nn++) {
                tmp1d[nn]= fn(i0, i1, i2start + nn, i3) + alpha * tmp1d[nn - 1];
              }
            #else
              fn_tmp(i0, i1, i2start, i3) = (fn(i0, i1, i2start-1, i3) + fn(i0, i1, i2start, i3));
              for(int nn = 1; i2start + nn <= i2end; nn++) {
                fn_tmp(i0, i1, i2start + nn, i3) = fn(i0, i1, i2start + nn, i3) + alpha * fn_tmp(i0, i1, i2start + nn - 1, i3);
              }
            #endif

            // fn[iend+1] stores the precomputed right sum
            float64 fnend = (fn(i0, i1, i2end + 1, i3) + fn(i0, i1, i2end, i3)); 
            float64 alpha_k = alpha;
            for(int nn = 1; i2start <= i2end - nn; nn++) {
              fnend += fn(i0, i1, i2end - nn, i3) * alpha_k; //STDALGO
              alpha_k *= alpha;
            }

            #if defined( LONG_ENOUGH_BUFFER )
              fn(i0, i1, i2end, i3) = fnend * sqrt3;
              for(int nn = i2end - 1; nn >= i2start; nn--) {
                fn(i0, i1, nn, i3) = beta * tmp1d[nn-i2start] + alpha * fn(i0, i1, nn + 1, i3);
              }
            #else
              fn(i0, i1, i2end, i3) = fnend * sqrt3;
              for(int nn = i2end - 1; nn >= i2start; nn--) {
                fn(i0, i1, nn, i3) = beta * fn_tmp(i0, i1, nn, i3) + alpha * fn(i0, i1, nn + 1, i3);
              }
            #endif
          }

          // col update
          for(int i2 = i2start; i2 <= i2end; i2++) {
            // fn[istart-1] stores the precomputed left sum
            #if defined( LONG_ENOUGH_BUFFER )
              tmp1d[0] = (fn(i0, i1, i2, i3start-1) + fn(i0, i1, i2, i3start));
              for(int nn = 1; i3start + nn <= i3end; nn++) {
                tmp1d[nn] = fn(i0, i1, i2, i3start + nn) + alpha * tmp1d[nn - 1];
              }
            #else
              fn_tmp(i0, i1, i2, i3start) = (fn(i0, i1, i2, i3start-1) + fn(i0, i1, i2, i3start));
              for(int nn = 1; i3start + nn <= i3end; nn++) {
                fn_tmp(i0, i1, i2, i3start + nn) = fn(i0, i1, i2, i3start + nn) + alpha * fn_tmp(i0, i1, i2, i3start + nn - 1);
              }
            #endif

            // fn[iend+1] stores the precomputed right sum
            float64 fnend = (fn(i0, i1, i2, i3end + 1) + fn(i0, i1, i2, i3end)); 
            float64 alpha_k = alpha;
            for(int nn = 1; i3start <= i3end - nn; nn++) {
              fnend += fn(i0, i1, i2, i3end - nn) * alpha_k; //STDALGO
              alpha_k *= alpha;
            }

            fn(i0, i1, i2, i3end) = fnend * sqrt3;
            #if defined( LONG_ENOUGH_BUFFER )
              for(int nn = i3end - 1; nn >= i3start; nn--) {
                fn(i0, i1, i2, nn) = beta * tmp1d[nn-i3start] + alpha * fn(i0, i1, i2, nn + 1);
              }
            #else
              for(int nn = i3end - 1; nn >= i3start; nn--) {
                fn(i0, i1, i2, nn) = beta * fn_tmp(i0, i1, i2, nn) + alpha * fn(i0, i1, i2, nn + 1);
              }
            #endif
          }
        }
      }
    #else
      const int i0start = n0_min + HALO_PTS - 2;
      const int i0end   = n0_max - HALO_PTS + 1;
      const int i1start = n1_min + HALO_PTS - 2;
      const int i1end   = n1_max - HALO_PTS + 1;
      #pragma omp parallel for collapse(2)
      for(int i3=n3_min; i3 < n3_max; i3++) {
        for(int i2=n2_min; i2 < n2_max; i2++) {
          const float64 alpha = sqrt3 - 2;
          const float64 beta  = sqrt3 * (1 - alpha * alpha);
          #if defined( LONG_ENOUGH_BUFFER )
            float64 tmp1d[LONG_BUFFER_WIDTH];
          #endif

          // row update
          for(int i1 = i1start-1; i1 <= i1end + 1; i1++) {
            // fn[istart-1] stores the precomputed left sum
            #if defined( LONG_ENOUGH_BUFFER )
              tmp1d[0] = (fn(i0start-1, i1, i2, i3) + fn(i0start, i1, i2, i3));
              for(int nn = 1; i0start + nn <= i0end; nn++) {
                tmp1d[nn] = fn(i0start + nn, i1, i2, i3) + alpha * tmp1d[nn-1];
              }
            #else
              fn_tmp(i0start, i1, i2, i3) = (fn(i0start-1, i1, i2, i3) + fn(i0start, i1, i2, i3));
              for(int nn = 1; i0start + nn <= i0end; nn++) {
                fn_tmp(i0start+nn, i1, i2, i3) = fn(i0start + nn, i1, i2, i3) + alpha * fn_tmp(i0start + nn - 1, i1, i2, i3);
              }
            #endif
             
            // fn[iend+1] stores the precomputed right sum
            float64 fnend = (fn(i0end+1, i1, i2, i3) + fn(i0end, i1, i2, i3));
            float64 alpha_k = alpha;
            for(int nn = 1; i0start <= i0end - nn; nn++) {
              fnend += fn(i0end - nn, i1, i2, i3) * alpha_k; //STDALGO
              alpha_k *= alpha;
            }
            
            fn(i0end, i1, i2, i3) = fnend * sqrt3;
            #if defined( LONG_ENOUGH_BUFFER )
              for(int nn = i0end - 1; nn >= i0start; nn--) {
                fn(nn, i1, i2, i3) = beta * tmp1d[nn-i0start] + alpha * fn(nn + 1, i1, i2, i3);
              }
            #else
              for(int nn = i0end - 1; nn >= i0start; nn--) {
                fn(nn, i1, i2, i3) = beta * fn_tmp(nn, i1, i2, i3) + alpha * fn(nn + 1, i1, i2, i3);
              }
            #endif
          }

          // col update
          for(int i0 = i0start; i0 <= i0end; i0++) {
            // fn[istart-1] stores the precomputed left sum
            #if defined( LONG_ENOUGH_BUFFER )
              tmp1d[0] = (fn(i0, i1start-1, i2, i3) + fn(i0, i1start, i2, i3));
              for(int nn = 1; i1start + nn <= i1end; nn++) {
                tmp1d[nn] = fn(i0, i1start + nn, i2, i3) + alpha * tmp1d[nn - 1];
              }
            #else
              fn_tmp(i0, i1start, i2, i3) = (fn(i0, i1start-1, i2, i3) + fn(i0, i1start, i2, i3));
              for(int nn = 1; i1start + nn <= i1end; nn++) {
                fn_tmp(i0, i1start + nn, i2, i3) = fn(i0, i1start + nn, i2, i3) + alpha * fn_tmp(i0, i1start + nn - 1, i2, i3);
              }
            #endif
             
            // fn[iend+1] stores the precomputed right sum
            float64 fnend = (fn(i0, i1end+1, i2, i3) + fn(i0, i1end, i2, i3));
            float64 alpha_k = alpha;
            for(int nn = 1; i1start <= i1end - nn; nn++) {
              fnend += fn(i0, i1end - nn, i2, i3) * alpha_k; //STDALGO
              alpha_k *= alpha;
            }
             
            fn(i0, i1end, i2, i3) = fnend * sqrt3;
            #if defined( LONG_ENOUGH_BUFFER )
              for(int nn = i1end - 1; nn >= i1start; nn--) {
                fn(i0, nn, i2, i3) = beta * tmp1d[nn-i1start] + alpha * fn(i0, nn + 1, i2, i3);
              }
            #else
              for(int nn = i1end - 1; nn >= i1start; nn--) {
                fn(i0, nn, i2, i3) = beta * fn_tmp(i0, nn, i2, i3) + alpha * fn(i0, nn + 1, i2, i3);
              }
            #endif
          }
        }
      }
    #endif
  }

  // Layout Right
  template <class LayoutPolicy,
            typename std::enable_if_t< std::is_same_v<LayoutPolicy, stdex::layout_right>, std::nullptr_t> = nullptr
  >
  void computeCoeff(RealView4D &fn, RealView4D &fn_tmp) {
    const float64 sqrt3 = sqrt(3);
    const int n0_min = fn.begin(0), n0_max = fn.end(0);
    const int n1_min = fn.begin(1), n1_max = fn.end(1);
    const int n2_min = fn.begin(2), n2_max = fn.end(2);
    const int n3_min = fn.begin(3), n3_max = fn.end(3);
    #if defined( ENABLE_OPENMP_OFFLOAD )
      const int i0start = n0_min + HALO_PTS - 2;
      const int i0end   = n0_max - HALO_PTS + 1;
      const int i1start = n1_min + HALO_PTS - 2;
      const int i1end   = n1_max - HALO_PTS + 1;
      #pragma omp target teams distribute parallel for simd collapse(2)
      for(int i2=n2_min; i2 < n2_max; i2++) {
        for(int i3=n3_min; i3 < n3_max; i3++) {
          const float64 alpha = sqrt3 - 2;
          const float64 beta  = sqrt3 * (1 - alpha * alpha);
          #if defined( LONG_ENOUGH_BUFFER )
            float64 tmp1d[LONG_BUFFER_WIDTH];
          #endif
          // row update
          for(int i1 = i1start-1; i1 <= i1end + 1; i1++) {
            // fn[istart-1] stores the precomputed left sum
            #if defined( LONG_ENOUGH_BUFFER )
              tmp1d[0] = (fn(i0start-1, i1, i2, i3) + fn(i0start, i1, i2, i3));
              for(int nn = 1; i0start + nn <= i0end; nn++) {
                tmp1d[nn] = fn(i0start + nn, i1, i2, i3) + alpha * tmp1d[nn - 1];
              }
            #else
              fn_tmp(i0start, i1, i2, i3) = (fn(i0start-1, i1, i2, i3) + fn(i0start, i1, i2, i3));
              for(int nn = 1; i0start + nn <= i0end; nn++) {
                fn_tmp(i0start+nn, i1, i2, i3) = fn(i0start + nn, i1, i2, i3) + alpha * fn_tmp(i0start + nn - 1, i1, i2, i3);
              }
            #endif
             
            // fn[iend+1] stores the precomputed right sum
            float64 fnend = (fn(i0end+1, i1, i2, i3) + fn(i0end, i1, i2, i3));
            float64 alpha_k = alpha;
            for(int nn = 1; i0start <= i0end - nn; nn++) {
              fnend += fn(i0end - nn, i1, i2, i3) * alpha_k; //STDALGO
              alpha_k *= alpha;
            }
            
            fn(i0end, i1, i2, i3) = fnend * sqrt3;
            #if defined( LONG_ENOUGH_BUFFER )
              for(int nn = i0end - 1; nn >= i0start; nn--) {
                fn(nn, i1, i2, i3) = beta * tmp1d[nn-i0start] + alpha * fn(nn + 1, i1, i2, i3);
              }
            #else
              for(int nn = i0end - 1; nn >= i0start; nn--) {
                fn(nn, i1, i2, i3) = beta * fn_tmp(nn, i1, i2, i3) + alpha * fn(nn + 1, i1, i2, i3);
              }
            #endif
          }

          // col update
          for(int i0 = i0start; i0 <= i0end; i0++) {
            // fn[istart-1] stores the precomputed left sum
            #if defined( LONG_ENOUGH_BUFFER )
              tmp1d[0] = (fn(i0, i1start-1, i2, i3) + fn(i0, i1start, i2, i3));
              for(int nn = 1; i1start + nn <= i1end; nn++) {
                tmp1d[nn] = fn(i0, i1start + nn, i2, i3) + alpha * tmp1d[nn - 1];
              }
            #else
              fn_tmp(i0, i1start, i2, i3) = (fn(i0, i1start-1, i2, i3) + fn(i0, i1start, i2, i3));
              for(int nn = 1; i1start + nn <= i1end; nn++) {
                fn_tmp(i0, i1start + nn, i2, i3) = fn(i0, i1start + nn, i2, i3) + alpha * fn_tmp(i0, i1start + nn - 1, i2, i3);
              }
            #endif
             
            // fn[iend+1] stores the precomputed right sum
            float64 fnend = (fn(i0, i1end+1, i2, i3) + fn(i0, i1end, i2, i3));
            float64 alpha_k = alpha;
            for(int nn = 1; i1start <= i1end - nn; nn++) {
              fnend += fn(i0, i1end - nn, i2, i3) * alpha_k; //STDALGO
              alpha_k *= alpha;
            }
             
            fn(i0, i1end, i2, i3) = fnend * sqrt3;
            #if defined( LONG_ENOUGH_BUFFER )
              for(int nn = i1end - 1; nn >= i1start; nn--) {
                fn(i0, nn, i2, i3) = beta * tmp1d[nn-i1start] + alpha * fn(i0, nn + 1, i2, i3);
              }
            #else
              for(int nn = i1end - 1; nn >= i1start; nn--) {
                fn(i0, nn, i2, i3) = beta * fn_tmp(i0, nn, i2, i3) + alpha * fn(i0, nn + 1, i2, i3);
              }
            #endif
          }
        }
      }
    #else
      const int i2start = n2_min + HALO_PTS - 2;
      const int i2end   = n2_max - HALO_PTS + 1;
      const int i3start = n3_min + HALO_PTS - 2;
      const int i3end   = n3_max - HALO_PTS + 1;
      #pragma omp parallel for collapse(2)
      for(int i1=n1_min; i1 < n1_max; i1++) {
        for(int i0=n0_min; i0 < n0_max; i0++) {
          const float64 alpha = sqrt3 - 2;
          const float64 beta  = sqrt3 * (1 - alpha * alpha);
          #if defined( LONG_ENOUGH_BUFFER )
            float64 tmp1d[LONG_BUFFER_WIDTH];
          #endif
          // row update
          for(int i3 = i3start-1; i3 <= i3end + 1; i3++) {
            // fn[istart-1] stores the precomputed left sum
            #if defined( LONG_ENOUGH_BUFFER )
              tmp1d[0] = (fn(i0, i1, i2start-1, i3) + fn(i0, i1, i2start, i3));
              for(int nn = 1; i2start + nn <= i2end; nn++) {
                tmp1d[nn] = fn(i0, i1, i2start + nn, i3) + alpha * tmp1d[nn - 1];
              }
            #else
              fn_tmp(i0, i1, i2start, i3) = (fn(i0, i1, i2start-1, i3) + fn(i0, i1, i2start, i3));
              for(int nn = 1; i2start + nn <= i2end; nn++) {
                fn_tmp(i0, i1, i2start + nn, i3) = fn(i0, i1, i2start + nn, i3) + alpha * fn_tmp(i0, i1, i2start + nn - 1, i3);
              }
            #endif

            // fn[iend+1] stores the precomputed right sum
            float64 fnend = (fn(i0, i1, i2end + 1, i3) + fn(i0, i1, i2end, i3)); 
            float64 alpha_k = alpha;
            for(int nn = 1; i2start <= i2end - nn; nn++) {
              fnend += fn(i0, i1, i2end - nn, i3) * alpha_k; //STDALGO
              alpha_k *= alpha;
            }

            fn(i0, i1, i2end, i3) = fnend * sqrt3;
            #if defined( LONG_ENOUGH_BUFFER )
              for(int nn = i2end - 1; nn >= i2start; nn--) {
                fn(i0, i1, nn, i3) = beta * tmp1d[nn-i2start] + alpha * fn(i0, i1, nn + 1, i3);
              }
            #else
              for(int nn = i2end - 1; nn >= i2start; nn--) {
                fn(i0, i1, nn, i3) = beta * fn_tmp(i0, i1, nn, i3) + alpha * fn(i0, i1, nn + 1, i3);
              }
            #endif
          }

          // col update
          for(int i2 = i2start; i2 <= i2end; i2++) {
            // fn[istart-1] stores the precomputed left sum
            #if defined( LONG_ENOUGH_BUFFER )
              tmp1d[0] = (fn(i0, i1, i2, i3start-1) + fn(i0, i1, i2, i3start));
              for(int nn = 1; i3start + nn <= i3end; nn++) {
                tmp1d[nn] = fn(i0, i1, i2, i3start + nn) + alpha * tmp1d[nn - 1];
              }
            #else
              fn_tmp(i0, i1, i2, i3start) = (fn(i0, i1, i2, i3start-1) + fn(i0, i1, i2, i3start));
              for(int nn = 1; i3start + nn <= i3end; nn++) {
                fn_tmp(i0, i1, i2, i3start + nn) = fn(i0, i1, i2, i3start + nn) + alpha * fn_tmp(i0, i1, i2, i3start + nn - 1);
              }
            #endif

            // fn[iend+1] stores the precomputed right sum
            float64 fnend = (fn(i0, i1, i2, i3end + 1) + fn(i0, i1, i2, i3end)); 
            float64 alpha_k = alpha;
            for(int nn = 1; i3start <= i3end - nn; nn++) {
              fnend += fn(i0, i1, i2, i3end - nn) * alpha_k; //STDALGO
              alpha_k *= alpha;
            }

            fn(i0, i1, i2, i3end) = fnend * sqrt3;
            #if defined( LONG_ENOUGH_BUFFER )
              for(int nn = i3end - 1; nn >= i3start; nn--) {
                fn(i0, i1, i2, nn) = beta * tmp1d[nn-i3start] + alpha * fn(i0, i1, i2, nn + 1);
              }
            #else
              for(int nn = i3end - 1; nn >= i3start; nn--) {
                fn(i0, i1, i2, nn) = beta * fn_tmp(i0, i1, i2, nn) + alpha * fn(i0, i1, i2, nn + 1);
              }
            #endif
          }
        }
      }
    #endif
  }

  // Declaration
  void computeCoeff_xy(Config *conf, Impl::Transpose<float64, default_layout> *transpose, RealView4D &fn) {
    using layout_type = RealView4D::layout_type;
    Domain *dom = &(conf->dom_);

    const int nx_min  = dom->local_nxmin_[0] - HALO_PTS; 
    const int ny_min  = dom->local_nxmin_[1] - HALO_PTS;
    const int nvx_min = dom->local_nxmin_[2] - HALO_PTS;
    const int nvy_min = dom->local_nxmin_[3] - HALO_PTS;
    const int nx_max  = dom->local_nxmax_[0] + HALO_PTS + 1;
    const int ny_max  = dom->local_nxmax_[1] + HALO_PTS + 1;
    const int nvx_max = dom->local_nxmax_[2] + HALO_PTS + 1;
    const int nvy_max = dom->local_nxmax_[3] + HALO_PTS + 1;

    const size_t nx  = static_cast<size_t>(nx_max - nx_min);
    const size_t ny  = static_cast<size_t>(ny_max - ny_min);
    const size_t nvx = static_cast<size_t>(nvx_max - nvx_min);
    const size_t nvy = static_cast<size_t>(nvy_max - nvy_min);

    if(std::is_same_v<layout_type, stdex::layout_left>) {
      #if defined( ENABLE_OPENMP_OFFLOAD )
        RealView4D fn_trans("fn_trans", shape_nd<DIMENSION>{nvx,nvy,nx,ny}, range_nd<DIMENSION>{nvx_min,nvy_min,nx_min,ny_min});
        RealView4D fn_tmp("fn_tmp", shape_nd<DIMENSION>{nvx,nvy,nx,ny}, range_nd<DIMENSION>{nvx_min,nvy_min,nx_min,ny_min});

        transpose->forward(fn.data(), fn_trans.data());
        computeCoeff<layout_type>(fn_trans, fn_tmp);
        transpose->backward(fn_trans.data(), fn.data());
      #else
        RealView4D fn_tmp("fn_tmp", shape_nd<DIMENSION>{nx,ny,nvx,nvy}, range_nd<DIMENSION>{nx_min,ny_min,nvx_min,nvy_min});
        computeCoeff<layout_type>(fn, fn_tmp);
      #endif
    } else {
      #if defined( ENABLE_OPENMP_OFFLOAD )
        RealView4D fn_tmp("fn_tmp", shape_nd<DIMENSION>{nx,ny,nvx,nvy}, range_nd<DIMENSION>{nx_min,ny_min,nvx_min,nvy_min});
        computeCoeff<layout_type>(fn, fn_tmp);
      #else
        RealView4D fn_trans("fn_trans", shape_nd<DIMENSION>{nvx,nvy,nx,ny}, range_nd<DIMENSION>{nvx_min,nvy_min,nx_min,ny_min});
        RealView4D fn_tmp("fn_tmp", shape_nd<DIMENSION>{nvx,nvy,nx,ny}, range_nd<DIMENSION>{nvx_min,nvy_min,nx_min,ny_min});
        transpose->forward(fn.data(), fn_trans.data());
        computeCoeff<layout_type>(fn_trans, fn_tmp);
        transpose->backward(fn_trans.data(), fn.data());
      #endif
    }
  }

  // Layout Left
  void computeCoeff_vxvy(Config *conf, Impl::Transpose<float64, default_layout> *transpose, RealView4D &fn) {
    using layout_type = RealView4D::layout_type;
    Domain *dom = &(conf->dom_);

    const int nx_min = dom->local_nxmin_[0] - HALO_PTS; 
    const int ny_min = dom->local_nxmin_[1] - HALO_PTS;
    const int nvx_min = dom->local_nxmin_[2] - HALO_PTS;
    const int nvy_min = dom->local_nxmin_[3] - HALO_PTS;
    const int nx_max = dom->local_nxmax_[0] + HALO_PTS + 1;
    const int ny_max = dom->local_nxmax_[1] + HALO_PTS + 1;
    const int nvx_max = dom->local_nxmax_[2] + HALO_PTS + 1;
    const int nvy_max = dom->local_nxmax_[3] + HALO_PTS + 1;
    const size_t nx  = static_cast<size_t>(nx_max - nx_min);
    const size_t ny  = static_cast<size_t>(ny_max - ny_min);
    const size_t nvx = static_cast<size_t>(nvx_max - nvx_min);
    const size_t nvy = static_cast<size_t>(nvy_max - nvy_min);

    if(std::is_same_v<layout_type, stdex::layout_left>) {
      #if defined( ENABLE_OPENMP_OFFLOAD )
        RealView4D fn_tmp("fn_tmp", shape_nd<DIMENSION>{nx,ny,nvx,nvy}, range_nd<DIMENSION>{nx_min,ny_min,nvx_min,nvy_min});
        computeCoeff<layout_type>(fn, fn_tmp);
      #else
        RealView4D fn_trans("fn_trans", shape_nd<DIMENSION>{nvx,nvy,nx,ny}, range_nd<DIMENSION>{nvx_min,nvy_min,nx_min,ny_min});
        RealView4D fn_tmp("fn_tmp", shape_nd<DIMENSION>{nvx,nvy,nx,ny}, range_nd<DIMENSION>{nvx_min,nvy_min,nx_min,ny_min});
        transpose->forward(fn.data(), fn_trans.data());
        computeCoeff<layout_type>(fn_trans, fn_tmp);
        transpose->backward(fn_trans.data(), fn.data());
      #endif
    } else {
      #if defined( ENABLE_OPENMP_OFFLOAD )
        RealView4D fn_trans("fn_trans", shape_nd<DIMENSION>{nvx,nvy,nx,ny}, range_nd<DIMENSION>{nvx_min,nvy_min,nx_min,ny_min});
        RealView4D fn_tmp("fn_tmp", shape_nd<DIMENSION>{nvx,nvy,nx,ny}, range_nd<DIMENSION>{nvx_min,nvy_min,nx_min,ny_min});
        transpose->forward(fn.data(), fn_trans.data());
        computeCoeff<layout_type>(fn_trans, fn_tmp);
        transpose->backward(fn_trans.data(), fn.data());
      #else
        RealView4D fn_tmp("fn_tmp", shape_nd<DIMENSION>{nx,ny,nvx,nvy}, range_nd<DIMENSION>{nx_min,ny_min,nvx_min,nvy_min});
        computeCoeff<layout_type>(fn, fn_tmp);
      #endif
    }
  }
};

#endif

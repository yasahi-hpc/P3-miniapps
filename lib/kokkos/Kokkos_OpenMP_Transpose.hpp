#ifndef __KOKKOS_OPENMP_TRANSPOSE_H__
#define __KOKKOS_OPENMP_TRANSPOSE_H__

#include <omp.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include "../Index.hpp"

/*
 * Simple wrapper for transpoes
 * For CPU, LayoutRight (C style is assumed)
 */

template <typename RealType> using Complex = Kokkos::complex<RealType>;

namespace Impl {
  template <typename ScalarType, class ArrayLayout = Kokkos::LayoutRight, int blocksize = 16, 
            typename std::enable_if<std::is_same<ScalarType, int             >::value ||
                                    std::is_same<ScalarType, float           >::value ||
                                    std::is_same<ScalarType, double          >::value ||
                                    std::is_same<ScalarType, Complex<float>  >::value ||
                                    std::is_same<ScalarType, Complex<double> >::value
                                   >::type * = nullptr>
  struct Transpose {
    private:
      int col_;
      int row_;
      const int blocksize_ = blocksize;

    public:
      using array_layout = ArrayLayout;

    public:

      Transpose() = delete;
      Transpose(int row, int col) : row_(row), col_(col) {
        if(std::is_same<array_layout, Kokkos::LayoutRight>::value) {
          row_ = col;
          col_ = row;
        }
      }

      ~Transpose() {}

    public:
      // Interfaces
      void forward(ScalarType *dptr_in, ScalarType *dptr_out) {
        transpose_(dptr_in, dptr_out, row_, col_);
      }

      void backward(ScalarType *dptr_in, ScalarType *dptr_out) {
        transpose_(dptr_in, dptr_out, col_, row_);
      }

    private:
      // Out-place transpose
      void transpose_(ScalarType *dptr_in, ScalarType *dptr_out, int row, int col) {
        #pragma omp parallel for schedule(static) collapse(2)
        for(int j = 0; j < col; j += blocksize_) {
          for(int i = 0; i < row; i += blocksize_) {
            for(int c = j; c < j + blocksize_ && c < col; c++) {
              for(int r = i; r < i + blocksize_ && r < row; r++) {
                int idx_src = Index::coord_2D2int(r, c, row, col);
                int idx_dst = Index::coord_2D2int(c, r, col, row);
                dptr_out[idx_dst] = dptr_in[idx_src];
              }
            }
          }
        }
      }
  };
}

#endif

#ifndef __OPENMP_TRANSPOSE_HPP__
#define __OPENMP_TRANSPOSE_HPP__

#include <omp.h>
#include <complex>
#include <experimental/mdspan>
#include "../Index.hpp"

template <typename RealType> using Complex = std::complex<RealType>;
namespace stdex = std::experimental;

namespace Impl {
  template <typename ScalarType, class LayoutPolicy = stdex::layout_right, int blocksize = 16,
            std::enable_if_t<std::is_same_v<ScalarType, int             > ||
                             std::is_same_v<ScalarType, float           > ||
                             std::is_same_v<ScalarType, double          > ||
                             std::is_same_v<ScalarType, Complex<float>  > ||
                             std::is_same_v<ScalarType, Complex<double> >
                            , std::nullptr_t> = nullptr>

  struct Transpose {
    private:
      int col_;
      int row_;
      const int blocksize_ = blocksize;

    public:
      using array_layout = LayoutPolicy;

    public:
      Transpose(int row, int col) : row_(row), col_(col) {
        if(std::is_same_v<array_layout, stdex::layout_right>) {
          row_ = col;
          col_ = row;
        }
      }
      ~Transpose(){}

    public:
      // Interfaces
      void forward(ScalarType *dptr_in, ScalarType *dptr_out) {
        exec(dptr_in ,dptr_out, row_, col_);
      }

      void backward(ScalarType *dptr_in, ScalarType *dptr_out) {
        exec(dptr_in ,dptr_out, col_, row_);
      }

    private:
      void exec(ScalarType *dptr_in, ScalarType *dptr_out, int row, int col) {
        #pragma omp parallel for collapse(2)
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
};

#endif

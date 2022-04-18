#ifndef __KOKKOS_CUDA_TRANSPOSE_HPP__
#define __KOKKOS_CUDA_TRANSPOSE_HPP__

#include <cublas_v2.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

template <typename RealType> using Complex = Kokkos::complex<RealType>;

namespace Impl {
  template <typename RealType, class ArrayLayout = Kokkos::LayoutLeft,
            typename std::enable_if<std::is_same<RealType, float           >::value ||
                                    std::is_same<RealType, double          >::value ||
                                    std::is_same<RealType, Complex<float>  >::value ||
                                    std::is_same<RealType, Complex<double> >::value
                                   >::type * = nullptr>
  struct Transpose {
    private:
      int col_;
      int row_;
      cublasHandle_t handle_;

    public:
      using array_layout = ArrayLayout;

    public:
      Transpose() = delete;
      Transpose(int row, int col) : row_(row), col_(col) {
        // default initialization assumes LayoutLeft
        if(std::is_same<array_layout, Kokkos::LayoutRight>::value) {
          row_ = col;
          col_ = row;
        }
        cublasCreate(&handle_);
      }

      ~Transpose() {
        cublasDestroy(handle_);
      }

      // Out-place transpose
      void forward(RealType *dptr_in, RealType *dptr_out) {
        cublasTranspose_(dptr_in, dptr_out, row_, col_);
      }

      void backward(RealType *dptr_in, RealType *dptr_out) {
        cublasTranspose_(dptr_in, dptr_out, col_, row_);
      }

    private:

      // Float32 specialization
      void cublasTranspose_(float *dptr_in, float *dptr_out, int row, int col) {
        const float alpha = 1.0;
        const float beta  = 0.0;
        cublasSgeam(handle_,     // handle
                    CUBLAS_OP_T, // transa
                    CUBLAS_OP_T, // transb
                    col,         // m
                    row,         // n
                    &alpha,      // alpha 
                    dptr_in,     // A
                    row,         // lda: leading dimension of two-dimensional array used to store A
                    &beta,       // beta
                    dptr_in,     // B
                    row,         // ldb: leading dimension of two-dimensional array used to store B
                    dptr_out,    // C
                    col);        // ldc; leading dimension of two-dimensional array used to store C
      }

      // Float64 specialization
      void cublasTranspose_(double *dptr_in, double *dptr_out, int row, int col) {
        const double alpha = 1.;
        const double beta  = 0.;
        cublasDgeam(handle_,     // handle
                    CUBLAS_OP_T, // transa
                    CUBLAS_OP_T, // transb
                    col,         // m
                    row,         // n
                    &alpha,      // alpha 
                    dptr_in,     // A
                    row,         // lda: leading dimension of two-dimensional array used to store A
                    &beta,       // beta
                    dptr_in,     // B
                    row,         // ldb: leading dimension of two-dimensional array used to store B
                    dptr_out,    // C
                    col);        // ldc; leading dimension of two-dimensional array used to store C
      }

      // complex64 specialization
      void cublasTranspose_(Complex<float> *dptr_in, Complex<float> *dptr_out, int row, int col) {
        const cuComplex alpha = make_cuComplex(1.0, 0.0);
        const cuComplex beta  = make_cuComplex(0.0, 0.0);
        cublasCgeam(handle_,     // handle
                    CUBLAS_OP_T, // transa
                    CUBLAS_OP_N, // transb
                    col,         // m
                    row,         // n
                    &alpha,      // alpha 
                    reinterpret_cast<cuComplex*>(dptr_in), // A
                    row,         // lda: leading dimension of two-dimensional array used to store A
                    &beta,       // beta
                    reinterpret_cast<cuComplex*>(dptr_in), // B
                    row,         // ldb: leading dimension of two-dimensional array used to store B
                    reinterpret_cast<cuComplex*>(dptr_out), // C
                    col);        // ldc; leading dimension of two-dimensional array used to store C
      }

      // complex128 specialization
      void cublasTranspose_(Complex<double> *dptr_in, Complex<double> *dptr_out, int row, int col) {
        const cuDoubleComplex alpha = make_cuDoubleComplex(1., 0.);
        const cuDoubleComplex beta  = make_cuDoubleComplex(0., 0.);
        cublasZgeam(handle_,     // handle
                    CUBLAS_OP_T, // transa
                    CUBLAS_OP_N, // transb
                    col,         // m
                    row,         // n
                    &alpha,      // alpha 
                    reinterpret_cast<cuDoubleComplex*>(dptr_in),  // A
                    row,         // lda: leading dimension of two-dimensional array used to store A
                    &beta,       // beta
                    reinterpret_cast<cuDoubleComplex*>(dptr_in),  // B
                    row,         // ldb: leading dimension of two-dimensional array used to store B
                    reinterpret_cast<cuDoubleComplex*>(dptr_out), // C
                    col);        // ldc; leading dimension of two-dimensional array used to store C
      }
  };
}

#endif

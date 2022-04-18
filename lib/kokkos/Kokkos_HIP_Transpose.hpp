#ifndef __KOKKOS_HIP_TRANSPOSE_HPP__
#define __KOKKOS_HIP_TRANSPOSE_HPP__

/*
 * Simple wrapper for rocblas
 * HIP interface
 * https://github.com/ROCmSoftwarePlatform/rocBLAS
 */

#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <rocblas.h>
#include <type_traits>
#include <layout_contiguous/layout_contiguous.hpp>
#include "../HIP_Helper.hpp"

template <typename RealType> using Complex = Kokkos::complex<RealType>;

namespace Impl {
  template <typename ScalarType, class ArrayLayout = Kokkos::LayoutLeft,
            std::enable_if_t<std::is_same_v<ScalarType, float           > ||
                             std::is_same_v<ScalarType, double          > ||
                             std::is_same_v<ScalarType, Complex<float>  > ||
                             std::is_same_v<ScalarType, Complex<double> > 
                             , std::nullptr_t> = nullptr>
  struct Transpose {
    private:
      int col_;
      int row_;
      rocblas_handle handle_;

    public:
      using array_layout = ArrayLayout;

    public:
      Transpose() = delete;

      Transpose(int row, int col) : row_(row), col_(col) {
        if(std::is_same_v<array_layout, Kokkos::LayoutRight>) {
          row_ = col;
          col_ = row;
        }
        SafeHIPCall( rocblas_create_handle(&handle_) );
        SafeHIPCall( rocblas_set_pointer_mode(handle_, rocblas_pointer_mode_host) );
      }

      ~Transpose() {
        SafeHIPCall( rocblas_destroy_handle(handle_) );
      }

      // Out-place transpose
      void forward(ScalarType *dptr_in, ScalarType *dptr_out) {
        rocblasTranspose_(dptr_in, dptr_out, row_, col_);
        SafeHIPCall( hipDeviceSynchronize() );
      }

      void backward(ScalarType *dptr_in, ScalarType *dptr_out) {
        rocblasTranspose_(dptr_in, dptr_out, col_, row_);
        SafeHIPCall( hipDeviceSynchronize() );
      }

      // Float32 specialization
      template <typename SType=ScalarType,
                std::enable_if_t<std::is_same_v<SType, float>, std::nullptr_t> = nullptr>
      void rocblasTranspose_(SType *dptr_in, SType *dptr_out, int row, int col) {
        constexpr float alpha = 1.0;
        constexpr float beta  = 0.0;
        SafeHIPCall( 
          rocblas_sgeam(handle_,                     // handle
                        rocblas_operation_transpose, // transa
                        rocblas_operation_transpose, // transb
                        col,                         // m
                        row,                         // n
                        &alpha,                      // alpha
                        dptr_in,                     // A
                        row,                         // lda: leading dimension of two-dimensional array used to store A
                        &beta,                       // beta
                        dptr_in,                     // B
                        row,                         // ldb: leading dimension of two-dimensional array used to store B
                        dptr_out,                    // C
                        col)                         // ldc: leading dimension of two-dimensional array used to store C
        );
      }

      // Float64 specialization
      template <typename SType=ScalarType, 
                std::enable_if_t<std::is_same_v<SType, double>, std::nullptr_t> = nullptr>
      void rocblasTranspose_(SType *dptr_in, SType *dptr_out, int row, int col) {
        constexpr double alpha = 1.0;
        constexpr double beta  = 0.0;
        SafeHIPCall( 
          rocblas_dgeam(handle_,                     // handle
                        rocblas_operation_transpose, // transa
                        rocblas_operation_transpose, // transb
                        col,                         // m
                        row,                         // n
                        &alpha,                      // alpha
                        dptr_in,                     // A
                        row,                         // lda: leading dimension of two-dimensional array used to store A
                        &beta,                       // beta
                        dptr_in,                     // B
                        row,                         // ldb: leading dimension of two-dimensional array used to store B
                        dptr_out,                    // C
                        col)                         // ldc: leading dimension of two-dimensional array used to store C
        );
      }

      // complex64 specialization
      template <typename SType=ScalarType,
                std::enable_if_t<std::is_same_v<SType, Complex<float> >, std::nullptr_t> = nullptr>
      void rocblasTranspose_(SType *dptr_in, SType *dptr_out, int row, int col) {
        const rocblas_float_complex alpha(1.0);
        const rocblas_float_complex beta(0.0);
        SafeHIPCall( 
          rocblas_cgeam(handle_,                     // handle
                        rocblas_operation_transpose, // transa
                        rocblas_operation_transpose, // transb
                        col,                         // m
                        row,                         // n
                        &alpha,                      // alpha
                        reinterpret_cast<rocblas_float_complex*>(dptr_in), // A
                        row,                         // lda: leading dimension of two-dimensional array used to store A
                        &beta,                       // beta
                        reinterpret_cast<rocblas_float_complex*>(dptr_in), // B
                        row,                         // ldb: leading dimension of two-dimensional array used to store B
                        reinterpret_cast<rocblas_float_complex*>(dptr_out), // C
                        col)                         // ldc: leading dimension of two-dimensional array used to store C
        );
      }

      // complex128 specialization
      template <typename SType=ScalarType, 
                std::enable_if_t<std::is_same_v<SType, Complex<double> >, std::nullptr_t> = nullptr>
      void rocblasTranspose_(SType *dptr_in, SType *dptr_out, int row, int col) {
        const rocblas_double_complex alpha(1.0);
        const rocblas_double_complex beta(0.0);
        SafeHIPCall( 
          rocblas_zgeam(handle_,                     // handle
                        rocblas_operation_transpose, // transa
                        rocblas_operation_transpose, // transb
                        col,                         // m
                        row,                         // n
                        &alpha,                      // alpha
                        reinterpret_cast<rocblas_double_complex*>(dptr_in), // A
                        row,                         // lda: leading dimension of two-dimensional array used to store A
                        &beta,                       // beta
                        reinterpret_cast<rocblas_double_complex*>(dptr_in), // B
                        row,                         // ldb: leading dimension of two-dimensional array used to store B
                        reinterpret_cast<rocblas_double_complex*>(dptr_out), // C
                        col)                         // ldc: leading dimension of two-dimensional array used to store C
        );
      }
  };
};

#endif

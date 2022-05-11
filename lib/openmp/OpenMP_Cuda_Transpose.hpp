#ifndef __OPENMP_CUDA_TRANSPOSE_HPP__
#define __OPENMP_CUDA_TRANSPOSE_HPP__

#include <cublas_v2.h>
#include <complex>
#include <omp.h>
#include <experimental/mdspan>

template <typename RealType> using Complex = std::complex<RealType>;
namespace stdex = std::experimental;

namespace Impl {
  template <typename RealType, class LayoutPolicy = stdex::layout_left,
            std::enable_if_t<std::is_same_v<RealType, float           > ||
                             std::is_same_v<RealType, double          > ||
                             std::is_same_v<RealType, Complex<float>  > ||
                             std::is_same_v<RealType, Complex<double> >
                             , std::nullptr_t> = nullptr
  >
  struct Transpose {
    private:
      int col_;
      int row_;
      cublasHandle_t handle_;

    public:
      using array_layout = LayoutPolicy;
     
    public:
      Transpose() = delete;
    
      Transpose(int row, int col) : row_(row), col_(col) {
        if(std::is_same_v<array_layout, stdex::layout_right>) {
          row_ = col;
          col_ = row;
        }
        cublasCreate(&handle_);
        //cublasSetStream(handle_, 0); // This does not work
      }

      ~Transpose() {
        cublasDestroy(handle_);
      }

      // Out-place transpose
      void forward(RealType *dptr_in, RealType *dptr_out) {
        #pragma omp target data use_device_ptr(dptr_in, dptr_out)
        cublasTranspose_(dptr_in, dptr_out, row_, col_);
        sync_();
      }

      void backward(RealType *dptr_in, RealType *dptr_out) {
        #pragma omp target data use_device_ptr(dptr_in, dptr_out)
        cublasTranspose_(dptr_in, dptr_out, col_, row_);
        sync_();
      }

    private:
      /* For some reason, cublas kernel runs with a non-default stream and
       * it runs concurrently with OpenMP offload kernel.
       * Accordingly, synchrnoization is needed, which is achieved by the deep copy of a dummy data.
       */
      void sync_() {
        int dummy[1];
        #pragma omp target map(to: dummy)
        {}
      }

      // float32 specialization
      template <typename RType=RealType,
                std::enable_if_t<std::is_same_v<RType, float>, std::nullptr_t> = nullptr>
      void cublasTranspose_(RType *dptr_in, RType *dptr_out, int row, int col) {
        constexpr float alpha = 1.0;
        constexpr float beta  = 0.0;
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

      // float64 specialization
      template <typename RType=RealType,
                std::enable_if_t<std::is_same_v<RType, double>, std::nullptr_t> = nullptr>
      void cublasTranspose_(RType *dptr_in, RType *dptr_out, int row, int col) {
        constexpr double alpha = 1.;
        constexpr double beta  = 0.;
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
      template <typename RType=RealType,
                std::enable_if_t<std::is_same_v<RType, Complex<float> >, std::nullptr_t> = nullptr>
      void cublasTranspose_(RType *dptr_in, RType *dptr_out, int row, int col) {
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
      template <typename RType=RealType,
                std::enable_if_t<std::is_same_v<RType, Complex<double> >, std::nullptr_t> = nullptr>
      void cublasTranspose_(RType *dptr_in, RType *dptr_out, int row, int col) {
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
};

#endif

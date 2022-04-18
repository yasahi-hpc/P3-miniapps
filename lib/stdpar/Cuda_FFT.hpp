#ifndef __CUDA_FFT_HPP__
#define __CUDA_FFT_HPP__

/*
 * Simple wrapper for FFT class (only 2D FFT implemented now)
 *
 * Input data is assumed to be LayuotLeft
 */

#include <complex>
#include <cufft.h>
#include <cassert>
#include <type_traits>
#include <openacc.h>
#include <layout_contiguous/layout_contiguous.hpp>
#include "../Cuda_Helper.hpp"

template <typename RealType> using Complex = std::complex<RealType>;

namespace Impl {

  template <typename RealType, class LayoutPolicy = layout_contiguous_at_left,
            typename std::enable_if<std::is_same<RealType, float>::value ||
                                    std::is_same<RealType, double>::value 
                                   >::type * = nullptr> 
  struct FFT {
    private:
      // FFT plan
      cufftHandle backward_plan_, forward_plan_;

      // Number of real points in the x1 (x) dimension
      int nx1_; 

      // Number of real points in the x2 (y) dimension
      int nx2_;

      // Number of batches
      int nb_batches_;

      // number of complex points+1 in the x1 (x) dimension
      int nx1h_;

      // number of complex points+1 in the x2 (y) dimension
      int nx2h_;

      // Normalization coefficient
      RealType normcoeff_;

    public:
      using array_layout = LayoutPolicy;

    public:
    FFT(int nx1, int nx2)
      : nx1_(nx1), nx2_(nx2), nb_batches_(1) {
      init();
    }

    FFT(int nx1, int nx2, int nb_batches) 
      : nx1_(nx1), nx2_(nx2), nb_batches_(nb_batches) {
      init();
    }

    virtual ~FFT() {
      cufftDestroy(forward_plan_);
      cufftDestroy(backward_plan_);
    }

    void rfft2(RealType *dptr_in, Complex<RealType> *dptr_out) {
      rfft2_(dptr_in, dptr_out);
    }

    void irfft2(Complex<RealType> *dptr_in, RealType *dptr_out) {
      irfft2_(dptr_in, dptr_out);
    }

    RealType normcoeff() const {return normcoeff_;}
    int nx1() {return nx1_;}
    int nx2() {return nx2_;}
    int nx1h() {return nx1h_;}
    int nx2h() {return nx2h_;}
    int nb_batches() {return nb_batches_;}

    private:
    void init() {
      static_assert(std::is_same<array_layout, layout_contiguous_at_left>::value, "The input Layout must be LayoutLeft");
      nx1h_ = nx1_/2 + 1;
      nx2h_ = nx2_/2 + 1;

      normcoeff_ = static_cast<RealType>(1.0) / static_cast<RealType>(nx1_ * nx2_);

      // Create forward/backward plans
      cufftCreate(&forward_plan_);
      cufftCreate(&backward_plan_);

      assert(nb_batches_ >= 1);

      cufftType forward_type, backward_type;
      if(std::is_same<RealType, float>::value) {
        forward_type  = CUFFT_R2C;
        backward_type = CUFFT_C2R;
      }

      if(std::is_same<RealType, double>::value) {
        forward_type  = CUFFT_D2Z;
        backward_type = CUFFT_Z2D;
      }

      if(nb_batches_ == 1) {
        SafeCudaCall(
          cufftPlan2d(&forward_plan_, nx2_, nx1_, forward_type)
        );

        SafeCudaCall(
          cufftPlan2d(&backward_plan_, nx2_, nx1_, backward_type)
        );
      } else {
        // Batched plan
        int rank = 2;
        int n[2];
        int inembed[2], onembed[2];
        int istride, ostride;
        int idist, odist;
        n[0] = nx2_; n[1] = nx1_;
        idist = nx2_*nx1_;
        odist = nx2_*(nx1h_);

        inembed[0] = nx2_; inembed[1] = nx1_;
        onembed[0] = nx2_; onembed[1] = nx1h_;
        istride = 1; ostride = 1;

        // Forward plan
        SafeCudaCall(
          cufftPlanMany(&forward_plan_,
                        rank,       // rank
                        n,          // dimensions = {pixels.x, pixels.y}
                        inembed,    // Input size with pitch
                        istride,    // Distance between two successive input elements
                        idist,      // Distance between batches for input array
                        onembed,    // Output size with pitch
                        ostride,    // Distance between two successive output elements
                        odist,      // Distance between batches for output array
                        forward_type,  // Cufft Type
                        nb_batches_) // The number of FFTs executed by this plan
        );

        // Backward plan
        SafeCudaCall(
          cufftPlanMany(&backward_plan_,
                        rank,       // rank
                        n,          // dimensions = {pixels.x, pixels.y}
                        onembed,    // Input size with pitch
                        ostride,    // Distance between two successive input elements
                        odist,      // Distance between batches for input array
                        inembed,    // Output size with pitch
                        istride,    // Distance between two successive output elements
                        idist,      // Distance between batches for output array
                        backward_type, // Cufft Type
                        nb_batches_) // The number of FFTs executed by this plan
        );
      }
    }

    private:
    inline void rfft2_(float *dptr_in, Complex<float> *dptr_out) {
      cufftExecR2C(forward_plan_, 
                   reinterpret_cast<float *>(dptr_in), 
                   reinterpret_cast<cuComplex *>(dptr_out));
    }

    inline void rfft2_(double *dptr_in, Complex<double> *dptr_out) {
      cufftExecD2Z(forward_plan_, 
                   reinterpret_cast<double *>(dptr_in), 
                   reinterpret_cast<cuDoubleComplex *>(dptr_out));
    }

    inline void irfft2_(Complex<float> *dptr_in, float *dptr_out) {
      cufftExecC2R(backward_plan_, 
                   reinterpret_cast<cuComplex *>(dptr_in), 
                   reinterpret_cast<float *>(dptr_out));
    }

    inline void irfft2_(Complex<double> *dptr_in, double *dptr_out) {
      cufftExecZ2D(backward_plan_, 
                   reinterpret_cast<cuDoubleComplex *>(dptr_in), 
                   reinterpret_cast<double *>(dptr_out));
    }
  };
};
#endif

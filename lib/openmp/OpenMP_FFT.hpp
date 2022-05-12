#ifndef __OPENMP_FFT_HPP__
#define __OPENMP_FFT_HPP__

/*
 * Simple wrapper for FFT class (only 2D FFT implemented now)
 * fftw interface for OpenMP
 * Input data can be LayoutLeft or LayoutRight (default)
 *
 */

#include <fftw3.h>
#include <omp.h>
#include <cassert>
#include <iostream>
#include <complex>
#include <experimental/mdspan>
#include "../Index.hpp"

template <typename RealType> using Complex = std::complex<RealType>;
namespace stdex = std::experimental;

namespace Impl {

  template <typename RealType, class LayoutPolicy = stdex::layout_right,
            typename std::enable_if<std::is_same_v<RealType, float> ||
                                    std::is_same_v<RealType, double>
                                   >::type * = nullptr>
  struct FFT {
    private:
      int nx1_, nx2_, nb_batches_;
      int nx1h_, nx2h_;

      // fftwPlan type is defined for each precision
      using fftwPlanType = typename std::conditional_t<std::is_same_v<RealType, float>, fftwf_plan, fftw_plan>;
      fftwPlanType forward_c2c_plan_, forward_r2c_plan_;
      fftwPlanType backward_c2c_plan_, backward_c2r_plan_;

      // Normalization coefficient
      RealType normcoeff_;

      // Thread private buffers
      Complex<RealType> *buffer_c_;
      Complex<RealType> *thread_private_buffers_nx1h_, *thread_private_buffers_nx2_;
      Complex<RealType> *thread_private_buffers_nx2_out_;
      RealType          *thread_private_buffers_nx1_r2c_;
      RealType          *thread_private_buffers_nx1_c2r_;

    public:
      using array_layout = LayoutPolicy;

    public:
    FFT(int nx1, int nx2)
      : nx1_(nx1), nx2_(nx2), nb_batches_(1),
        buffer_c_(nullptr), thread_private_buffers_nx1h_(nullptr),
        thread_private_buffers_nx2_(nullptr), thread_private_buffers_nx2_out_(nullptr),
        thread_private_buffers_nx1_r2c_(nullptr), thread_private_buffers_nx1_c2r_(nullptr) {
      init();
    }

    FFT(int nx1, int nx2, int batch)
      : nx1_(nx1), nx2_(nx2), nb_batches_(batch), 
        buffer_c_(nullptr), thread_private_buffers_nx1h_(nullptr),
        thread_private_buffers_nx2_(nullptr), thread_private_buffers_nx2_out_(nullptr),
        thread_private_buffers_nx1_r2c_(nullptr), thread_private_buffers_nx1_c2r_(nullptr) {
      init();
    }

    virtual ~FFT() {
      if(buffer_c_ != nullptr)                       delete [] buffer_c_;
      if(thread_private_buffers_nx1h_ != nullptr)    delete [] thread_private_buffers_nx1h_;
      if(thread_private_buffers_nx2_ != nullptr)     delete [] thread_private_buffers_nx2_;
      if(thread_private_buffers_nx2_out_ != nullptr) delete [] thread_private_buffers_nx2_out_;
      if(thread_private_buffers_nx1_r2c_ != nullptr) delete [] thread_private_buffers_nx1_r2c_;
      if(thread_private_buffers_nx1_c2r_ != nullptr) delete [] thread_private_buffers_nx1_c2r_;

      destroyPlans();
    }

    public:
    RealType normcoeff() {return normcoeff_;}
    int nx1() {return nx1_;}
    int nx2() {return nx2_;}
    int nx1h() {return nx1h_;}
    int nx2h() {return nx2h_;}
    int nb_batches() {return nb_batches_;}
    // Interfaces called from outside 
    void fft(Complex<RealType> *dptr_in, Complex<RealType> *dptr_out) {
      fft_(dptr_in, dptr_out);
    }

    void rfft(RealType *dptr_in, Complex<RealType> *dptr_out) {
      rfft_(dptr_in, dptr_out);
    }

    void ifft(Complex<RealType> *dptr_in, Complex<RealType> *dptr_out) {
      ifft_(dptr_in, dptr_out);
    }

    void irfft(Complex<RealType> *dptr_in, RealType *dptr_out) {
      irfft_(dptr_in, dptr_out);
    }

    /* @brief 2D FFT wrapper for batched case
     * @param[in]  dptr_in(nx1h,nx2,batch)
     * @param[out] dptr_out(nx1,nx2,batch)
     */
    void rfft2(RealType *dptr_in, Complex<RealType> *dptr_out) {
      if(std::is_same_v<LayoutPolicy, layout_contiguous_at_left>) {
        if(nb_batches_ == 1) {
          rfft2_serial_left(dptr_in, dptr_out);
        } else {
          rfft2_batch_left(dptr_in, dptr_out);
        }
      } else {
        if(nb_batches_ == 1) {
          rfft2_serial_right(dptr_in, dptr_out);
        } else {
          rfft2_batch_right(dptr_in, dptr_out);
        }
      }
    }

    /* @brief 2D FFT wrapper for batched case
     * @param[in]  dptr_in(nx1h,nx2,batch)
     * @param[out] dptr_out(nx1,nx2,batch)
     */
    void irfft2(Complex<RealType> *dptr_in, RealType *dptr_out) {
      if(std::is_same_v<LayoutPolicy, layout_contiguous_at_left>) {
        if(nb_batches_ == 1) {
          irfft2_serial_left(dptr_in, dptr_out);
        }
        else {
          irfft2_batch_left(dptr_in, dptr_out);
        }
      } else {
        if(nb_batches_ == 1) {
          irfft2_serial_right(dptr_in, dptr_out);
        }
        else {
          irfft2_batch_right(dptr_in, dptr_out);
        }
      }
    }

    private:
    // Initialization
    // float type
    template <typename RType = RealType>
    typename std::enable_if_t<std::is_same_v<RType, float>, void>
    init() {
      nx1h_ = nx1_/2 + 1;
      nx2h_ = nx2_/2 + 1;

      normcoeff_ = static_cast<RealType>(1.0) / static_cast<RealType>(nx1_ * nx2_);

      assert(nb_batches_ >= 1);

      // Single precision
      fftwf_complex *c_in, *c_out;
      fftwf_complex *c_in_c2r, *c_out_r2c;
      float         *in, *out;

      c_in  = fftwf_alloc_complex(nx2_);
      c_out = fftwf_alloc_complex(nx2_);
      
      in  = fftwf_alloc_real(nx1_);
      out = fftwf_alloc_real(nx1_+2);
      c_in_c2r = fftwf_alloc_complex(nx1h_);
      c_out_r2c = fftwf_alloc_complex(nx1h_);

      forward_c2c_plan_  = fftwf_plan_dft_1d(nx2_, c_in, c_out, FFTW_FORWARD,  FFTW_ESTIMATE);
      backward_c2c_plan_ = fftwf_plan_dft_1d(nx2_, c_out, c_in, FFTW_BACKWARD, FFTW_ESTIMATE);

      forward_r2c_plan_  = fftwf_plan_dft_r2c_1d(nx1_, in, c_out_r2c, FFTW_ESTIMATE);
      backward_c2r_plan_ = fftwf_plan_dft_c2r_1d(nx1_, c_in_c2r, out, FFTW_ESTIMATE);

      fftwf_free(in);   fftwf_free(out);
      fftwf_free(c_in); fftwf_free(c_out);
      fftwf_free(c_in_c2r); fftwf_free(c_out_r2c);

      // Malloc thread private buffers
      int nb_threads=0;
      #pragma omp parallel
      nb_threads = omp_get_num_threads();

      buffer_c_                       = new Complex<RealType>[nb_batches_*nx1h_*nx2_];
      thread_private_buffers_nx1h_    = new Complex<RealType>[nb_threads*nx1h_];
      thread_private_buffers_nx2_     = new Complex<RealType>[nb_threads*nx2_];
      thread_private_buffers_nx2_out_ = new Complex<RealType>[nb_threads*nx2_];
      thread_private_buffers_nx1_r2c_ = new RealType[nb_threads*nx1_];
      thread_private_buffers_nx1_c2r_ = new RealType[nb_threads*(nx1_+2)];
    }

    // double type
    template <typename RType = RealType>
    typename std::enable_if_t<std::is_same_v<RType, double>, void>
    init() {
      nx1h_ = nx1_/2 + 1;
      nx2h_ = nx2_/2 + 1;

      normcoeff_ = static_cast<RealType>(1.0) / static_cast<RealType>(nx1_ * nx2_);

      assert(nb_batches_ >= 1);

      // Double precision
      fftw_complex *c_in, *c_out;
      fftw_complex *c_in_c2r, *c_out_r2c;
      double      *in, *out;

      c_in = fftw_alloc_complex(nx2_);
      c_out = fftw_alloc_complex(nx2_);
      
      in = fftw_alloc_real(nx1_);
      out = fftw_alloc_real(nx1_+2);
      c_in_c2r = fftw_alloc_complex(nx1h_);
      c_out_r2c = fftw_alloc_complex(nx1h_);

      forward_c2c_plan_  = fftw_plan_dft_1d(nx2_, c_in, c_out, FFTW_FORWARD,  FFTW_ESTIMATE);
      backward_c2c_plan_ = fftw_plan_dft_1d(nx2_, c_out, c_in, FFTW_BACKWARD, FFTW_ESTIMATE);

      forward_r2c_plan_  = fftw_plan_dft_r2c_1d(nx1_, in, c_out_r2c, FFTW_ESTIMATE);
      backward_c2r_plan_ = fftw_plan_dft_c2r_1d(nx1_, c_in_c2r, out, FFTW_ESTIMATE);

      fftw_free(in);   fftw_free(out);
      fftw_free(c_in); fftw_free(c_out);
      fftw_free(c_in_c2r); fftw_free(c_out_r2c);

      // Malloc thread private buffers
      int nb_threads=0;
      #pragma omp parallel
      nb_threads = omp_get_num_threads();

      buffer_c_                       = new Complex<RealType>[nb_batches_*nx1h_*nx2_];
      thread_private_buffers_nx1h_    = new Complex<RealType>[nb_threads*nx1h_];
      thread_private_buffers_nx2_     = new Complex<RealType>[nb_threads*nx2_];
      thread_private_buffers_nx2_out_ = new Complex<RealType>[nb_threads*nx2_];
      thread_private_buffers_nx1_r2c_ = new RealType[nb_threads*nx1_];
      thread_private_buffers_nx1_c2r_ = new RealType[nb_threads*(nx1_+2)];
    }
    
    // Finalization
    // float
    template <typename RType = RealType>
    typename std::enable_if_t<std::is_same_v<RType, float>, void>
    destroyPlans() {
      fftwf_destroy_plan(forward_c2c_plan_);
      fftwf_destroy_plan(backward_c2c_plan_);
      fftwf_destroy_plan(forward_r2c_plan_);
      fftwf_destroy_plan(backward_c2r_plan_);
    }

    // double
    template <typename RType = RealType>
    typename std::enable_if_t<std::is_same_v<RType, double>, void>
    destroyPlans() {
      fftw_destroy_plan(forward_c2c_plan_);
      fftw_destroy_plan(backward_c2c_plan_);
      fftw_destroy_plan(forward_r2c_plan_);
      fftw_destroy_plan(backward_c2r_plan_);
    }

    private:
    // Layout Left specialization
    /* @brief 2D FFT wrapper for batched case (LayoutLeft)
     * @param[in]  dptr_in(nx1,nx2)
     * @param[out] dptr_out(nx1h,nx2)
     */
    void rfft2_serial_left(RealType *dptr_in, Complex<RealType> *dptr_out) {
      #pragma omp parallel
      {
        int tid = omp_get_thread_num();
        RealType          *thread_private_buffer_nx1  = &thread_private_buffers_nx1_r2c_[nx1_*tid];
        Complex<RealType> *thread_private_buffer_nx1h = &thread_private_buffers_nx1h_[nx1h_*tid];
        Complex<RealType> *thread_private_buffer_nx2  = &thread_private_buffers_nx2_[nx2_*tid];
        // Fourier Transform in x direction
        #pragma omp for schedule(static)
        for(int ix2=0; ix2 < nx2_; ix2++) {
          for(int ix1=0; ix1 < nx1_; ix1++) {
            int idx = Index::coord_2D2int(ix1, ix2, nx1_, nx2_);
            thread_private_buffer_nx1[ix1] = dptr_in[idx];
          }
          rfft(thread_private_buffer_nx1, thread_private_buffer_nx1h);

          // Transpose (nx1h, nx2) -> (nx2, nx1h)
          for(int ix1=0; ix1 < nx1h_; ix1++) {
            int idx = Index::coord_2D2int(ix2, ix1, nx2_, nx1h_);
            buffer_c_[idx] = thread_private_buffer_nx1h[ix1];
          }
        }

        // Fourier Transform in y direction
        #pragma omp for schedule(static)
        for(int ix1=0; ix1 < nx1h_; ix1++) {
          int offset = nx2_ * ix1;
          fft(&buffer_c_[offset], thread_private_buffer_nx2);
          for(int ix2=0; ix2 < nx2_; ix2++) {
            int idx = Index::coord_2D2int(ix1, ix2, nx1h_, nx2_);
            dptr_out[idx] = thread_private_buffer_nx2[ix2];
          }
        }
      }
    }

    /* @brief 2D FFT wrapper for batched case (LayoutLeft)
     * @param[in]  dptr_in(nx1,nx2,batch)
     * @param[out] dptr_out(nx1h,nx2,batch)
     */
    void rfft2_batch_left(RealType *dptr_in, Complex<RealType> *dptr_out) {
      #pragma omp parallel
      {
        int tid = omp_get_thread_num();
        RealType          *thread_private_buffer_nx1  = &thread_private_buffers_nx1_r2c_[nx1_*tid];
        Complex<RealType> *thread_private_buffer_nx1h = &thread_private_buffers_nx1h_[nx1h_*tid];
        Complex<RealType> *thread_private_buffer_nx2  = &thread_private_buffers_nx2_[nx2_*tid];
        // Fourier Transform in x direction
        #pragma omp for schedule(static), collapse(2)
        for(int ib=0; ib<nb_batches_; ib++) {
          for(int ix2=0; ix2 < nx2_; ix2++) {
            for(int ix1=0; ix1 < nx1_; ix1++) {
              int idx = Index::coord_3D2int(ix1, ix2, ib, nx1_, nx2_, nb_batches_);
              thread_private_buffer_nx1[ix1] = dptr_in[idx];
            }

            rfft(thread_private_buffer_nx1, thread_private_buffer_nx1h);

            // Transpose (nx1h, nx2, batch) -> (nx2, nx1h, batch)
            for(int ix1=0; ix1 < nx1h_; ix1++) {
              int idx = Index::coord_3D2int(ix2, ix1, ib, nx2_, nx1h_, nb_batches_);
              buffer_c_[idx] = thread_private_buffer_nx1h[ix1];
            }
          }
        }

        // Fourier Transform in y direction
        #pragma omp for schedule(static), collapse(2)
        for(int ib=0; ib<nb_batches_; ib++) {
          for(int ix1=0; ix1 < nx1h_; ix1++) {
            int offset = nx2_ * Index::coord_2D2int(ix1, ib, nx1h_, nb_batches_);
            fft(&buffer_c_[offset], thread_private_buffer_nx2);
            for(int ix2=0; ix2 < nx2_; ix2++) {
              int idx = Index::coord_3D2int(ix1, ix2, ib, nx1h_, nx2_, nb_batches_);
              dptr_out[idx] = thread_private_buffer_nx2[ix2];
            }
          }
        }
      }
    }

    /* @brief 2D FFT wrapper for serial case (LayoutLeft)
     * @param[in]  dptr_in(nx1h,nx2)
     * @param[out] dptr_out(nx1,nx2)
     */
    void irfft2_serial_left(Complex<RealType> *dptr_in, RealType *dptr_out) {
      #pragma omp parallel
      {
        int tid = omp_get_thread_num();
        RealType          *thread_private_buffer_nx1 = &thread_private_buffers_nx1_c2r_[(nx1_+2)*tid];
        Complex<RealType> *thread_private_buffer_nx2 = &thread_private_buffers_nx2_[nx2_*tid];
        Complex<RealType> *thread_private_buffer_nx2_out = &thread_private_buffers_nx2_out_[nx2_*tid];
        // Inverse Fourier Transform in y direction
        #pragma omp for schedule(static)
        for(int ix1=0; ix1 < nx1h_; ix1++) {
          for(int ix2=0; ix2 < nx2_; ix2++) {
            int idx = Index::coord_2D2int(ix1, ix2, nx1h_,nx2_);
            thread_private_buffer_nx2[ix2] = dptr_in[idx];
          }
          ifft(thread_private_buffer_nx2, thread_private_buffer_nx2_out);
          for(int ix2=0; ix2 < nx2_; ix2++) {
            int idx = Index::coord_2D2int(ix1, ix2, nx1h_, nx2_);
            buffer_c_[idx] = thread_private_buffer_nx2_out[ix2];
          }
        }
        
        // Inverse Fourier Transform in x direction
        #pragma omp for schedule(static)
        for(int ix2=0; ix2 < nx2_; ix2++) {
          int offset_in  = nx1h_ * ix2;

          irfft(&buffer_c_[offset_in], thread_private_buffer_nx1);
          for(int ix1=0; ix1 < nx1_; ix1++) {
            int idx = Index::coord_2D2int(ix1, ix2, nx1_, nx2_);
            dptr_out[idx] = thread_private_buffer_nx1[ix1];
          }
        }
      }
    }

    /* @brief 2D FFT wrapper for batched case
     *        In the host code, we assume LayoutRight (C style)
     * @param[in]  dptr_in(nx1h,nx2,batch)
     * @param[out] dptr_out(nx1,nx2,batch)
     */
    void irfft2_batch_left(Complex<RealType> *dptr_in, RealType *dptr_out) {
      #pragma omp parallel
      {
        int tid = omp_get_thread_num();
        RealType          *thread_private_buffer_nx1     = &thread_private_buffers_nx1_c2r_[(nx1_+2)*tid];
        Complex<RealType> *thread_private_buffer_nx2     = &thread_private_buffers_nx2_[nx2_*tid];
        Complex<RealType> *thread_private_buffer_nx2_out = &thread_private_buffers_nx2_out_[nx2_*tid];
        // Inverse Fourier Transform in y direction
        #pragma omp for schedule(static), collapse(2)
        for(int ib=0; ib < nb_batches_; ib++) {
          for(int ix1=0; ix1 < nx1h_; ix1++) {
            for(int ix2=0; ix2 < nx2_; ix2++) {
              int idx = Index::coord_3D2int(ix1, ix2, ib, nx1h_, nx2_, nb_batches_);
              thread_private_buffer_nx2[ix2] = dptr_in[idx];
            }
            ifft(thread_private_buffer_nx2, thread_private_buffer_nx2_out);
            for(int ix2=0; ix2 < nx2_; ix2++) {
              int idx = Index::coord_3D2int(ix1, ix2, ib, nx1h_, nx2_, nb_batches_);
              buffer_c_[idx] = thread_private_buffer_nx2_out[ix2];
            }
          }
        }
        
        // Inverse Fourier Transform in x direction
        #pragma omp for schedule(static), collapse(2)
        for(int ib=0; ib < nb_batches_; ib++) {
          for(int ix2=0; ix2 < nx2_; ix2++) {
            int offset  = nx1h_ * Index::coord_2D2int(ix2, ib, nx2_, nb_batches_);

            irfft(&buffer_c_[offset], thread_private_buffer_nx1);
            for(int ix1=0; ix1 < nx1_; ix1++) {
              int idx = Index::coord_3D2int(ix1, ix2, ib, nx1_, nx2_, nb_batches_);
              dptr_out[idx] = thread_private_buffer_nx1[ix1];
            }
          }
        }
      }
    }

    private:
    // Layout Right specialization
    /* @brief 2D FFT wrapper for batched case (LayoutRight)
     * @param[in]  dptr_in(nx1,nx2)
     * @param[out] dptr_out(nx1h,nx2)
     */
    void rfft2_serial_right(RealType *dptr_in, Complex<RealType> *dptr_out) {
      #pragma omp parallel
      {
        int tid = omp_get_thread_num();
        RealType          *thread_private_buffer_nx1  = &thread_private_buffers_nx1_r2c_[nx1_*tid];
        Complex<RealType> *thread_private_buffer_nx1h = &thread_private_buffers_nx1h_[nx1h_*tid];
        Complex<RealType> *thread_private_buffer_nx2  = &thread_private_buffers_nx2_[nx2_*tid];
        // Fourier Transform in x direction
        #pragma omp for schedule(static)
        for(int ix2=0; ix2 < nx2_; ix2++) {
          for(int ix1=0; ix1 < nx1_; ix1++) {
            int idx = Index::coord_2D2int(ix2, ix1, nx2_, nx1_);
            thread_private_buffer_nx1[ix1] = dptr_in[idx];
          }
          rfft(thread_private_buffer_nx1, thread_private_buffer_nx1h);

          for(int ix1=0; ix1 < nx1h_; ix1++) {
            int idx = Index::coord_2D2int(ix2, ix1, nx2_, nx1h_);
            buffer_c_[idx] = thread_private_buffer_nx1h[ix1];
          }
        }

        // Fourier Transform in y direction
        #pragma omp for schedule(static)
        for(int ix1=0; ix1 < nx1h_; ix1++) {
          int offset = nx2_ * ix1;
          fft(&buffer_c_[offset], thread_private_buffer_nx2);
          for(int ix2=0; ix2 < nx2_; ix2++) {
            int idx = Index::coord_2D2int(ix2, ix1, nx2_, nx1h_);
            dptr_out[idx] = thread_private_buffer_nx2[ix2];
          }
        }
      }
    }

    /* @brief 2D FFT wrapper for batched case
     *        In the host code, we assume LayoutRight (C style)
     * @param[in]  dptr_in(nx1,nx2,batch)
     * @param[out] dptr_out(nx1h,nx2,batch)
     */
    void rfft2_batch_right(RealType *dptr_in, Complex<RealType> *dptr_out) {
      #pragma omp parallel
      {
        int tid = omp_get_thread_num();
        RealType          *thread_private_buffer_nx1  = &thread_private_buffers_nx1_r2c_[nx1_*tid];
        Complex<RealType> *thread_private_buffer_nx1h = &thread_private_buffers_nx1h_[nx1h_*tid];
        Complex<RealType> *thread_private_buffer_nx2  = &thread_private_buffers_nx2_[nx2_*tid];
        // Fourier Transform in x direction
        #pragma omp for schedule(static), collapse(2)
        for(int ib=0; ib<nb_batches_; ib++) {
          for(int ix2=0; ix2 < nx2_; ix2++) {
            for(int ix1=0; ix1 < nx1_; ix1++) {
              int idx = Index::coord_3D2int(ib, ix2, ix1, nb_batches_, nx2_, nx1_);
              thread_private_buffer_nx1[ix1] = dptr_in[idx];
            }

            rfft(thread_private_buffer_nx1, thread_private_buffer_nx1h);

            for(int ix1=0; ix1 < nx1h_; ix1++) {
              int idx = Index::coord_3D2int(ix2, ix1, ib, nx2_, nx1h_, nb_batches_);
              buffer_c_[idx] = thread_private_buffer_nx1h[ix1];
            }
          }
        }

        // Fourier Transform in y direction
        #pragma omp for schedule(static), collapse(2)
        for(int ib=0; ib<nb_batches_; ib++) {
          for(int ix1=0; ix1 < nx1h_; ix1++) {
            int offset = nx2_ * Index::coord_2D2int(ix1, ib, nx1h_, nb_batches_);
            fft(&buffer_c_[offset], thread_private_buffer_nx2);
            for(int ix2=0; ix2 < nx2_; ix2++) {
              int idx = Index::coord_3D2int(ib, ix2, ix1, nb_batches_, nx2_, nx1h_);
              dptr_out[idx] = thread_private_buffer_nx2[ix2];
            }
          }
        }
      }
    }

    /* @brief 2D FFT wrapper for serial case
     *        In the host code, we assume LayoutRight (C style)
     * @param[in]  dptr_in(nx1h,nx2)
     * @param[out] dptr_out(nx1,nx2)
     */
    void irfft2_serial_right(Complex<RealType> *dptr_in, RealType *dptr_out) {
      #pragma omp parallel
      {
        int tid = omp_get_thread_num();
        RealType          *thread_private_buffer_nx1 = &thread_private_buffers_nx1_c2r_[(nx1_+2)*tid];
        Complex<RealType> *thread_private_buffer_nx2 = &thread_private_buffers_nx2_[nx2_*tid];
        Complex<RealType> *thread_private_buffer_nx2_out = &thread_private_buffers_nx2_out_[nx2_*tid];
        // Inverse Fourier Transform in y direction
        #pragma omp for schedule(static)
        for(int ix1=0; ix1 < nx1h_; ix1++) {
          for(int ix2=0; ix2 < nx2_; ix2++) {
            int idx = Index::coord_2D2int(ix2, ix1, nx2_, nx1h_);
            thread_private_buffer_nx2[ix2] = dptr_in[idx];
          }
          ifft(thread_private_buffer_nx2, thread_private_buffer_nx2_out);
          for(int ix2=0; ix2 < nx2_; ix2++) {
            int idx = Index::coord_2D2int(ix1, ix2, nx1h_, nx2_);
            buffer_c_[idx] = thread_private_buffer_nx2_out[ix2];
          }
        }
        
        // Inverse Fourier Transform in x direction
        #pragma omp for schedule(static)
        for(int ix2=0; ix2 < nx2_; ix2++) {
          int offset_in  = nx1h_ * ix2;

          irfft(&buffer_c_[offset_in], thread_private_buffer_nx1);
          for(int ix1=0; ix1 < nx1_; ix1++) {
            int idx = Index::coord_2D2int(ix2, ix1, nx2_, nx1_);
            dptr_out[idx] = thread_private_buffer_nx1[ix1];
          }
        }
      }
    }

    /* @brief 2D FFT wrapper for batched case (LayoutRight)
     * @param[in]  dptr_in(nx1h,nx2,batch)
     * @param[out] dptr_out(nx1,nx2,batch)
     */
    void irfft2_batch_right(Complex<RealType> *dptr_in, RealType *dptr_out) {
      #pragma omp parallel
      {
        int tid = omp_get_thread_num();
        RealType          *thread_private_buffer_nx1     = &thread_private_buffers_nx1_c2r_[(nx1_+2)*tid];
        Complex<RealType> *thread_private_buffer_nx2     = &thread_private_buffers_nx2_[nx2_*tid];
        Complex<RealType> *thread_private_buffer_nx2_out = &thread_private_buffers_nx2_out_[nx2_*tid];
        // Inverse Fourier Transform in y direction
        #pragma omp for schedule(static), collapse(2)
        for(int ib=0; ib < nb_batches_; ib++) {
          for(int ix1=0; ix1 < nx1h_; ix1++) {
            for(int ix2=0; ix2 < nx2_; ix2++) {
              int idx = Index::coord_3D2int(ib, ix2, ix1, nb_batches_, nx2_, nx1h_);
              thread_private_buffer_nx2[ix2] = dptr_in[idx];
            }
            ifft(thread_private_buffer_nx2, thread_private_buffer_nx2_out);
            for(int ix2=0; ix2 < nx2_; ix2++) {
              int idx = Index::coord_3D2int(ix1, ix2, ib, nx1h_, nx2_, nb_batches_);
              buffer_c_[idx] = thread_private_buffer_nx2_out[ix2];
            }
          }
        }
        
        // Inverse Fourier Transform in x direction
        #pragma omp for schedule(static), collapse(2)
        for(int ib=0; ib < nb_batches_; ib++) {
          for(int ix2=0; ix2 < nx2_; ix2++) {
            int offset  = nx1h_ * Index::coord_2D2int(ix2, ib, nx2_, nb_batches_);

            irfft(&buffer_c_[offset], thread_private_buffer_nx1);
            for(int ix1=0; ix1 < nx1_; ix1++) {
              int idx = Index::coord_3D2int(ib, ix2, ix1, nb_batches_, nx2_, nx1_);
              dptr_out[idx] = thread_private_buffer_nx1[ix1];
            }
          }
        }
      }
    }

    private:
    // 1D FFT wrappers for FFTW
    // FFTW interfaces for double precision
    void fft_(Complex<double> *dptr_in, Complex<double> *dptr_out) {
      fftw_complex *in  = reinterpret_cast<fftw_complex*>(dptr_in);
      fftw_complex *out = reinterpret_cast<fftw_complex*>(dptr_out);
      fftw_execute_dft(forward_c2c_plan_, in, out);
    }

    void rfft_(double *dptr_in, Complex<double> *dptr_out) {
      fftw_complex *out = reinterpret_cast<fftw_complex*>(dptr_out);
      fftw_execute_dft_r2c(forward_r2c_plan_, dptr_in, out);
    }

    void ifft_(Complex<double> *dptr_in, Complex<double> *dptr_out) {
      fftw_complex *in  = reinterpret_cast<fftw_complex*>(dptr_in);
      fftw_complex *out = reinterpret_cast<fftw_complex*>(dptr_out);
      fftw_execute_dft(backward_c2c_plan_, in, out);
    }

    void irfft_(Complex<double> *dptr_in, double *dptr_out) {
      fftw_complex *in = reinterpret_cast<fftw_complex*>(dptr_in);
      fftw_execute_dft_c2r(backward_c2r_plan_, in, dptr_out);
    }

    // FFTW interfaces for single precision
    void fft_(Complex<float> *dptr_in, Complex<float> *dptr_out) {
      fftwf_complex *in  = reinterpret_cast<fftwf_complex*>(dptr_in);
      fftwf_complex *out = reinterpret_cast<fftwf_complex*>(dptr_out);
      fftwf_execute_dft(forward_c2c_plan_, in, out);
    }

    void rfft_(float *dptr_in, Complex<float> *dptr_out) {
      fftwf_complex *out = reinterpret_cast<fftwf_complex*>(dptr_out);
      fftwf_execute_dft_r2c(forward_r2c_plan_, dptr_in, out);
    }

    void ifft_(Complex<float> *dptr_in, Complex<float> *dptr_out) {
      fftwf_complex *in  = reinterpret_cast<fftwf_complex*>(dptr_in);
      fftwf_complex *out = reinterpret_cast<fftwf_complex*>(dptr_out);
      fftwf_execute_dft(backward_c2c_plan_, in, out);
    }

    void irfft_(Complex<float> *dptr_in, float *dptr_out) {
      fftwf_complex *in = reinterpret_cast<fftwf_complex*>(dptr_in);
      fftwf_execute_dft_c2r(backward_c2r_plan_, in, dptr_out);
    }

  };
};

#endif

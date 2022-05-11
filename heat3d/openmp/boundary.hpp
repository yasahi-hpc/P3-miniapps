#ifndef __MPI_COMM_HPP__
#define __MPI_COMM_HPP__

#include <cassert>
#include <vector>
#include <complex>
#include "../Timer.hpp"
#include "types.hpp"
#include "Config.hpp"

struct Halo {
  using value_type = RealView2D::value_type;
  using int_type = int;

  std::string name_;
  RealView2D left_buffer_, right_buffer_;
  size_t size_;

  Halo() = delete;
  Halo(const std::string name, std::array<size_t, 2> shape) : name_(name) {
    #if defined( ENABLE_OPENMP_OFFLOAD )
      #pragma omp target enter data map(alloc: this[0:1])
    #endif

    left_buffer_  = RealView2D(name + "_left_buffer", shape[0], shape[1]);
    right_buffer_ = RealView2D(name + "_right_buffer", shape[0], shape[1]);
    left_buffer_.fill();
    right_buffer_.fill();
    assert(left_buffer_.size() == right_buffer_.size() );
    size_ = left_buffer_.size();
  }
 
  ~Halo() {
    #if defined( ENABLE_OPENMP_OFFLOAD )
      #pragma omp target exit data map(delete: this[0:1])
    #endif
  }
 
  const std::string name() const noexcept {return name_;}
  RealView2D left_buffer() const { return left_buffer_; }
  RealView2D right_buffer() const { return right_buffer_; }
  RealView2D &left_buffer() { return left_buffer_; }
  RealView2D &right_buffer() { return right_buffer_; }

  size_t size() const { return size_; }
};

struct Boundary {
  using value_type = RealView3D::value_type;

private:

  // Data shape
  std::vector<size_t> shape_;

  // Halo data
  Halo *x_send_halo_, *x_recv_halo_;
  Halo *y_send_halo_, *y_recv_halo_;
  Halo *z_send_halo_, *z_recv_halo_;

  int halo_width_;

public:
  Boundary() = delete;

  Boundary(const std::vector<size_t> &shape)
    : shape_(shape), halo_width_(1) {

    #if defined( ENABLE_OPENMP_OFFLOAD )
      #pragma omp target enter data map(alloc: this[0:1])
    #endif
    setHalos();
  }

  ~Boundary() {
    #if defined( ENABLE_OPENMP_OFFLOAD )
      #pragma omp target exit data map(delete: this[0:1])
    #endif
    delete x_send_halo_;
    delete y_send_halo_;
    delete z_send_halo_;
    delete x_recv_halo_;
    delete y_recv_halo_;
    delete z_recv_halo_;
  }

  void exchangeHalos(const Config &conf, RealView3D &u, std::vector<Timer*> &timers) {
    bool use_timer = timers.size() > 0;

    if(use_timer) timers[HaloPack]->begin();
    pack(conf, u);
    if(use_timer) timers[HaloPack]->end();

    if(use_timer) timers[HaloSwap]->begin();
    swap(x_recv_halo_, x_send_halo_);
    swap(y_recv_halo_, y_send_halo_);
    swap(z_recv_halo_, z_send_halo_);
    if(use_timer) timers[HaloSwap]->end();

    if(use_timer) timers[HaloUnpack]->begin();
    unpack(conf, u);
    if(use_timer) timers[HaloUnpack]->end();
  }

private:
  void setHalos() {
    x_send_halo_ = new Halo("x_send", {shape_[1], shape_[2]});
    x_recv_halo_ = new Halo("x_recv", {shape_[1], shape_[2]});

    y_send_halo_ = new Halo("y_send", {shape_[0], shape_[2]});
    y_recv_halo_ = new Halo("y_recv", {shape_[0], shape_[2]});

    z_send_halo_ = new Halo("z_send", {shape_[0], shape_[1]});
    z_recv_halo_ = new Halo("z_recv", {shape_[0], shape_[1]});
  }

  /* Pack data to send buffers
   */
  void pack(const Config &conf, const RealView3D &u) {
    int nx = conf.nx, ny = conf.ny, nz = conf.nz;

    // xhalos
    #if defined(ENABLE_OPENMP_OFFLOAD)
      #pragma omp target teams distribute parallel for simd collapse(2)
    #else
      #pragma omp parallel for
    #endif
    for(int iz=0; iz < nz; iz++) {
      LOOP_SIMD
      for(int iy=0; iy < ny; iy++) {
        x_send_halo_->left_buffer_(iy, iz)  = u(0,    iy, iz);
        x_send_halo_->right_buffer_(iy, iz) = u(nx-1, iy, iz);
      }
    }

    // yhalos
    #if defined(ENABLE_OPENMP_OFFLOAD)
      #pragma omp target teams distribute parallel for simd collapse(2)
    #else
      #pragma omp parallel for
    #endif
    for(int iz=0; iz < nz; iz++) {
      LOOP_SIMD
      for(int ix=0; ix < nx; ix++) {
        y_send_halo_->left_buffer_(ix, iz)  = u(ix, 0,    iz);
        y_send_halo_->right_buffer_(ix, iz) = u(ix, ny-1, iz);
      }
    }

    // zhalos
    #if defined(ENABLE_OPENMP_OFFLOAD)
      #pragma omp target teams distribute parallel for simd collapse(2)
    #else
      #pragma omp parallel for
    #endif
    for(int iy=0; iy < ny; iy++) {
      LOOP_SIMD
      for(int ix=0; ix < nx; ix++) {
        z_send_halo_->left_buffer_(ix, iy)  = u(ix, iy, 0   );
        z_send_halo_->right_buffer_(ix, iy) = u(ix, iy, nz-1);
      }
    }
  }

  /* Unpack data from recv buffers
   */
  void unpack(const Config &conf, RealView3D &u) {
    int nx = conf.nx, ny = conf.ny, nz = conf.nz;

    // xhalos
    #if defined(ENABLE_OPENMP_OFFLOAD)
      #pragma omp target teams distribute parallel for simd collapse(2)
    #else
      #pragma omp parallel for
    #endif
    for(int iz=0; iz < nz; iz++) {
      LOOP_SIMD
      for(int iy=0; iy < ny; iy++) {
        u(-1, iy, iz) = x_recv_halo_->left_buffer_(iy, iz);
        u(nx, iy, iz) = x_recv_halo_->right_buffer_(iy, iz);
      }
    }

    // yhalos
    #if defined(ENABLE_OPENMP_OFFLOAD)
      #pragma omp target teams distribute parallel for simd collapse(2)
    #else
      #pragma omp parallel for
    #endif
    for(int iz=0; iz < nz; iz++) {
      LOOP_SIMD
      for(int ix=0; ix < nx; ix++) {
        u(ix, -1, iz) = y_recv_halo_->left_buffer_(ix, iz);
        u(ix, ny, iz) = y_recv_halo_->right_buffer_(ix, iz);
      }
    }

    // zhalos
    #if defined(ENABLE_OPENMP_OFFLOAD)
      #pragma omp target teams distribute parallel for simd collapse(2)
    #else
      #pragma omp parallel for
    #endif
    for(int iy=0; iy < ny; iy++) {
      LOOP_SIMD
      for(int ix=0; ix < nx; ix++) {
        u(ix, iy, -1) = z_recv_halo_->left_buffer_(ix, iy);
        u(ix, iy, nz) = z_recv_halo_->right_buffer_(ix, iy);
      }
    }
  }

  void swap(Halo *recv, Halo *send) {
    recv->left_buffer_.swap(  send->right_buffer_ );
    recv->right_buffer_.swap( send->left_buffer_ );
  }
};

#endif

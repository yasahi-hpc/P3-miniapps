#ifndef __BOUNDARY_HPP__
#define __BOUNDARY_HPP__

#include <cassert>
#include <vector>
#include "../Timer.hpp"
#include "types.hpp"
#include "Parallel_For.hpp"
#include "utils.hpp"

struct Halo {
  using value_type = RealView2D::value_type;
  using int_type = int;
  std::string name_;
  RealView2D left_buffer_, right_buffer_;
  size_t size_;

  Halo() = delete;
  Halo(const std::string name, std::array<size_t, 2> shape) : name_(name) {
    left_buffer_  = RealView2D(name + "_left_buffer", shape);
    right_buffer_ = RealView2D(name + "_right_buffer", shape);
    left_buffer_.fill(); 
    right_buffer_.fill();
    assert(left_buffer_.size() == right_buffer_.size() );
    size_ = left_buffer_.size();
  }
  
  ~Halo() {}

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
    setHalos();
  }

  ~Boundary() {
    delete x_send_halo_;
    delete y_send_halo_;
    delete z_send_halo_;
    delete x_recv_halo_;
    delete y_recv_halo_;
    delete z_recv_halo_;
  }

  void exchangeHalos(RealView3D &u, std::vector<Timer*> &timers) {
    bool use_timer = timers.size() > 0;
    auto _u = u.mdspan();
    const std::pair inner_x(0, _u.extent(0) - 2 * halo_width_);
    const std::pair inner_y(0, _u.extent(1) - 2 * halo_width_);
    const std::pair inner_z(0, _u.extent(2) - 2 * halo_width_);

    // Exchange in x direction
    {
      int i = 0;
      auto send_left_x  = submdspan(_u, 0, inner_y, inner_z);
      auto send_right_x = submdspan(_u, _u.extent(i) - 2 * halo_width_ - 1, inner_y, inner_z);
      auto recv_left_x  = submdspan(_u, -1, inner_y, inner_z);
      auto recv_right_x = submdspan(_u, _u.extent(i) - 2 * halo_width_, inner_y, inner_z);
      if(use_timer) timers[HaloPack]->begin();
      pack(x_send_halo_, send_left_x, send_right_x);
      if(use_timer) timers[HaloPack]->end();

      if(use_timer) timers[HaloSwap]->begin();
      swap(x_recv_halo_, x_send_halo_);
      if(use_timer) timers[HaloSwap]->end();

      if(use_timer) timers[HaloUnpack]->begin();
      unpack(recv_left_x, recv_right_x, x_recv_halo_);
      if(use_timer) timers[HaloUnpack]->end();
    }

    // Exchange in y direction
    {
      int i = 1;
      auto send_left_y  = submdspan(_u, inner_x, 0, inner_z);
      auto send_right_y = submdspan(_u, inner_x, _u.extent(i) - 2 * halo_width_ - 1, inner_z);
      auto recv_left_y  = submdspan(_u, inner_x, -1, inner_z);
      auto recv_right_y = submdspan(_u, inner_x, _u.extent(i) - 2 * halo_width_, inner_z);
      if(use_timer) timers[HaloPack]->begin();
      pack(y_send_halo_, send_left_y, send_right_y);
      if(use_timer) timers[HaloPack]->end();

      if(use_timer) timers[HaloSwap]->begin();
      swap(y_recv_halo_, y_send_halo_);
      if(use_timer) timers[HaloSwap]->end();

      if(use_timer) timers[HaloUnpack]->begin();
      unpack(recv_left_y, recv_right_y, y_recv_halo_);
      if(use_timer) timers[HaloUnpack]->end();
    }

    // Exchange in z direction
    {
      int i = 2;
      auto send_left_z  = submdspan(_u, inner_x, inner_y, 0);
      auto send_right_z = submdspan(_u, inner_x, inner_y, _u.extent(i) - 2 * halo_width_ - 1);
      auto recv_left_z  = submdspan(_u, inner_x, inner_y, -1);
      auto recv_right_z = submdspan(_u, inner_x, inner_y, _u.extent(i) - 2 * halo_width_);
      if(use_timer) timers[HaloPack]->begin();
      pack(z_send_halo_, send_left_z, send_right_z);
      if(use_timer) timers[HaloPack]->end();

      if(use_timer) timers[HaloSwap]->begin();
      swap(z_recv_halo_, z_send_halo_);
      if(use_timer) timers[HaloSwap]->end();

      if(use_timer) timers[HaloUnpack]->begin();
      unpack(recv_left_z, recv_right_z, z_recv_halo_);
      if(use_timer) timers[HaloUnpack]->end();
    }
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

public:
  /* Pack data to send buffers
   */
  template < class mdspan2d_type >
  void pack(const Halo *halo, const mdspan2d_type &left, const mdspan2d_type &right) {
    auto left_buffer = halo->left_buffer().mdspan();
    auto right_buffer = halo->right_buffer().mdspan();
    assert( left.extents() == right.extents() );
    assert( left.extents() == left_buffer.extents() );
    assert( left.extents() == right_buffer.extents() );

    Iterate_policy<2> policy2d({0, 0}, {left.extent(0), left.extent(1)});
    Impl::for_each(policy2d,
                   [=](const int ix, const int iy) {
                     left_buffer(ix, iy)  = left(ix, iy);
                     right_buffer(ix, iy) = right(ix, iy);
                   });
  }

  /* Unpack data from recv buffers
   */
  template < class mdspan2d_type >
  void unpack(const mdspan2d_type &left, const mdspan2d_type &right, const Halo *halo) {
    auto left_buffer = halo->left_buffer().mdspan();
    auto right_buffer = halo->right_buffer().mdspan();
    assert( left.extents() == right.extents() );
    assert( left.extents() == left_buffer.extents() );
    assert( left.extents() == right_buffer.extents() );

    Iterate_policy<2> policy2d({0, 0}, {left.extent(0), left.extent(1)});
    Impl::for_each(policy2d,
                   [=](const int ix, const int iy) {
                     left(ix, iy) = left_buffer(ix, iy);
                     right(ix, iy) = right_buffer(ix, iy);
                   });
  }

private:
  void swap(Halo *recv, const Halo *send) {
    auto send_left_buffer  = send->left_buffer();
    auto send_right_buffer = send->right_buffer();
    recv->left_buffer().swap( send_right_buffer );
    recv->right_buffer().swap( send_left_buffer );
  }

};

#endif

#ifndef __MPI_COMM_HPP__
#define __MPI_COMM_HPP__

#include <cassert>
#include <vector>
#include <complex>
#include <mpi.h>
#include "../Timer.hpp"
#include "Types.hpp"
#include "Parallel_For.hpp"
#include "Utils.hpp"

constexpr int UP     = 0;
constexpr int DOWN   = 1;
constexpr int LEFT   = 2;
constexpr int RIGHT  = 3;
constexpr int TOP    = 4;
constexpr int BOTTOM = 5;

template <typename RealType> using Complex = std::complex<RealType>;

template <typename T,
          std::enable_if_t<std::is_same_v<T, int             > ||
                           std::is_same_v<T, float           > ||
                           std::is_same_v<T, double          > ||
                           std::is_same_v<T, Complex<float>  > ||
                           std::is_same_v<T, Complex<double> > 
                           , std::nullptr_t> = nullptr
>
MPI_Datatype get_mpi_data_type() {
  MPI_Datatype type;
  if(std::is_same_v<T, int             >) type = MPI_INT;
  if(std::is_same_v<T, float           >) type = MPI_FLOAT;
  if(std::is_same_v<T, double          >) type = MPI_DOUBLE;
  if(std::is_same_v<T, Complex<float>  >) type = MPI_COMPLEX;
  if(std::is_same_v<T, Complex<double> >) type = MPI_DOUBLE_COMPLEX;

  return type;
}

struct Halo {
  using value_type = RealView2D::value_type;
  using int_type = int;
  std::string name_;
  RealView2D left_buffer_, right_buffer_;
  size_t size_;
  int_type left_rank_, right_rank_;
  int_type left_tag_, right_tag_;
  MPI_Comm communicator_;
  MPI_Datatype mpi_data_type_;
  bool is_comm_;

  Halo() = delete;
  Halo(const std::string name, std::array<size_t, 2> shape, int_type left_rank, int_type right_rank,
       int_type left_tag, int_type right_tag, MPI_Comm communicator, bool is_comm)
    : name_(name), left_rank_(left_rank), right_rank_(right_rank),
    left_tag_(left_tag), right_tag_(right_tag), communicator_(communicator), is_comm_(is_comm) {
    left_buffer_  = RealView2D(name + "_left_buffer", shape);
    right_buffer_ = RealView2D(name + "_right_buffer", shape);
    left_buffer_.fill(); 
    right_buffer_.fill();
    assert(left_buffer_.size() == right_buffer_.size() );
    size_ = left_buffer_.size();
    mpi_data_type_ = get_mpi_data_type<value_type>();
  }
  
  ~Halo() {}

  const std::string name() const noexcept {return name_;}
  RealView2D left_buffer() const { return left_buffer_; }
  RealView2D right_buffer() const { return right_buffer_; }
  RealView2D &left_buffer() { return left_buffer_; }
  RealView2D &right_buffer() { return right_buffer_; }

  size_t size() const { return left_buffer_.size(); }
  int_type left_rank() const { return left_rank_; }
  int_type right_rank() const { return right_rank_; }
  int_type left_tag() const { return left_tag_; }
  int_type right_tag() const { return right_tag_; }
  MPI_Comm communicator() const { return communicator_; }
  MPI_Datatype type() const { return mpi_data_type_; }
  bool is_comm() const {return is_comm_; }
};


struct Comm {
  using value_type = RealView3D::value_type;

private:
  // ID of the MPI process
  int rank_;
  
  // Number of MPI processes
  int size_;

  // Data shape
  std::vector<size_t> shape_;
 
  // MPI topology
  std::vector<int> topology_;
  std::vector<int> cart_rank_;

  // MPI Data type
  MPI_Datatype mpi_data_type_;

  // Halo data
  Halo *x_send_halo_, *x_recv_halo_;
  Halo *y_send_halo_, *y_recv_halo_;
  Halo *z_send_halo_, *z_recv_halo_;

  int halo_width_;

public:
  Comm() = delete;

  Comm(int &argc, char **argv, const std::vector<size_t> &shape, const std::vector<int> &topology)
    : shape_(shape), topology_(topology), halo_width_(1) {
    int required = MPI_THREAD_MULTIPLE;
    int provided;
    ::MPI_Init_thread(&argc, &argv, required, &provided);
    ::MPI_Comm_size(MPI_COMM_WORLD, &size_);
    ::MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

    setDevice( rank_ );

    mpi_data_type_ = get_mpi_data_type<value_type>();
 
    setTopology();
  }

  ~Comm() {}
  void cleanup() {
    delete x_send_halo_;
    delete y_send_halo_;
    delete z_send_halo_;
    delete x_recv_halo_;
    delete y_recv_halo_;
    delete z_recv_halo_;
  }
  void finalize() { ::MPI_Finalize(); }
  bool is_master() { return rank_==0; }
  int size() const { return size_; }
  int rank() const { return rank_; }
  int cart_rank(size_t i) const {
    assert( i < cart_rank_.size() );
    return cart_rank_.at(i);
  }
  std::vector<int> cart_rank() const {
    return cart_rank_;
  }
  int topology(size_t i) const {
    assert( i < topology_.size() );
    return topology_.at(i);
  }
  std::vector<int> topology() const {
    return topology_;
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

      if(use_timer) timers[HaloComm]->begin();
      commP2P(x_recv_halo_, x_send_halo_);
      if(use_timer) timers[HaloComm]->end();

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

      if(use_timer) timers[HaloComm]->begin();
      commP2P(y_recv_halo_, y_send_halo_);
      if(use_timer) timers[HaloComm]->end();

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

      if(use_timer) timers[HaloComm]->begin();
      commP2P(z_recv_halo_, z_send_halo_);
      if(use_timer) timers[HaloComm]->end();

      if(use_timer) timers[HaloUnpack]->begin();
      unpack(recv_left_z, recv_right_z, z_recv_halo_);
      if(use_timer) timers[HaloUnpack]->end();
    }
  }

private:
  void setTopology() {
    assert(topology_.size() == 3);
    int topology_size = std::accumulate(topology_.begin(), topology_.end(), 1, std::multiplies<int>());
    assert(topology_size == size_);

    // Create a Cartesian Communicator
    int ndims = 3;
    int periods[ndims] = {1, 1, 1}; // Periodic in all directions
    int reorder = 1;
    int old_rank = rank_;
    MPI_Comm cart_comm;

    ::MPI_Cart_create(MPI_COMM_WORLD, ndims, topology_.data(), periods, reorder, &cart_comm);
    ::MPI_Comm_rank(cart_comm, &rank_);

    if(rank_ != old_rank) {
      std::cout << "Rank change: from " << old_rank << " to " << rank_ << std::endl;
    }

    // Define new coordinate
    cart_rank_.resize(ndims); // (rankx, ranky, rankz)
    ::MPI_Cart_coords(cart_comm, rank_, ndims, cart_rank_.data());

    int neighbors[6];
    ::MPI_Cart_shift(cart_comm, 0, 1, &neighbors[UP],   &neighbors[DOWN]);   // x direction
    ::MPI_Cart_shift(cart_comm, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]);  // y direction
    ::MPI_Cart_shift(cart_comm, 2, 1, &neighbors[TOP],  &neighbors[BOTTOM]); // z direction

    bool is_comm_x = topology_.at(0) > 1;
    bool is_comm_y = topology_.at(1) > 1;
    bool is_comm_z = topology_.at(2) > 1;

    int left_tag = 0, right_tag = 1;
    x_send_halo_ = new Halo("x_send", {shape_[1], shape_[2]}, neighbors[UP],   neighbors[DOWN],  left_tag, right_tag, cart_comm, is_comm_x);
    x_recv_halo_ = new Halo("x_recv", {shape_[1], shape_[2]}, neighbors[UP],   neighbors[DOWN],  right_tag, left_tag, cart_comm, is_comm_x);

    y_send_halo_ = new Halo("y_send", {shape_[0], shape_[2]}, neighbors[LEFT], neighbors[RIGHT], left_tag, right_tag, cart_comm, is_comm_y);
    y_recv_halo_ = new Halo("y_recv", {shape_[0], shape_[2]}, neighbors[LEFT], neighbors[RIGHT], right_tag, left_tag, cart_comm, is_comm_y);

    z_send_halo_ = new Halo("z_send", {shape_[0], shape_[1]}, neighbors[TOP],  neighbors[BOTTOM], left_tag, right_tag, cart_comm, is_comm_z);
    z_recv_halo_ = new Halo("z_recv", {shape_[0], shape_[1]}, neighbors[TOP],  neighbors[BOTTOM], right_tag, left_tag, cart_comm, is_comm_z);

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

    int2 begin = make_int2(0, 0);
    int2 end = make_int2(left.extent(0), left.extent(1));

    Impl::for_each<default_iterate_layout>(begin, end, 
                                           [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix, const int iy) {
                                             left_buffer(ix, iy) = left(ix, iy);
                                             right_buffer(ix, iy) = right(ix, iy);
                                           });
    synchronize();
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

    int2 begin = make_int2(0, 0);
    int2 end = make_int2(left.extent(0), left.extent(1));

    Impl::for_each<default_iterate_layout>(begin, end, 
                                           [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix, const int iy) {
                                             left(ix, iy) = left_buffer(ix, iy);
                                             right(ix, iy) = right_buffer(ix, iy);
                                           });
    synchronize();
  }

  /* Send to dest, recv from dest 
   * ex. Nprocs = 4
   *           rank0  rank1  rank2  rank3
     SendLeft    3      0      1      2
     SendRight   1      2      3      0
     RecvLeft    3      0      1      2
     RecvRight   1      2      3      0
   * */

private:
  void commP2P(Halo *recv, const Halo *send) {
    if(send->is_comm()) {
      MPI_Status  status[4];
      MPI_Request request[4];
      MPI_Irecv(recv->left_buffer_.data(),  recv->size(), recv->type(), recv->left_rank(),  recv->left_tag(), recv->communicator(), &request[0]);
      MPI_Irecv(recv->right_buffer_.data(), recv->size(), recv->type(), recv->right_rank(), recv->right_tag(), recv->communicator(), &request[1]);
      MPI_Isend(send->left_buffer_.data(),  send->size(), send->type(), send->left_rank(),  send->left_tag(), send->communicator(), &request[2]);
      MPI_Isend(send->right_buffer_.data(), send->size(), send->type(), send->right_rank(), send->right_tag(), send->communicator(), &request[3]);

      MPI_Waitall( 4, request, status );

    } else {
      auto send_left_buffer  = send->left_buffer();
      auto send_right_buffer = send->right_buffer();
      recv->left_buffer().swap( send_right_buffer );
      recv->right_buffer().swap( send_left_buffer );
    }
  }

};

#endif

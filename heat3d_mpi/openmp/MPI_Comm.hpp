#ifndef __MPI_COMM_HPP__
#define __MPI_COMM_HPP__

#include <cassert>
#include <vector>
#include <complex>
#include <mpi.h>
#include "../Timer.hpp"
#include "Types.hpp"
#include "Config.hpp"
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
    #if defined( ENABLE_OPENMP_OFFLOAD )
      #pragma omp target enter data map(alloc: this[0:1])
    #endif

    left_buffer_  = RealView2D(name + "_left_buffer", shape[0], shape[1]);
    right_buffer_ = RealView2D(name + "_right_buffer", shape[0], shape[1]);
    left_buffer_.fill(); 
    right_buffer_.fill();
    assert(left_buffer_.size() == right_buffer_.size() );
    size_ = left_buffer_.size();
    mpi_data_type_ = get_mpi_data_type<value_type>();
  }
  
  ~Halo() {
    #if defined( ENABLE_OPENMP_OFFLOAD )
      #pragma omp target exit data map(delete: this[0:1])
    #endif
  }

  const std::string name() const noexcept {return name_;}
  size_t size() const { return left_buffer_.size(); }
  int_type left_rank() const { return left_rank_; }
  int_type right_rank() const { return right_rank_; }
  int_type left_tag() const { return left_tag_; }
  int_type right_tag() const { return right_tag_; }
  MPI_Comm communicator() const { return communicator_; }
  MPI_Datatype type() const { return mpi_data_type_; }
  bool is_comm() const {return is_comm_; }

private:
  DISALLOW_COPY_AND_ASSIGN(Halo);
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

    mpi_data_type_ = get_mpi_data_type<value_type>();

    #if defined( ENABLE_OPENMP_OFFLOAD )
      #pragma omp target enter data map(alloc: this[0:1])
    #endif

    setTopology();
  }

  ~Comm() {
    #if defined( ENABLE_OPENMP_OFFLOAD )
      #pragma omp target exit data map(delete: this[0:1])
    #endif
  }

  void cleanup() {
    #if defined(ENABLE_OPENMP_OFFLOAD)
     #pragma omp target exit data map(delete: x_send_halo_[0:1], x_recv_halo_[0:1])
     #pragma omp target exit data map(delete: y_send_halo_[0:1], y_recv_halo_[0:1])
     #pragma omp target exit data map(delete: z_send_halo_[0:1], z_recv_halo_[0:1])
    #endif
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

  void exchangeHalos(const Config &conf, RealView3D &u, std::vector<Timer*> &timers) {
    bool use_timer = timers.size() > 0;

    if(use_timer) timers[HaloPack]->begin();
    pack(conf, u);
    if(use_timer) timers[HaloPack]->end();

    if(use_timer) timers[HaloComm]->begin();
    commP2P(x_recv_halo_, x_send_halo_);
    commP2P(y_recv_halo_, y_send_halo_);
    commP2P(z_recv_halo_, z_send_halo_);
    if(use_timer) timers[HaloComm]->end();

    if(use_timer) timers[HaloUnpack]->begin();
    unpack(conf, u);
    if(use_timer) timers[HaloUnpack]->end();
  }

private:
  void setTopology() {
    assert(topology_.size() == 3);
    int topology_size = std::accumulate(topology_.begin(), topology_.end(), 1, std::multiplies<int>());
    assert(topology_size == size_);

    // Create a Cartesian Communicator
    constexpr int ndims = 3;
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
    ::MPI_Cart_shift(cart_comm, 0, 1, &neighbors[UP],   &neighbors[DOWN]);   // z direction
    ::MPI_Cart_shift(cart_comm, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]);  // y direction
    ::MPI_Cart_shift(cart_comm, 2, 1, &neighbors[TOP],  &neighbors[BOTTOM]); // x direction

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

    #if defined(ENABLE_OPENMP_OFFLOAD)
      #pragma omp target enter data map(alloc: x_send_halo_[0:1], x_recv_halo_[0:1])
      #pragma omp target enter data map(alloc: y_send_halo_[0:1], y_recv_halo_[0:1])
      #pragma omp target enter data map(alloc: z_send_halo_[0:1], z_recv_halo_[0:1])
    #endif
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

  /* Send to dest, recv from dest 
   * ex. Nprocs = 4
   *           rank0  rank1  rank2  rank3
     SendLeft    3      0      1      2
     SendRight   1      2      3      0
     RecvLeft    3      0      1      2
     RecvRight   1      2      3      0
   * */
  void commP2P(Halo *recv, Halo *send) {
    if(send->is_comm()) {
      MPI_Status  status[4];
      MPI_Request request[4];
      auto *recv_left_buffer  = recv->left_buffer_.data();
      auto *recv_right_buffer = recv->right_buffer_.data();
      auto *send_left_buffer  = send->left_buffer_.data();
      auto *send_right_buffer = send->right_buffer_.data();
      #if defined( ENABLE_OPENMP_OFFLOAD )
        #pragma omp target data use_device_ptr(recv_left_buffer, recv_right_buffer, send_left_buffer, send_right_buffer)
      #endif
      {
        MPI_Irecv(recv_left_buffer,  recv->size(), recv->type(), recv->left_rank(),  0, recv->communicator(), &request[0]);
        MPI_Irecv(recv_right_buffer, recv->size(), recv->type(), recv->right_rank(), 1, recv->communicator(), &request[1]);
        MPI_Isend(send_left_buffer,  send->size(), send->type(), send->left_rank(),  1, send->communicator(), &request[2]);
        MPI_Isend(send_right_buffer, send->size(), send->type(), send->right_rank(), 0, send->communicator(), &request[3]);
        MPI_Waitall( 4, request, status );
      }
    } else {
      recv->left_buffer_.swap(  send->right_buffer_ );
      recv->right_buffer_.swap( send->left_buffer_ );
    }
  }
};

#endif

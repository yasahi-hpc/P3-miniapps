#ifndef __COMM_HPP__
#define __COMM_HPP__

#include <vector>
#include <mpi.h>
#include <numeric>
#include <execution>
#include <algorithm>
#include <cassert>
#include <string>
#include "Types.hpp"
#include "Config.hpp"
#include "Parallel_For.hpp"
#include "../Index.hpp"
#include "../Timer.hpp"

static constexpr int VUNDEF = -100000;

// Element considered within the algorithm of the Unbalanced Recursive Bisection (URB)
struct Urbnode{
  // xmin and xmax set the interval of the box 
  // owned by a specific MPI process
  int xmin_[DIMENSION];
  int xmax_[DIMENSION];
  
  // Number of processes
  int nbp_;
  
  // ID of the MPI process
  int pid_;
};

// One single halo part stuck to the local
// This halo part should be a regular contiguous box without 
// hole inside. This halo part will be then bond to the box
// owned by the local MPI process.
struct Halo{
  int xmin_[DIMENSION];
  int xmax_[DIMENSION];
  int pid_; 
  int size_;
  int lxmin_[DIMENSION];
  int lxmax_[DIMENSION];
  float64 *buf_;
  int tag_;
};

// In Kokkos, we manage all the halos in a single data structure
struct Halos{
  using RangeView2D = View2D<int>;
  using RangeView3D = View3D<int>;
  RealView1D buf_flatten_;
  RangeView2D xmin_;
  RangeView2D xmax_;
  RangeView2D bc_in_min_;
  RangeView2D bc_in_max_;
  RangeView2D lxmin_;
  RangeView2D lxmax_;

  shape_nd<DIMENSION> nhalo_max_;
  int size_;     // buffer size of each halo
  int nb_halos_; // the number of halos
  int pid_;      // MPI rank
  int nbp_;      // the number of MPI processes

  std::vector<int> sizes_;
  std::vector<int> pids_;
  std::vector<int> tags_;

  /* Used for boundary condition */
  RangeView3D map_bc_;  // f -> flatten_buf (used for send buffers)
  RangeView2D map_orc_; // f -> flatten_buf (used for send buffers)
  RangeView3D sign1_;       
  RangeView3D sign2_;       
  RangeView3D sign3_;       
  RangeView3D sign4_;       
  IntView1D   orcsum_;      

  /* Used for merge */
  RangeView2D map_;         // f -> flatten_buf
  IntView2D   flatten_map_; // buf -> flatten_buf

  int offset_local_copy_;   // The head address for the local copy
  std::vector<int> merged_sizes_; // buffer size of each halo
  std::vector<int> merged_pids_; // merged process id to interact with
  std::vector<int> merged_tags_; //
  std::vector<int> id_map_; // mapping the original id to the merged id
  std::vector<int> merged_heads_;
  std::vector<int> total_size_orc_;
  int total_size_; // total size of all the buffers
  int nb_merged_halos_; // number of halos after merge
  int nb_nocomms_;

public:
  // Constructor and destructor
  Halos() {}
  ~Halos() {}

public:
  float64* head(const int i) {
    float64* dptr_buf = buf_flatten_.data() + merged_heads_.at(i);
    return dptr_buf;
  }

  int merged_size(const int i) {
    return merged_sizes_.at(i);
  }

  int merged_pid(const int i) {
    return merged_pids_.at(i);
  }

  int merged_tag(const int i) {
    return merged_tags_.at(i);
  }
     
  int nb_reqs(){ return (nbp_ - nb_nocomms_); }
  int nb_halos(){ return nb_halos_; } 

  void listBoundary(Config *conf, const std::string name, IntView2D &map_2D2flatten) {
    const Domain *dom = &(conf->dom_);
    int nx = dom->nxmax_[0], ny = dom->nxmax_[1], nvx = dom->nxmax_[2], nvy = dom->nxmax_[3];
    const int nx_min  = dom->local_nxmin_[0] - HALO_PTS;
    const int ny_min  = dom->local_nxmin_[1] - HALO_PTS;
    const int nvx_min = dom->local_nxmin_[2] - HALO_PTS;
    const int nvy_min = dom->local_nxmin_[3] - HALO_PTS;
    const int nx_max  = dom->local_nxmax_[0] + 1 + HALO_PTS;
    const int ny_max  = dom->local_nxmax_[1] + 1 + HALO_PTS;
    const int nvx_max = dom->local_nxmax_[2] + 1 + HALO_PTS;
    const int nvy_max = dom->local_nxmax_[3] + 1 + HALO_PTS;
    int bc_sign[8];
    for(int k = 0; k < DIMENSION; k++) {
      bc_sign[2 * k + 0] = -1;
      bc_sign[2 * k + 1] = 1;
    }

    sign1_   = RangeView3D(name + "_sign1",   total_size_, DIMENSION, DIMENSION);
    sign2_   = RangeView3D(name + "_sign2",   total_size_, DIMENSION, DIMENSION);
    sign3_   = RangeView3D(name + "_sign3",   total_size_, DIMENSION, DIMENSION);
    sign4_   = RangeView3D(name + "_sign4",   total_size_, DIMENSION, DIMENSION);
    map_orc_ = RangeView2D(name + "_map_orc", total_size_, DIMENSION);
    map_bc_  = RangeView3D(name + "_map_bc",  total_size_, DIMENSION, DIMENSION);
    map_orc_.fill(0); map_bc_.fill(0);
    sign1_.fill(0); sign2_.fill(0); sign3_.fill(0); sign4_.fill(0);

    int count0 = 0, count1 = 0, count2 = 0, count3 = 0;
    for(int ib = 0; ib < nb_halos_; ib++) {
      int halo_min[4], halo_max[4];
      for(int k = 0; k < DIMENSION; k++) {
        halo_min[k] = xmin_(ib, k);
        halo_max[k] = xmax_(ib, k);
      }
       
      const int halo_nx =  halo_max[0] - halo_min[0] + 1;
      const int halo_ny  = halo_max[1] - halo_min[1] + 1;
      const int halo_nvx = halo_max[2] - halo_min[2] + 1;
      const int halo_nvy = halo_max[3] - halo_min[3] + 1;

      int bc_in[8], orcheck[4];
      int orcsum = 0;
      char bitconf = 0;

      for(int k = 0; k < 4; k++) {
        bc_in[2 * k + 0] = bc_in_min_(ib, k);
        bc_in[2 * k + 1] = bc_in_max_(ib, k);
        orcheck[k] = (bc_in[2 * k] != VUNDEF) || (bc_in[2 * k + 1] != VUNDEF);
        orcsum += orcheck[k];
        bitconf |= 1 << k;
      }

      int sign1[4], sign2[4], sign3[4], sign4[4];
      for(int k1 = 0; k1 < 8; k1++) {
        if(bc_in[k1] != VUNDEF) {
          int vdx[4], vex[4];
          for(int ii = 0; ii < 4; ii++) {
            sign1[ii] = 0, vdx[ii] = halo_min[ii], vex[ii] = halo_max[ii];
            if(ii == k1/2)
              sign1[ii] = bc_sign[k1], vdx[ii] = bc_in[k1], vex[ii] = bc_in[k1];
          }

          for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
            for(int jvx = vdx[2]; jvx <= vex[2]; jvx++) {
              for(int jy = vdx[1]; jy <= vex[1]; jy++) {
                for(int jx = vdx[0]; jx <= vex[0]; jx++) {
                  int rx  = (nx  + jx  - sign1[0]) % nx;
                  int ry  = (ny  + jy  - sign1[1]) % ny;
                  int rvx = (nvx + jvx - sign1[2]) % nvx;
                  int rvy = (nvy + jvy - sign1[3]) % nvy;
                  for(int ii = 0; ii < 4; ii++) {
                    sign1_(count0, 0, ii) = sign1[ii];
                  }
                  map_bc_(count0, 0, 0) = rx;
                  map_bc_(count0, 0, 1) = ry;
                  map_bc_(count0, 0, 2) = rvx;
                  map_bc_(count0, 0, 3) = rvy;
                  int idx = Index::coord_4D2int(jx  - halo_min[0],
                                                jy  - halo_min[1],
                                                jvx - halo_min[2],
                                                jvy - halo_min[3],
                                                halo_nx, halo_ny, halo_nvx, halo_nvy);
                  map_orc_(count0, 0) = map_2D2flatten(idx, ib); //index in 2D buffer 
                  assert(map_2D2flatten(idx, ib) < total_size_);

                  for(int j1 = 1; j1 <= MMAX; j1++) {
                    assert( (rx  + sign1_(count0, 0, 0) * j1) >= nx_min  );
                    assert( (ry  + sign1_(count0, 0, 1) * j1) >= ny_min  );
                    assert( (rvx + sign1_(count0, 0, 2) * j1) >= nvx_min );
                    assert( (rvy + sign1_(count0, 0, 3) * j1) >= nvy_min );
                    
                    assert( (rx  + sign1_(count0, 0, 0) * j1) < nx_max  );
                    assert( (ry  + sign1_(count0, 0, 1) * j1) < ny_max  );
                    assert( (rvx + sign1_(count0, 0, 2) * j1) < nvx_max );
                    assert( (rvy + sign1_(count0, 0, 3) * j1) < nvy_max );
                  }
                  count0++;
                }
              }
            }
          }
        }
      }

      if(orcsum > 1) {
        for(int k1 = 0; k1 < 8; k1++) {
          for(int k2 = 2 * (1 + k1/2); k2 < 8; k2++) {
            if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF) {
              int vdx[4], vex[4];
              for(int ii = 0; ii < 4; ii++) {
                sign1[ii] = 0, sign2[ii] = 0, vdx[ii] = halo_min[ii], vex[ii] = halo_max[ii];
                if(ii == k1/2)
                  sign1[ii] = bc_sign[k1], vex[ii] = vdx[ii] = bc_in[k1];
                if(ii == k2/2)
                  sign2[ii] = bc_sign[k2], vex[ii] = vdx[ii] = bc_in[k2];
              }

              for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
                for(int jvx = vdx[2]; jvx <= vex[2]; jvx++) {
                  for(int jy = vdx[1]; jy <= vex[1]; jy++) {
                    for(int jx = vdx[0]; jx <= vex[0]; jx++) {
                      int rx  = (nx  + jx  - sign1[0] - sign2[0]) % nx;
                      int ry  = (ny  + jy  - sign1[1] - sign2[1]) % ny;
                      int rvx = (nvx + jvx - sign1[2] - sign2[2]) % nvx;
                      int rvy = (nvy + jvy - sign1[3] - sign2[3]) % nvy;
                      for(int ii = 0; ii < 4; ii++) {
                        sign1_(count1, 1, ii) = sign1[ii];
                        sign2_(count1, 1, ii) = sign2[ii];
                      }
                      map_bc_(count1, 1, 0) = rx;
                      map_bc_(count1, 1, 1) = ry;
                      map_bc_(count1, 1, 2) = rvx;
                      map_bc_(count1, 1, 3) = rvy;
                      int idx = Index::coord_4D2int(jx  - halo_min[0],
                                                    jy  - halo_min[1],
                                                    jvx - halo_min[2],
                                                    jvy - halo_min[3],
                                                    halo_nx, halo_ny, halo_nvx, halo_nvy);
                                                                   
                      map_orc_(count1, 1) = map_2D2flatten(idx, ib); //index in 2D buffer 
                      assert(map_2D2flatten(idx, ib) < total_size_);
                      for(int j2 = 1; j2 <= MMAX; j2++) {
                        for(int j1 = 1; j1 <= MMAX; j1++) {
                          assert( (rx  + sign1_(count1, 1, 0) * j1 + sign2_(count1, 1, 0) * j2) >= nx_min );
                          assert( (ry  + sign1_(count1, 1, 1) * j1 + sign2_(count1, 1, 1) * j2) >= ny_min );
                          assert( (rvx + sign1_(count1, 1, 2) * j1 + sign2_(count1, 1, 2) * j2) >= nvx_min );
                          assert( (rvy + sign1_(count1, 1, 3) * j1 + sign2_(count1, 1, 3) * j2) >= nvy_min );
                        
                          assert( (rx  + sign1_(count1, 1, 0) * j1 + sign2_(count1, 1, 0) * j2) <  nx_max );
                          assert( (ry  + sign1_(count1, 1, 1) * j1 + sign2_(count1, 1, 1) * j2) <  ny_max );
                          assert( (rvx + sign1_(count1, 1, 2) * j1 + sign2_(count1, 1, 2) * j2) <  nvx_max );
                          assert( (rvy + sign1_(count1, 1, 3) * j1 + sign2_(count1, 1, 3) * j2) <  nvy_max );
                        }
                      }
                      count1++;
                    }
                  }
                }
              }
            } // if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF)
          } // for(int k2 = 2 * (1 + k1/2); k2 < 8; k2++)
        } // for(int k1 = 0; k1 < 8; k1++)
      } // if(orcsum > 1)

      if(orcsum > 2) {
        for(int k1 = 0; k1 < 8; k1++) {
          for(int k2 = 2 * (1 + k1/2); k2 < 8; k2++) {
            for(int k3 = 2 * (1 + k2/2); k3 < 8; k3++) {
              if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF && bc_in[k3] != VUNDEF) {
                int vdx[4], vex[4];
                for(int ii = 0; ii < 4; ii++) {
                  sign1[ii] = 0, sign2[ii] = 0, sign3[ii] = 0, vdx[ii] = halo_min[ii], vex[ii] = halo_max[ii];
                  if(ii == k1/2)
                    sign1[ii] = bc_sign[k1], vex[ii] = vdx[ii] = bc_in[k1];
                  if(ii == k2/2)
                    sign2[ii] = bc_sign[k2], vex[ii] = vdx[ii] = bc_in[k2];
                  if(ii == k3/2)
                    sign3[ii] = bc_sign[k3], vex[ii] = vdx[ii] = bc_in[k3];
                }

                for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
                  for(int jvx = vdx[2]; jvx <= vex[2]; jvx++) {
                    for(int jy = vdx[1]; jy <= vex[1]; jy++) {
                      for(int jx = vdx[0]; jx <= vex[0]; jx++) {
                        int rx  = (nx  + jx  - sign1[0] - sign2[0] - sign3[0]) % nx;
                        int ry  = (ny  + jy  - sign1[1] - sign2[1] - sign3[1]) % ny;
                        int rvx = (nvx + jvx - sign1[2] - sign2[2] - sign3[2]) % nvx;
                        int rvy = (nvy + jvy - sign1[3] - sign2[3] - sign3[3]) % nvy;
                        for(int ii = 0; ii < 4; ii++) {
                          sign1_(count2, 2, ii) = sign1[ii];
                          sign2_(count2, 2, ii) = sign2[ii];
                          sign3_(count2, 2, ii) = sign3[ii];
                        }
                        map_bc_(count2, 2, 0) = rx;
                        map_bc_(count2, 2, 1) = ry;
                        map_bc_(count2, 2, 2) = rvx;
                        map_bc_(count2, 2, 3) = rvy;
                        int idx = Index::coord_4D2int(jx  - halo_min[0],
                                                      jy  - halo_min[1],
                                                      jvx - halo_min[2],
                                                      jvy - halo_min[3],
                                                      halo_nx, halo_ny, halo_nvx, halo_nvy);

                        map_orc_(count2, 2) = map_2D2flatten(idx, ib); //index in 2D buffer 
                        assert(map_2D2flatten(idx, ib) < total_size_);
                        for(int j3 = 1; j3 <= MMAX; j3++) {
                          for(int j2 = 1; j2 <= MMAX; j2++) {
                            for(int j1 = 1; j1 <= MMAX; j1++) {
                              assert( (rx  + sign1_(count2, 2, 0) * j1 + sign2_(count2, 2, 0) * j2 + sign3_(count2, 2, 0) * j3) >= nx_min );
                              assert( (ry  + sign1_(count2, 2, 1) * j1 + sign2_(count2, 2, 1) * j2 + sign3_(count2, 2, 1) * j3) >= ny_min );
                              assert( (rvx + sign1_(count2, 2, 2) * j1 + sign2_(count2, 2, 2) * j2 + sign3_(count2, 2, 2) * j3) >= nvx_min );
                              assert( (rvy + sign1_(count2, 2, 3) * j1 + sign2_(count2, 2, 3) * j2 + sign3_(count2, 2, 3) * j3) >= nvy_min );
                                              
                              assert( (rx  + sign1_(count2, 2, 0) * j1 + sign2_(count2, 2, 0) * j2 + sign3_(count2, 2, 0) * j3) <  nx_max );
                              assert( (ry  + sign1_(count2, 2, 1) * j1 + sign2_(count2, 2, 1) * j2 + sign3_(count2, 2, 1) * j3) <  ny_max );
                              assert( (rvx + sign1_(count2, 2, 2) * j1 + sign2_(count2, 2, 2) * j2 + sign3_(count2, 2, 2) * j3) <  nvx_max );
                              assert( (rvy + sign1_(count2, 2, 3) * j1 + sign2_(count2, 2, 3) * j2 + sign3_(count2, 2, 3) * j3) <  nvy_max );
                            }
                          }
                        }
                        count2++;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      if(orcsum > 3) {
        for(int k1 = 0; k1 < 2; k1++) {
          for(int k2 = 2; k2 < 4; k2++) {
            for(int k3 = 4; k3 < 6; k3++) {
              for(int k4 = 6; k4 < 8; k4++) {
                if(bc_in[k1] != VUNDEF && bc_in[k2] != VUNDEF && bc_in[k3] != VUNDEF && bc_in[k4] != VUNDEF) {
                  int vdx[4], vex[4];
                  for(int ii = 0; ii < 4; ii++) {
                    sign1[ii] = 0, sign2[ii] = 0, sign3[ii] = 0, sign4[ii] = 0, vdx[ii] = halo_min[ii], vex[ii] = halo_max[ii];
                    if(ii == k1/2)
                      sign1[ii] = bc_sign[k1], vex[ii] = vdx[ii] = bc_in[k1];
                    if(ii == k2/2)
                      sign2[ii] = bc_sign[k2], vex[ii] = vdx[ii] = bc_in[k2];
                    if(ii == k3/2)
                      sign3[ii] = bc_sign[k3], vex[ii] = vdx[ii] = bc_in[k3];
                    if(ii == k4/2)
                      sign4[ii] = bc_sign[k4], vex[ii] = vdx[ii] = bc_in[k4];
                  }

                  for(int jvy = vdx[3]; jvy <= vex[3]; jvy++) {
                    for(int jvx = vdx[2]; jvx <= vex[2]; jvx++) {
                      for(int jy = vdx[1]; jy <= vex[1]; jy++) {
                        for(int jx = vdx[0]; jx <= vex[0]; jx++) {
                          int rx  = (nx  + jx  - sign1[0]) % nx;
                          int ry  = (ny  + jy  - sign2[1]) % ny;
                          int rvx = (nvx + jvx - sign3[2]) % nvx;
                          int rvy = (nvy + jvy - sign4[3]) % nvy;
                          for(int ii = 0; ii < 4; ii++) {
                            sign1_(count3, 3, ii) = sign1[ii];
                            sign2_(count3, 3, ii) = sign2[ii];
                            sign3_(count3, 3, ii) = sign3[ii];
                            sign4_(count3, 3, ii) = sign4[ii];
                          }
                          map_bc_(count3, 3, 0) = rx;
                          map_bc_(count3, 3, 1) = ry;
                          map_bc_(count3, 3, 2) = rvx;
                          map_bc_(count3, 3, 3) = rvy;
                          int idx = Index::coord_4D2int(jx  - halo_min[0],
                                                        jy  - halo_min[1],
                                                        jvx - halo_min[2],
                                                        jvy - halo_min[3],
                                                        halo_nx, halo_ny, halo_nvx, halo_nvy);
                          map_orc_(count3, 3) = map_2D2flatten(idx, ib); //index in 2D buffer 
                          assert(map_2D2flatten(idx, ib) < total_size_);
                           
                          for(int j4 = 1; j4 <= MMAX; j4++) {
                            for(int j3 = 1; j3 <= MMAX; j3++) {
                              for(int j2 = 1; j2 <= MMAX; j2++) {
                                for(int j1 = 1; j1 <= MMAX; j1++) {
                                  assert( (rx  + sign1_(count3, 3, 0) * j1) >= nx_min );
                                  assert( (ry  + sign2_(count3, 3, 1) * j2) >= ny_min );
                                  assert( (rvx + sign3_(count3, 3, 2) * j3) >= nvx_min );
                                  assert( (rvy + sign4_(count3, 3, 3) * j4) >= nvy_min );
                                  
                                  assert( (rx  + sign1_(count3, 3, 0) * j1) <  nx_max );
                                  assert( (ry  + sign2_(count3, 3, 1) * j2) <  ny_max );
                                  assert( (rvx + sign3_(count3, 3, 2) * j3) <  nvx_max );
                                  assert( (rvy + sign4_(count3, 3, 3) * j4) <  nvy_max );
                                }
                              }
                            }
                          }
                          count3++;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    } // for(int ib = 0; ib < nb_halos_; ib++)

    total_size_orc_.at(0) = count0;
    total_size_orc_.at(1) = count1;
    total_size_orc_.at(2) = count2;
    total_size_orc_.at(3) = count3;
    
    map_bc_.updateDevice();
    map_orc_.updateDevice();
    sign1_.updateDevice();
    sign2_.updateDevice();
    sign3_.updateDevice();
    sign4_.updateDevice();
  }

  void mergeLists(Config *conf, const std::string name, IntView2D &map_2D2flatten) {
    nb_nocomms_ = 1; // communication with same pid is a local copy
    int total_size = 0;
    std::vector<int> id_map( nb_halos_ );
    std::vector< std::vector<int> > group_same_dst;

    for(int pid = 0; pid < nbp_; pid++) {
      if(pid == pid_) {
        offset_local_copy_ = total_size;
      }
      std::vector<int> same_dst;
      for(auto it = pids_.begin(); it != pids_.end(); it++) {
        if(*it == pid) {
          int dst_id = std::distance(pids_.begin(), it);
          same_dst.push_back( std::distance(pids_.begin(), it) );
        }
      }

      // Save merged data for current pid
      int size = 0;
      int tag  = 0;

      if(same_dst.empty()) {
        if(pid != pid_) nb_nocomms_++; // In case there is no connection between pid and pid_
      } else {
        tag = tags_[same_dst[0]]; // Use the tag of first element
      }
      for(auto it: same_dst) {
        id_map.at(it) = total_size + size;
        size += sizes_[it];
      }
                                               
      merged_sizes_.push_back(size);
      merged_pids_.push_back(pid);
      merged_tags_.push_back(tag);
      total_size += size; // Size of the total size summed up for previous pids
      group_same_dst.push_back(same_dst);
    }

    total_size_  = total_size;
    map_         = RangeView2D("map", total_size, DIMENSION); // This is used for receive buffer
    buf_flatten_ = RealView1D(name + "_buf_flat", total_size);
    flatten_map_ = IntView2D("flatten_map", total_size, 2); // storing (idx_in_buf, buf_id)

    const Domain *dom = &(conf->dom_);
    int nx_max  = dom->nxmax_[0];
    int ny_max  = dom->nxmax_[1];
    int nvx_max = dom->nxmax_[2];
    int nvy_max = dom->nxmax_[3];
                         
    int local_xstart  = dom->local_nxmin_[0];
    int local_ystart  = dom->local_nxmin_[1];
    int local_vxstart = dom->local_nxmin_[2];
    int local_vystart = dom->local_nxmin_[3];

    int idx_flatten = 0;
    for(auto same_dst: group_same_dst) {
      // Keeping the head index of each halo sets for MPI communication
      merged_heads_.push_back(idx_flatten);
      for(auto it: same_dst) {
        const int ix_min  = xmin_(it, 0), ix_max  = xmax_(it, 0);
        const int iy_min  = xmin_(it, 1), iy_max  = xmax_(it, 1);
        const int ivx_min = xmin_(it, 2), ivx_max = xmax_(it, 2);
        const int ivy_min = xmin_(it, 3), ivy_max = xmax_(it, 3);
                                                                   
        const int nx  = ix_max  - ix_min  + 1;
        const int ny  = iy_max  - iy_min  + 1;
        const int nvx = ivx_max - ivx_min + 1;
        const int nvy = ivy_max - ivy_min + 1;
                                                                                                           
        for(int ivy = ivy_min; ivy <= ivy_max; ivy++) {
          for(int ivx = ivx_min; ivx <= ivx_max; ivx++) {
            for(int iy = iy_min; iy <= iy_max; iy++) {
              for(int ix = ix_min; ix <= ix_max; ix++) {
                int idx = Index::coord_4D2int(ix-ix_min,
                                              iy-iy_min,
                                              ivx-ivx_min,
                                              ivy-ivy_min,
                                              nx, ny, nvx, nvy);

                // Map is different for send/recv buffers
                if(name == "send") {
                  const int ix_bc  = (nx_max  + ix)  % nx_max;
                  const int iy_bc  = (ny_max  + iy)  % ny_max;
                  const int ivx_bc = (nvx_max + ivx) % nvx_max;
                  const int ivy_bc = (nvy_max + ivy) % nvy_max;
                  map_(idx_flatten, 0) = ix_bc;
                  map_(idx_flatten, 1) = iy_bc;
                  map_(idx_flatten, 2) = ivx_bc;
                  map_(idx_flatten, 3) = ivy_bc;
                } else {
                  map_(idx_flatten, 0) = ix;
                  map_(idx_flatten, 1) = iy;
                  map_(idx_flatten, 2) = ivx;
                  map_(idx_flatten, 3) = ivy;
                }
                                                                                                 
                // h_flatten_map is used for send buffer
                flatten_map_(idx_flatten, 0) = idx;
                flatten_map_(idx_flatten, 1) = it;
                map_2D2flatten(idx, it) = idx_flatten;
                idx_flatten++;
              }
            }
          }
        }
      }
    }

    // Deep copy
    map_.updateDevice();
    flatten_map_.updateDevice();
  }

  void set(Config *conf, std::vector<Halo> &list, const std::string name, const int nb_process, const int pid) {
    nbp_ = nb_process;
    pid_ = pid;
    nb_merged_halos_ = nbp_;

    nb_halos_ = list.size();
    pids_.resize(nb_halos_);
    tags_.resize(nb_halos_);
    total_size_orc_.resize(DIMENSION);

    xmin_      = RangeView2D("halo_xmin",  nb_halos_, DIMENSION);
    xmax_      = RangeView2D("halo_xmax",  nb_halos_, DIMENSION);
    lxmin_     = RangeView2D("halo_lxmin", nb_halos_, DIMENSION);
    lxmax_     = RangeView2D("halo_lxmax", nb_halos_, DIMENSION);
    bc_in_min_ = RangeView2D("bc_in_min",  nb_halos_, DIMENSION);
    bc_in_max_ = RangeView2D("bc_in_max",  nb_halos_, DIMENSION);

    std::vector<int> nx_halos, ny_halos, nvx_halos, nvy_halos;
    for(size_t i = 0; i < nb_halos_; i++) {
      Halo *halo = &(list[i]);
      int tmp_size = (halo->xmax_[0] - halo->xmin_[0] + 1) * (halo->xmax_[1] - halo->xmin_[1] + 1)
                   * (halo->xmax_[2] - halo->xmin_[2] + 1) * (halo->xmax_[3] - halo->xmin_[3] + 1);
      sizes_.push_back(tmp_size);
      nx_halos.push_back(halo->xmax_[0] - halo->xmin_[0] + 1);
      ny_halos.push_back(halo->xmax_[1] - halo->xmin_[1] + 1);
      nvx_halos.push_back(halo->xmax_[2] - halo->xmin_[2] + 1);
      nvy_halos.push_back(halo->xmax_[3] - halo->xmin_[3] + 1);
      
      pids_[i] = halo->pid_;
      tags_[i] = halo->tag_;

      for(int j = 0; j < DIMENSION; j++) {
        xmin_(i, j)  = halo->xmin_[j]; 
        xmax_(i, j)  = halo->xmax_[j];
        lxmin_(i, j) = halo->lxmin_[j]; 
        lxmax_(i, j) = halo->lxmax_[j]; 
        int lxmin = lxmin_(i, j) - HALO_PTS, lxmax = lxmax_(i, j) + HALO_PTS;
        bc_in_min_(i, j) = (xmin_(i, j) <= lxmin && lxmin <= xmax_(i, j)) ? lxmin : VUNDEF;
        bc_in_max_(i, j) = (xmin_(i, j) <= lxmax && lxmax <= xmax_(i, j)) ? lxmax : VUNDEF;
      }
    }

    xmin_.updateDevice();
    xmax_.updateDevice();
    bc_in_min_.updateDevice();
    bc_in_max_.updateDevice();

    // Prepare large enough buffer
    auto max_size = std::max_element(sizes_.begin(), sizes_.end());
    size_ = *max_size;
    
    nhalo_max_[0] = *std::max_element(nx_halos.begin(),  nx_halos.end());
    nhalo_max_[1] = *std::max_element(ny_halos.begin(),  ny_halos.end());
    nhalo_max_[2] = *std::max_element(nvx_halos.begin(), nvx_halos.end());
    nhalo_max_[3] = *std::max_element(nvy_halos.begin(), nvy_halos.end());
    
    IntView2D map_2D2flatten(name + "_map_2D2flatten", size_, nb_halos_);
    mergeLists(conf, name, map_2D2flatten);
    if(name == "send") {
      // 2D Buffer is used only for send buffer [TO DO] delete this
      listBoundary(conf, name, map_2D2flatten);
    }
  }
};

struct Distrib {
private:
  // Pseudo variable
  int NB_DIMS_DECOMPOSITION = 4;

  // ID of the MPI process
  int pid_;

  // Number of MPI processes
  int nbp_;

  // List of boxes representing all local MPI domains
  std::vector<Urbnode> ulist_;

  // List of halo buffers (receiving side)
  std::vector<Halo> recv_list_;
  Halos *recv_buffers_;

  // List of halo buffers (sending side)
  std::vector<Halo> send_list_;
  Halos *send_buffers_; // May be better to use pointer for explicit deallocation

  // List of halo buffers (sending side)
  Urbnode *node_;

  // Use spline or not
  bool spline_;

  // Global domain size
  int nxmax_[DIMENSION];

public:
  Distrib() = delete;
  Distrib(int &argc, char **argv) : spline_(true), recv_buffers_(nullptr), send_buffers_(nullptr) {
    int required = MPI_THREAD_SERIALIZED;
    int provided;

    // Initialize MPI
    MPI_Init_thread(&argc, &argv, required, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &nbp_);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid_);
  };

  ~Distrib(){}

  // Deallocate pointer inside the Kokkos parallel region
  void cleanup() {
    delete recv_buffers_;
    delete send_buffers_;
  }

  void finalize() {
    MPI_Finalize();
  }

  // Getters
  bool master() {return pid_ == 0;}
  int pid(){return pid_;}
  int rank(){return pid_;}
  int nbp(){return nbp_;}
  Urbnode *node(){return node_;}
  std::vector<Urbnode> &nodes(){return ulist_;}

  // Initializers
  void createDecomposition(Config *conf);
  void neighboursList(Config *conf, RealView4D &halo_fn);
  void bookHalo(Config *conf);

  // Communication
  void exchangeHalo(Config *conf, RealView4D &halo_fn, std::vector<Timer*> &timers);

private:
  void getNeighbours(const Config *conf, const RealView4D &halo_fn, int xrange[8],
                     std::vector<Halo> &hlist, int lxmin[4], int lxmax[4], int count);

  void packAndBoundary(Config *conf, RealView4D &halo_fn, Halos *send_buffers);

  void pack(RealView4D &halo_fn, Halos *send_buffers) {
    const int total_size = send_buffers->total_size_;
    auto _halo_fn = halo_fn.mdspan();
    auto _map     = send_buffers->map_.mdspan();
    auto _buf_flatten = send_buffers->buf_flatten_.mdspan();

    Iterate_policy<1> policy1d(total_size);
    Impl::for_each(policy1d, 
                   [=](const int idx){
                     const int ix  = _map(idx, 0), iy  = _map(idx, 1);
                     const int ivx = _map(idx, 2), ivy = _map(idx, 3);
                     _buf_flatten(idx) = _halo_fn(ix, iy, ivx, ivy);
                   });
  }

  void unpack(RealView4D &halo_fn, Halos *recv_buffers) {
    const int total_size = recv_buffers->total_size_;
    auto _halo_fn = halo_fn.mdspan();
    auto _map = recv_buffers->map_.mdspan();
    auto _buf_flatten = recv_buffers->buf_flatten_.mdspan();

    Iterate_policy<1> policy1d(total_size);
    Impl::for_each(policy1d, 
                   [=](const int idx){
                     const int ix  = _map(idx, 0), iy  = _map(idx, 1);
                     const int ivx = _map(idx, 2), ivy = _map(idx, 3);
                     _halo_fn(ix, iy, ivx, ivy) =  _buf_flatten(idx);
                   });
  };

  void local_copy(Halos *send_buffers, Halos *recv_buffers) {
    const int send_offset = send_buffers->offset_local_copy_;
    const int recv_offset = recv_buffers->offset_local_copy_;
    const int total_size  = send_buffers->merged_size(pid_);

    auto _send_buf_flatten = send_buffers->buf_flatten_.mdspan();
    auto _recv_buf_flatten = recv_buffers->buf_flatten_.mdspan();

    Iterate_policy<1> policy1d(total_size);
    Impl::for_each(policy1d, 
                   [=](const int idx) {
                     _recv_buf_flatten(idx + recv_offset) = _send_buf_flatten(idx + send_offset);
                   });
  }

  // Version to survive
  void boundary_condition(RealView4D &halo_fn, Halos *send_buffers) {
    float64 alpha = sqrt(3) - 2;
    int orcsum = 0;
    const int total_size0 = send_buffers->total_size_orc_.at(orcsum);

    auto _halo_fn = halo_fn.mdspan();
    auto _map_bc = send_buffers->map_bc_.mdspan();
    auto _map_orc = send_buffers->map_orc_.mdspan();
    auto _sign1 = send_buffers->sign1_.mdspan();
    auto _sign2 = send_buffers->sign2_.mdspan();
    auto _sign3 = send_buffers->sign3_.mdspan();
    auto _sign4 = send_buffers->sign4_.mdspan();
    auto _buf_flatten = send_buffers->buf_flatten_.mdspan();

    if(total_size0 > 0) {
      Iterate_policy<1> policy1d(total_size0);
      Impl::for_each(policy1d, 
                     [=](const int idx) {
                       const int rx  = _map_bc(idx, orcsum, 0), ry  = _map_bc(idx, orcsum, 1);
                       const int rvx = _map_bc(idx, orcsum, 2), rvy = _map_bc(idx, orcsum, 3);
                       const int idx_dst = _map_orc(idx, orcsum);
                       const int sign10 = _sign1(idx, orcsum, 0);
                       const int sign11 = _sign1(idx, orcsum, 1);
                       const int sign12 = _sign1(idx, orcsum, 2);
                       const int sign13 = _sign1(idx, orcsum, 3);
                       float64 fsum = 0.;
                       float64 alphap1 = alpha;
                       for(int j1 = 1; j1 <= MMAX; j1++) {
                         fsum += _halo_fn(rx  + sign10 * j1,
                                          ry  + sign11 * j1,
                                          rvx + sign12 * j1,
                                          rvy + sign13 * j1
                                         ) * alphap1;
                         alphap1 *= alpha;
                       }
                       _buf_flatten(idx_dst) = fsum;
                     });
    }

    orcsum = 1;
    const int total_size1 = send_buffers->total_size_orc_.at(orcsum);
    if(total_size1 > 0) {
      Iterate_policy<1> policy1d(total_size1);
      Impl::for_each(policy1d,
                     [=](const int idx) {
                       const int idx_dst = _map_orc(idx, orcsum);
                       const int rx  = _map_bc(idx, orcsum, 0), ry  = _map_bc(idx, orcsum, 1);
                       const int rvx = _map_bc(idx, orcsum, 2), rvy = _map_bc(idx, orcsum, 3);
                       const int sign10 = _sign1(idx, orcsum, 0), sign20 = _sign2(idx, orcsum, 0);
                       const int sign11 = _sign1(idx, orcsum, 1), sign21 = _sign2(idx, orcsum, 1);
                       const int sign12 = _sign1(idx, orcsum, 2), sign22 = _sign2(idx, orcsum, 2);
                       const int sign13 = _sign1(idx, orcsum, 3), sign23 = _sign2(idx, orcsum, 3);
                       float64 fsum = 0.;
                       float64 alphap2 = alpha;
                       for(int j2 = 1; j2 <= MMAX; j2++) {
                         float64 alphap1 = alpha * alphap2;
                         for(int j1 = 1; j1 <= MMAX; j1++) {
                           fsum += _halo_fn(rx  + sign10 * j1 + sign20 * j2,
                                            ry  + sign11 * j1 + sign21 * j2,
                                            rvx + sign12 * j1 + sign22 * j2,
                                            rvy + sign13 * j1 + sign23 * j2) * alphap1;
                           alphap1 *= alpha;
                         }
                         alphap2 *= alpha;
                       }
                       _buf_flatten(idx_dst) = fsum;
                     });
    }

    orcsum = 2;
    const int total_size2 = send_buffers->total_size_orc_.at(orcsum);
    if(total_size2 > 0) {
      Iterate_policy<1> policy1d(total_size2);
      Impl::for_each(policy1d,
                     [=](const int idx) {
                       const int idx_dst = _map_orc(idx, orcsum);
                       const int rx  = _map_bc(idx, orcsum, 0), ry  = _map_bc(idx, orcsum, 1);
                       const int rvx = _map_bc(idx, orcsum, 2), rvy = _map_bc(idx, orcsum, 3);
                       const int sign10 = _sign1(idx, orcsum, 0), sign20 = _sign2(idx, orcsum, 0), sign30 = _sign3(idx, orcsum, 0);
                       const int sign11 = _sign1(idx, orcsum, 1), sign21 = _sign2(idx, orcsum, 1), sign31 = _sign3(idx, orcsum, 1);
                       const int sign12 = _sign1(idx, orcsum, 2), sign22 = _sign2(idx, orcsum, 2), sign32 = _sign3(idx, orcsum, 2);
                       const int sign13 = _sign1(idx, orcsum, 3), sign23 = _sign2(idx, orcsum, 3), sign33 = _sign3(idx, orcsum, 3);
                       float64 fsum = 0.;
                       float64 alphap3 = alpha;
                       for(int j3 = 1; j3 <= MMAX; j3++) {
                         float64 alphap2 = alpha * alphap3;
                         for(int j2 = 1; j2 <= MMAX; j2++) {
                           float64 alphap1 = alpha * alphap2;
                           for(int j1 = 1; j1 <= MMAX; j1++) {
                             fsum += _halo_fn(rx  + sign10 * j1 + sign20 * j2 + sign30 * j3,
                                              ry  + sign11 * j1 + sign21 * j2 + sign31 * j3,
                                              rvx + sign12 * j1 + sign22 * j2 + sign32 * j3,
                                              rvy + sign13 * j1 + sign23 * j2 + sign33 * j3) * alphap1;
                             alphap1 *= alpha;
                           }
                           alphap2 *= alpha;
                         }
                         alphap3 *= alpha;
                       }
                       _buf_flatten(idx_dst) = fsum;
                     });
    }

    orcsum = 3;
    const int total_size3 = send_buffers->total_size_orc_.at(orcsum);
    if(total_size3 > 0) {
      Iterate_policy<1> policy1d(total_size3);
      Impl::for_each(policy1d,
                     [=](const int idx) {
                       const int idx_dst = _map_orc(idx, orcsum);
                       const int rx  = _map_bc(idx, orcsum, 0), ry  = _map_bc(idx, orcsum, 1);
                       const int rvx = _map_bc(idx, orcsum, 2), rvy = _map_bc(idx, orcsum, 3);
                       float64 fsum = 0.;
                       float64 alphap4 = alpha;
                       const int sign10 = _sign1(idx, orcsum, 0);
                       const int sign21 = _sign2(idx, orcsum, 1);
                       const int sign32 = _sign3(idx, orcsum, 2);
                       const int sign43 = _sign4(idx, orcsum, 3);
                       for(int j4 = 1; j4 <= MMAX; j4++) {
                         float64 alphap3 = alpha * alphap4;
                         for(int j3 = 1; j3 <= MMAX; j3++) {
                           float64 alphap2 = alpha * alphap3;
                           for(int j2 = 1; j2 <= MMAX; j2++) {
                             float64 alphap1 = alpha * alphap2;
                             for(int j1 = 1; j1 <= MMAX; j1++) {
                               fsum += _halo_fn(rx  + sign10 * j1,
                                                ry  + sign21 * j2,
                                                rvx + sign32 * j3,
                                                rvy + sign43 * j4) * alphap1;
                               alphap1 *= alpha;
                             }
                             alphap2 *= alpha;
                           }
                           alphap3 *= alpha;
                         }
                         alphap4 *= alpha;
                       }
                       _buf_flatten(idx_dst) = fsum;
                     });
    }
  }

  int mergeElts(std::vector<Halo> &v, std::vector<Halo>::iterator &f, std::vector<Halo>::iterator &g);
  
  // Wrapper for MPI communication
  void Isend(int &creq, std::vector<MPI_Request> &req);
          
  // Wrapper for MPI communication
  void Irecv(int &creq, std::vector<MPI_Request> &req);
                
  // Wrapper for MPI communication
  void Waitall(const int creq, std::vector<MPI_Request> &req, std::vector<MPI_Status> &stat) {
    const int nbreq = req.size();
    assert(creq == nbreq);
    MPI_Waitall(nbreq, req.data(), stat.data());
  }
};

#endif

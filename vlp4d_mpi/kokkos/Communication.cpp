#include "Communication.hpp"
#include "tiles.h"
#include <cmath>
#include <cassert>
#include <algorithm>

// Apply the Unbalanced Recursive Bisection (URB) algorithm to establish the boxes
// (parts of the computational domain) that are assigned to MPI processes. The result
// is stored within dis.ulist data structure
void Distrib::createDecomposition(Config *conf) {
  Domain &dom = conf->dom_;

  ulist_.resize(nbp_);
  for(int id=0; id<nbp_; id++)
    ulist_[id].pid_ = -1;

  ulist_[0].pid_ = 0;
  ulist_[0].nbp_ = nbp_;

  for(int j=0; j<DIMENSION; j++)
    ulist_[0].xmin_[j] = 0;

  for(int j=0; j<DIMENSION; j++) {
    ulist_[0].xmax_[j] = dom.nxmax_[j] - 1;
    nxmax_[j] = dom.nxmax_[j];
  }

  // Unbalanced Recursive Bisection
  // to create the domain decomposition among nbp processes
  for(int count = 1; count < nbp_; count++) {
    const float64 cutbound = 10000000000.;
    // Best score to choose a cut worse >= 1. / best = 0.
    float64 cutbest = cutbound;
    // Number of processes chosen for the cut
    int cutpart = -1;
    // Id of the process to be cut
    int cutid = -1;
    // Number of points to put into the newly created process
    int cutwidth = -1;
    // Direction of the cut
    int cutdir = -1;

    // Consider cutting only in the first NB_DIMS_DECOMPOSITION dimensions.
    // Valid values are in-between 1 and 4.
    for(int direction = 0; direction < NB_DIMS_DECOMPOSITION; direction++) {
      for(int id = 0; id < count; id++) {
        int nbp = ulist_[id].nbp_;
        if(nbp > 1) {
          int center = floor(nbp * 0.5);
          for(int part = center; part < nbp; part++) {
            const int width = ulist_[id].xmax_[direction] - ulist_[id].xmin_[direction] + 1;
            const float64 fract = static_cast<float64>(part) / static_cast<float64>(nbp);
            const float64 splitscore = 1. - width / static_cast<float64>(dom.nxmax_[direction]);
            //const float64 splitscore = 1. - round(width) / static_cast<float64>(dom.nxmax_[direction]);
            const int tmpwidth = round(fract * width);
            const float64 ratioscore = fabs(fract * width - tmpwidth);
            const float64 cutscore = ratioscore + splitscore + cutbound * (tmpwidth < MMAX + 2) + cutbound * ((width - tmpwidth) < (MMAX + 2));

            // select the best score to cut the former domain
            if(cutscore < cutbest) {
              cutid = id;
              cutbest = cutscore;
              cutpart = part;
              cutwidth = tmpwidth;
              cutdir = direction;
            }
          }//for(int part = center; part < nbp; part++)
        }//if(nbp > 1)
      }//for(int id = 0; id < count; id++)
    }//for(int direction = 0; direction < NB_DIMS_DECOMPOSITION; direction++)
    if(cutbest == cutbound) {
      printf("No cut found to create domain decomposition, reduce nb of processes\n");
    }

    for(int j=0; j<DIMENSION; j++) {
      ulist_[count].xmax_[j] = ulist_[cutid].xmax_[j];
      ulist_[count].xmin_[j] = ulist_[cutid].xmin_[j];
    }
    ulist_[cutid].nbp_          -= cutpart;
    ulist_[cutid].xmin_[cutdir] += cutwidth;
    ulist_[count].nbp_          = cutpart;
    ulist_[count].xmax_[cutdir] = ulist_[cutid].xmin_[cutdir] - 1;
  }//for(int count = 1; count < nbp_; count++)

  // Check that the sum of the subdomains size is equal
  // to the total domain size
  uint64 msum = 0;
  for(int id = 0; id < nbp_; id++) {
    const uint64 mcontrib = (ulist_[id].xmax_[3] - ulist_[id].xmin_[3] + 1) * (ulist_[id].xmax_[2] - ulist_[id].xmin_[2] + 1)
                          * (ulist_[id].xmax_[1] - ulist_[id].xmin_[1] + 1) * (ulist_[id].xmax_[0] - ulist_[id].xmin_[0] + 1);
    if(master())
      printf("[%d] local domain [%4u:%4u,%4u:%4u,%4u:%4u,%4u:%4u] cost %lu\n", id, ulist_[id].xmin_[0], ulist_[id].xmax_[0], ulist_[id].xmin_[1],
            ulist_[id].xmax_[1], ulist_[id].xmin_[2], ulist_[id].xmax_[2], ulist_[id].xmin_[3], ulist_[id].xmax_[3], mcontrib);
    msum += mcontrib;
  }
  uint64 mref = static_cast<uint64>(dom.nxmax_[3]) * static_cast<uint64>(dom.nxmax_[2]) * static_cast<uint64>(dom.nxmax_[1]) * static_cast<uint64>(dom.nxmax_[0]);

  if(mref != msum) {
    printf("Problem: sum check mref %lf mcalc %lu \n", mref, msum);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    exit(1);
  }

  node_ = &(ulist_[pid_]);

  // Update local domain size
  for(int j=0; j<DIMENSION; j++) {
    dom.local_nx_[j]    = node_->xmax_[j] - node_->xmin_[j] + 1;
    dom.local_nxmin_[j] = node_->xmin_[j];
    dom.local_nxmax_[j] = node_->xmax_[j];
  }
}

/*
  @biref Computes a list 'nlist' of *receiving* halos for the local MPI process.
         In order to achieve this, it fills the cells of halo_fn with the PID of the MPI.
         This function works in combination with comm_get_neibourghs, comm_merge_elts_ and comm_book_halo.
  @param[in]  halo_fn[]
    Identical to fn, used as a buffer to keep pid?
  @param[out] recv_list_
    Update the recv_list
  Used once in init.cpp
 */
void Distrib::neighboursList(Config *conf, RealOffsetView4D halo_fn) {
  const Domain& dom = conf->dom_;
  const int s_nxmax  = dom.nxmax_[0];
  const int s_nymax  = dom.nxmax_[1];
  const int s_nvxmax = dom.nxmax_[2];
  const int s_nvymax = dom.nxmax_[3];

  Urbnode &mynode = ulist_[pid_];
  typename RealOffsetView4D::HostMirror h_halo_fn = Kokkos::create_mirror_view(halo_fn);

  // Process periodic boundary condition to get a complete map
  for(int ivy = mynode.xmin_[3] - HALO_PTS; ivy < mynode.xmax_[3] + HALO_PTS + 1; ivy++) {
    for(int ivx = mynode.xmin_[2] - HALO_PTS; ivx < mynode.xmax_[2] + HALO_PTS + 1; ivx++) {
      const int jvy = (s_nvymax + ivy) % s_nvymax;
      const int jvx = (s_nvxmax + ivx) % s_nvxmax;

      for(int iy = mynode.xmin_[1] - HALO_PTS; iy < mynode.xmax_[1] + HALO_PTS + 1; iy++) {
        for(int ix = mynode.xmin_[0] - HALO_PTS; ix < mynode.xmax_[0] + HALO_PTS + 1; ix++) {
          const int jy = (s_nymax + iy) % s_nymax;
          const int jx = (s_nxmax + ix) % s_nxmax;

          int id = 0;
          bool notfound = true;
          while(notfound && (id<nbp_)) {
            const Urbnode &node = ulist_[id];
            if(    node.xmin_[0] <= jx  && jx  <= node.xmax_[0]
                && node.xmin_[1] <= jy  && jy  <= node.xmax_[1] 
                && node.xmin_[2] <= jvx && jvx <= node.xmax_[2]
                && node.xmin_[3] <= jvy && jvy <= node.xmax_[3]
                ) {
              h_halo_fn(ix, iy, ivx, ivy) = static_cast<float64>(id);
              notfound = 0;
            }
            id++;
          }
          assert(!notfound);
        }//for(int ix = mynode.xmin_[0] - HALO_PTS; ix < mynode.xmax_[0] + HALO_PTS + 1; ix++)
      }//for(int iy = mynode.xmin_[1] - HALO_PTS; iy < mynode.xmax_[1] + HALO_PTS + 1; iy++)
    }//for(int ivx = mynode.xmin_[2] - HALO_PTS; ivx < mynode.xmax_[2] + HALO_PTS + 1; ivx++)
  }//for(int ivy = mynode.xmin_[3] - HALO_PTS; ivy < mynode.xmax_[3] + HALO_PTS + 1; ivy++)

  // Use the map to establish neighbours in the communication
  // scheme of halo exchange
  int id = pid_;
  Urbnode &node = ulist_[id];
  std::vector<Halo> hlist;
  int count = 0;
  int jv[4];

  for(jv[0] = -1; jv[0] < 2; jv[0]++) {
    for(jv[1] = -1; jv[1] < 2; jv[1]++) {
      for(jv[2] = -1; jv[2] < 2; jv[2]++) {
        for(jv[3] = -1; jv[3] < 2; jv[3]++) {
          int face[8] = {
                         node.xmin_[0],
                         node.xmax_[0] + 1,
                         node.xmin_[1],
                         node.xmax_[1] + 1,
                         node.xmin_[2],
                         node.xmax_[2] + 1,
                         node.xmin_[3],
                         node.xmax_[3] + 1,
                        };
          for(int k = 0; k < 4; k++) {
            if(jv[k] == -1) {
              face[2 * k + 0] = node.xmin_[k] - HALO_PTS;
              face[2 * k + 1] = node.xmin_[k] - 1;
            }

            if(jv[k] == 1) {
              face[2 * k + 0] = node.xmax_[k] + 1;
              face[2 * k + 1] = node.xmax_[k] + HALO_PTS;
            }
          }

          if(jv[0] != 0 || jv[1] != 0 || jv[2] != 0 || jv[3] != 0)
            getNeighbours(conf, h_halo_fn, face, hlist, mynode.xmin_, mynode.xmax_, count);

          count++;
        }
      }
    }
  }
  // Compress the neighbours list using sorting + merging of list cells
  assert(dom.nxmax_[0] < 2010);
  assert(dom.nxmax_[1] < 2010);
  assert(dom.nxmax_[2] < 2010);
  assert(dom.nxmax_[3] < 2010);
  for(int k = 0; k < 4; k++) {
    int k0 = k;
    int k1 = (k + 1) % 4;
    int k2 = (k + 2) % 4;
    int k3 = (k + 3) % 4;
    auto larger = [k0, k1, k2, k3](const Halo &a, const Halo &b) {
      if(a.pid_ == b.pid_) {
        if(a.xmin_[k0] == b.xmin_[k0]) {
          if(a.xmin_[k1] == b.xmin_[k1]) {
            if(a.xmin_[k2] == b.xmin_[k2]) {
              return (a.xmin_[k3] < b.xmin_[k3]);
            } else {
              return (a.xmin_[k2] < b.xmin_[k2]);
            }
          } else {
            return (a.xmin_[k1] < b.xmin_[k1]);
          }
        } else {
          return (a.xmin_[k0] < b.xmin_[k0]);
        }
      } else {
        return (a.pid_ < b.pid_);
      }
    };
    std::sort(hlist.begin(), hlist.end(), larger);
    int cursize = hlist.size();
    int oldsize = cursize + 1;
    while(oldsize > cursize) {
      for(auto it = hlist.begin(); it != hlist.end(); ++it) {
        auto next = it + 1;
        if(next != hlist.end())
          mergeElts(hlist, it, next);
      }
      oldsize = cursize;
      cursize = hlist.size();
    }
  }

  // Print the halo list with neighbour id
  recv_list_.assign(hlist.begin(), hlist.end());
}

// Establish and allocate all receving/sending buffers
// for halo communication between MPI neighbours.
// To this end, it takes as an input 'rlist', the list
// of receiving halos. One ouput is 'slist', the halos
// to be sent.
/*
  @biref Get a list of the neighbours of the local MPI subdomain. Input is halo_fn
         that stores PIDs of the cells halo_fn(ivy,ivx,iy,ix). Ouput is hlist that
         includes a list of halo boxes. The union of all the hlist's boxes represents
         the actual halo of the local MPI subdomain.
         called in the initialization phase after neighboursList
  @param[in]  halo
    Identical to fn
  @param[in]  xrange[8] 
    face[8] = node.xmin[0], node.xmax[0] + 1, node.xmin[1], node.xmax[1] + 1, node.xmin[2], node.xmax[2] + 1, node.xmin[3], node.xmax[3] + 1}
    where node = ulist_[pid]
  @param[in]  lxmin[4]
    mynode.xmin, where mynode = ulist_[pid]
  @param[in]  lxmax[4]
    mynode.xmax, where mynode = ulist_[pid]
  @param[in]  count
    the total loop count of the 4D loop over halo regions ranging (-1:2) in each direction
  @param[inout] hlist
    Indentical to recv_list

  [Y. A comment] node and mynode seem to be identical, why distinguish?
  std::vector<urbnode_t>& nodes = (conf->dis).ulist;
  mynode = nodes[(conf->dis).pid]

  uint32_t id = dis.pid;
  node = nodes[id];
 */
void Distrib::bookHalo(Config *conf) {
  int size_halo = sizeof(Halo);
  std::vector<unsigned short> vectp(nbp_, 0);
  std::vector<unsigned short> redp(nbp_, 0);
  std::vector<MPI_Request> req;
  std::vector<MPI_Status>  stat;
  int tag = 111;
  int nbreq = 0;

  // Compute the number of local halos that should be received
  // depending of the PID that should send it.
  for(auto it = recv_list_.begin(); it != recv_list_.end(); ++it) {
    vectp[(*it).pid_]++;
  }

  // Reduce this 'vectp' array in order that each and ervey process
  // knows about the number of send/receive halos.
  MPI_Allreduce(vectp.data(), redp.data(), nbp_, MPI_UNSIGNED_SHORT, MPI_SUM, MPI_COMM_WORLD);

  // Initialize the vector 'slist' of halos to be sent
  nbreq = redp[pid_];
  send_list_.resize(nbreq);
  req.resize(nbreq + recv_list_.size());
  stat.resize(nbreq + recv_list_.size());

  // Receiving halo settings to be received to corresponding MPI processes
  for(size_t i = 0; i < send_list_.size(); i++) {
    Halo *sender = &(send_list_[i]);
    MPI_Irecv(sender, size_halo, MPI_BYTE, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &(req[i]));
  }
  // Send halo settings to be sent to corresponding MPI processes
  for(size_t i = 0; i < recv_list_.size(); i++) {
    Halo *recver = &(recv_list_[i]);
    int dest = recver->pid_;
    MPI_Isend(recver, size_halo, MPI_BYTE, dest, tag, MPI_COMM_WORLD, &(req[nbreq]));
    nbreq++;
  }
  // One should wait the end of the exchange before allocating the halo
  MPI_Waitall(nbreq, req.data(), stat.data());

  // All local halos dimensions are now well known.
  // Let's allocate them. (where these arrays are deallocated?
  for(size_t i = 0; i < send_list_.size(); i++) {
    Halo *halo = &(send_list_[i]);
    int32 size = (halo->xmax_[0] - halo->xmin_[0] + 1) * (halo->xmax_[1] - halo->xmin_[1] + 1) * (halo->xmax_[2] - halo->xmin_[2] + 1)
               * (halo->xmax_[3] - halo->xmin_[3] + 1);
    halo->size_ = size;
  }

  for(size_t i = 0; i < recv_list_.size(); i++) {
    Halo *halo = &(recv_list_[i]);
    int32 size = (halo->xmax_[0] - halo->xmin_[0] + 1) * (halo->xmax_[1] - halo->xmin_[1] + 1) * (halo->xmax_[2] - halo->xmin_[2] + 1)
               * (halo->xmax_[3] - halo->xmin_[3] + 1);
    halo->size_ = size;
  }

  // Set the PID of each halo's sender in 'slist''s components
  for(size_t i = 0; i < send_list_.size(); i++) {
    Halo *sender = &(send_list_[i]);
    sender->pid_ = stat[i].MPI_SOURCE;
  }

  // Copy into Kokkos suited structure
  // Halo size large enough to store 
  send_buffers_ = new Halos();
  recv_buffers_ = new Halos();
  send_buffers_->set(conf, send_list_, "send", nbp_, pid_);
  recv_buffers_->set(conf, recv_list_, "recv", nbp_, pid_);

  fprintf(stderr, "[%d] Number of halo blocs = %lu\n", pid_, send_list_.size());
}

/*
  @biref Possibly merge two halos 'f' and 'g' into a single one if ever it is possible.
         The vector 'v' contains 'f' and 'g' initially and 'g' can possibly be erased by this function.
         called in neighboursList function
  @param[in] v
    hlist
  @param[in] g
    The iterator of hlist pointing it + 1 (next)
  @param[out] f
    The iterator of hlist pointing it
 */
int Distrib::mergeElts(std::vector<Halo> &v, std::vector<Halo>::iterator &f, std::vector<Halo>::iterator &g) {
  if((*f).pid_ == (*g).pid_) {
    // Owned by the same MPI process
    for(uint32 i = 0; i < DIMENSION; i++) {
      bool equal = true;
      int retcode = 0;
      for(uint32 j = 0; j < DIMENSION; j++) {
        if(j != i) {
          equal = equal && ((*f).xmin_[j] == (*g).xmin_[j]) && ((*f).xmax_[j] == (*g).xmax_[j]);
        }
      }

      if(equal && ((*f).xmin_[i] == (*g).xmax_[i] + 1)) {
        (*f).xmin_[i] = (*g).xmin_[i];
        retcode = 1;
      }

      if(equal && ((*f).xmax_[i] + 1 == (*g).xmin_[i])) {
        (*f).xmax_[i] = (*g).xmax_[i];
        retcode = 2;
      }

      if(retcode != 0) {
        v.erase(g);
        return retcode;
      }
    }
  }
  return 0;
}

void Distrib::Isend(int &creq, std::vector<MPI_Request> &req) {
  int nb_halos = send_buffers_->nb_merged_halos_;
  int size_local_copy = send_buffers_->merged_size(pid_);
  for(int i = 0; i < nb_halos; i++) {
    if(i == pid_) {
      // Then local copy
      Kokkos::parallel_for("copy", size_local_copy, local_copy(send_buffers_, recv_buffers_));
    } else {
      // Then MPI communication
      float64 *head  = send_buffers_->head(i);
      const int size = send_buffers_->merged_size(i);
      const int pid  = send_buffers_->merged_pid(i);
      const int tag  = send_buffers_->merged_tag(i);
      if(size != 0)
        MPI_Isend(head, size, MPI_DOUBLE, pid, tag, MPI_COMM_WORLD, &(req[creq++]));
    }
  }
}
     
void Distrib::Irecv(int &creq, std::vector<MPI_Request> &req) {
  int nb_halos = recv_buffers_->nb_merged_halos_;
  int size_local_copy = recv_buffers_->merged_size(pid_);
  for(int i = 0; i < nb_halos; i++) {
    if(i != pid_) {
      // Then MPI communication
      float64 *head  = recv_buffers_->head(i);
      const int size = recv_buffers_->merged_size(i);
      const int pid  = recv_buffers_->merged_pid(i);
      const int tag  = recv_buffers_->merged_tag(i);
      if(size != 0)
        MPI_Irecv(head, size, MPI_DOUBLE, pid, tag, MPI_COMM_WORLD, &(req[creq++]));
    }
  }
}

void Distrib::boundary_condition_(RealOffsetView4D &halo_fn, Halos *send_buffers) {
  int orcsum = 0;
  const int total_size0 = send_buffers->total_size_orc_.at(orcsum);
  if(total_size0 > 0) {
    Kokkos::parallel_for("bc0", total_size0, boundary_condition_orc0(halo_fn, send_buffers));
  }

  orcsum = 1;
  const int total_size1 = send_buffers->total_size_orc_.at(orcsum);
  if(total_size1 > 0) {
    Kokkos::parallel_for("bc1", total_size1, boundary_condition_orc1(halo_fn, send_buffers));
  }

  orcsum = 2;
  const int total_size2 = send_buffers->total_size_orc_.at(orcsum);
  if(total_size2 > 0) {
    Kokkos::parallel_for("bc2", total_size2, boundary_condition_orc2(halo_fn, send_buffers));
  }

  orcsum = 3;
  const int total_size3 = send_buffers->total_size_orc_.at(orcsum);
  if(total_size3 > 0) {
    Kokkos::parallel_for("bc3", total_size3, boundary_condition_orc3(halo_fn, send_buffers));
  }
}

/*
  @biref Copy values of distribution function in to the halo regions (within slist) that will be sent.
         called in exchangeHalo function
  @param[in] send_list
  @param[in] recv_list
  @param[inout] halo_fn
    Indentical to fn?
 */
void Distrib::packAndBoundary(Config *conf, RealOffsetView4D halo_fn) {
  if(spline_) {
    /* Older version
    const int nx_send  = send_buffers_->nhalo_max_[0];
    const int ny_send  = send_buffers_->nhalo_max_[1];
    const int nvx_send = send_buffers_->nhalo_max_[2];
    const int nvy_send = send_buffers_->nhalo_max_[3];
    const int nb_send_halos = send_buffers_->nb_halos_;
    const int total_size = send_buffers_->total_size_;
    MDPolicyType_4D mdpolicy4d({{0, 0, 0, 0}},
                               {{nx_send, ny_send, nvx_send, nb_send_halos}},
                               {{TILE_SIZE0, TILE_SIZE1, TILE_SIZE2, TILE_SIZE3}}
                              );

    Kokkos::parallel_for("pack", mdpolicy4d, pack(conf, halo_fn, send_buffers_));

    #if defined( KOKKOS_ENABLE_CUDA )
      Kokkos::parallel_for("boundary_condition", mdpolicy4d, boundary_condition(conf, halo_fn, send_buffers_));
    #else
      Kokkos::parallel_for("boundary_condition", nb_send_halos, boundary_condition(conf, halo_fn, send_buffers_));
    #endif
    Kokkos::parallel_for("merged_pack", total_size, merged_pack(send_buffers_));
    */

    const int total_size = send_buffers_->total_size_;
    Kokkos::parallel_for("pack", total_size, pack_(halo_fn, send_buffers_));
    boundary_condition_(halo_fn, send_buffers_);
  }
}

void Distrib::unpack(RealOffsetView4D halo_fn) {
  int total_size = recv_buffers_->total_size_;
  Kokkos::parallel_for("merged_unpack", total_size, merged_unpack(halo_fn, recv_buffers_));
}


/*
  @biref Perform actually the exchange of halo with neighbouring MPI processes.
         'halo_fn' is the input/output distrib. function.
         'recv_list' and 'send_list' are the list of receive/send halos.
  @param[in] recv_list
  @param[in] send_list
  @param[inout] halo_fn
  [TO DO] names halo_fill_A and halo_fill_B should be renamed as halo_pack, halo_unpack
  This function is called inside the main loop
 */
void Distrib::exchangeHalo(Config *conf, RealOffsetView4D halo_fn, std::vector<Timer*> &timers) {
  std::vector<MPI_Request> req;
  std::vector<MPI_Status>  stat;
  int nbreq = 0, creq = 0;
  nbreq = recv_buffers_->nb_reqs() + send_buffers_->nb_reqs();
  creq  = 0;
  req.resize(nbreq);
  stat.resize(nbreq);

  // CUDA Aware MPI
  timers[TimerEnum::pack]->begin();
  Irecv(creq, req);
  packAndBoundary(conf, halo_fn);
  Kokkos::fence();
  timers[TimerEnum::pack]->end();

  timers[TimerEnum::comm]->begin();
  Isend(creq, req); // local copy done inside
  Waitall(creq, req, stat);

  // clear vectors
  std::vector<MPI_Request>().swap(req);
  std::vector<MPI_Status>().swap(stat);
  timers[TimerEnum::comm]->end();

  // copy halo regions back into distribution function
  timers[TimerEnum::unpack]->begin();
  unpack(halo_fn);
  Kokkos::fence();
  timers[TimerEnum::unpack]->end();
}

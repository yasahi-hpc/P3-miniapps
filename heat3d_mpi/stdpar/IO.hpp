#ifndef __IO_HPP__
#define __IO_HPP__

#include "Config.hpp"
#include "MPI_Comm.hpp"
#include "../Timer.hpp"
#include "../Helper.hpp"
#include <fstream>
#include <string>

void to_csv(Config &conf, Comm &comm, RealView3D &u, int iter, std::vector<Timer*> &timers);

void to_csv(Config &conf, Comm &comm, RealView3D &u, int iter, std::vector<Timer*> &timers) {
  if(iter % conf.freq_diag == 0) {
    timers[TimerEnum::IO]->begin();
    // Create a temporal global array
    const int nx = conf.nx, ny = conf.ny, nz = conf.nz;
    const int gnx = conf.gnx, gny = conf.gny, gnz = conf.gnz;
    RealView2D u_xy("u_xy", gnx, gny);
    RealView2D u_xyres("u_xyres", gnx, gny);

    u.updateSelf();

    // Get slice at the middle of the box in z direction
    const int giz0 = gnz / 2;

    auto cart_rank = comm.cart_rank();

    // Find the MPI process including giz0
    int iz0 = -1;
    bool found = false;

    for(int iz = 0; iz < nz; iz++) {
      int giz = iz + conf.nz * cart_rank.at(2);
      if(giz == giz0) {
        iz0 = iz;
        found = true;
        break;
      }
    }

    // copy to a local buffer at each process
    if(found) {
      for(int iy = 0; iy < ny; iy++) {
        for(int ix = 0; ix < nx; ix++) {
          int gix = ix + conf.nx * cart_rank.at(0);
          int giy = iy + conf.ny * cart_rank.at(1);
          u_xy(gix, giy) = u(ix, iy, iz0);
        }
      }
    }

    // Gather to master
    MPI_Datatype mpi_data_type = get_mpi_data_type<RealView2D::value_type>();
    MPI_Reduce(u_xy.data(), u_xyres.data(), u_xy.size(), mpi_data_type, MPI_SUM, 0, MPI_COMM_WORLD);

    // Open the file
    if(comm.is_master()) {
      std::string filename("data/heat3d_mpi/");
      filename += u.name() + "_xy_it" + zfill(iter, 6) + ".csv";
      std::ofstream fout(filename, std::ios::trunc);

      for(int iy = 0; iy < gny; iy++) {
        for(int ix = 0; ix < gnx; ix++) {
          if(ix == gnx-1) {
            fout << u_xyres(ix, iy) << std::endl;
          } else {
            fout << u_xyres(ix, iy) << ", ";
          }
        }
      }
      fout.close();
    }
    timers[TimerEnum::IO]->end();
  }
}

#endif

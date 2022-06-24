#ifndef __IO_HPP__
#define __IO_HPP__

#include "Config.hpp"
#include "../Timer.hpp"
#include "../Helper.hpp"
#include <fstream>
#include <string>

template <class View3DType>
void to_csv(Config &conf, View3DType &u, int iter, std::vector<Timer*> &timers) {
  if(iter % conf.freq_diag == 0) {
    timers[TimerEnum::IO]->begin();
    auto h_u = Kokkos::create_mirror_view(u);
    Kokkos::deep_copy(h_u, u);
    const int nx = conf.nx, ny = conf.ny, nz = conf.nz;

    // Get slice at the middle of the box in z direction
    const int iz0 = nz / 2;

    // Open the file u_
    std::string filename("data/heat3d/");
    filename += h_u.label() + "_xy_it" + zfill(iter, 6) + ".csv";
    std::ofstream fout(filename, std::ios::trunc);

    for(int iy = 0; iy < ny; iy++) {
      for(int ix = 0; ix < nx; ix++) {
        if(ix == nx-1) {
          fout << h_u(ix, iy, iz0) << std::endl;
        } else {
          fout << h_u(ix, iy, iz0) << ", ";
        }
      }
    }
    fout.close();
    timers[TimerEnum::IO]->end();
  }
}

#endif

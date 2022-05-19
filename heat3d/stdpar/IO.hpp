#ifndef __IO_HPP__
#define __IO_HPP__

#include "Config.hpp"
#include "../Timer.hpp"
#include "../Helper.hpp"
#include <fstream>
#include <string>

void to_csv(Config &conf, RealView3D &u, int iter, std::vector<Timer*> &timers);

void to_csv(Config &conf, RealView3D &u, int iter, std::vector<Timer*> &timers) {
  if(iter % conf.freq_diag == 0) {
    u.updateSelf();
    const int nx = conf.nx, ny = conf.ny, nz = conf.nz;

    // Get slice at the middle of the box in z direction
    const int iz0 = nz / 2;

    // Open the file u_
    std::string filename("data/heat3d/");
    filename += u.name() + "_xy_it" + zfill(iter, 6) + ".csv";
    std::ofstream fout(filename, std::ios::trunc);

    for(int iy = 0; iy < ny; iy++) {
      for(int ix = 0; ix < nx; ix++) {
        if(ix == nx-1) {
          fout << u(ix, iy, iz0) << std::endl;
        } else {
          fout << u(ix, iy, iz0) << ", ";
        }
      }
    }
    fout.close();
  }
}

#endif

#include "field.hpp"
#include "tiles.hpp"
#include "index.h"

void lu_solve_poisson(Config *conf, Efield *ef);

void field_rho(Config *conf, RealView4D &fn, Efield *ef) {
  const Domain *dom = &(conf->dom_);

  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];
  int nvx = dom->nxmax_[2], nvy = dom->nxmax_[3];
  double dvx = dom->dx_[2], dvy = dom->dx_[3];

  // Capturing a class member causes a problem
  // See https://github.com/kokkos/kokkos/issues/695
  RealView2D rho = ef->rho_;

  /*
  MDPolicy<2> integral_policy2d({{0, 0}},
                                {{nx, ny}},
                                {{TILE_SIZE0, TILE_SIZE1}}
                               );
  Kokkos::parallel_for("integral", integral_policy2d, KOKKOS_LAMBDA (const int ix, const int iy) {
    float64 sum = 0.;
    for(int ivy=0; ivy<nvy; ivy++) {
      for(int ivx=0; ivx<nvx; ivx++) {
        sum += fn(ix, iy, ivx, ivy);
      }
    }
    rho(ix, iy) = sum * dvx * dvy;
  });
  */

  // For some reason, the following version is significantly faster on both GPUs and CPUs
  Kokkos::parallel_for("integral", nx*ny, KOKKOS_LAMBDA (const int& ixy) {
    int2 idx_2D = Index::int2coord_2D(ixy, nx, ny);
    int ix = idx_2D.x, iy = idx_2D.y;
    float64 sum = 0.;
    for(int ivy=0; ivy<nvy; ivy++) {
      for(int ivx=0; ivx<nvx; ivx++) {
        sum += fn(ix, iy, ivx, ivy);
      }
    }
    rho(ix, iy) = sum * dvx * dvy;
  });

  Kokkos::fence();
};

void field_poisson(Config *conf, Efield *ef) {
  const Domain *dom = &(conf->dom_);
  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];
  int nvx = dom->nxmax_[2], nvy = dom->nxmax_[3];
  double dx = dom->dx_[0], dy = dom->dx_[1];
  double minPhyx = dom->minPhy_[0], minPhyy = dom->minPhy_[1];

  RealView2D rho = ef->rho_; 
  RealView2D ex  = ef->ex_; 
  RealView2D ey  = ef->ey_; 
  MDPolicy<2> poisson_policy2d({{0, 0}},
                               {{nx, ny}},
                               {{TILE_SIZE0, TILE_SIZE1}}
                              );
  switch(dom->idcase_)
  {
    case 2:
        Kokkos::parallel_for("poisson", poisson_policy2d, KOKKOS_LAMBDA (const int ix, const int iy) {
          ex(ix, iy) = -(minPhyx + ix * dx);
          ey(ix, iy) = 0.;
        });
        break;
    case 6:
        Kokkos::parallel_for("poisson", poisson_policy2d, KOKKOS_LAMBDA (const int ix, const int iy) {
          ey(ix, iy) = -(minPhyy + iy * dy);
          ex(ix, iy) = 0.;
        });
        break;
    case 10:
    case 20:
        Kokkos::parallel_for("poisson", poisson_policy2d, KOKKOS_LAMBDA (const int ix, const int iy) {
          rho(ix, iy) -= 1.;
        });
        lu_solve_poisson(conf, ef);
        break;
    default:
        lu_solve_poisson(conf, ef);
        break;
  }
  Kokkos::fence();
};

void lu_solve_poisson(Config *conf, Efield *ef) {
  const Domain *dom = &(conf->dom_);
  ef->solve_poisson_fftw(dom->maxPhy_[0], dom->maxPhy_[1]);
};

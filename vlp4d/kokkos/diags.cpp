#include "diags.hpp"
#include "tiles.hpp"
#include <stdio.h>

Diags::Diags(Config *conf) {
  const int nbiter = conf->dom_.nbiter_ + 1;

  nrj_ = RealHostView1D("nrj", nbiter);
  mass_ = RealHostView1D("mass", nbiter);
}

void Diags::compute(Config *conf, Efield *ef, int iter) {
  const Domain *dom = &conf->dom_;
  float64 iter_mass = 0.;
  float64 iter_nrj = 0.;
  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];

  assert(iter >= 0 && iter <= dom->nbiter_);
  RealView2D ex  = ef->ex_; 
  RealView2D ey  = ef->ey_; 
  RealView2D rho = ef->rho_; 

  // Capturing a class member causes a problem
  // See https://github.com/kokkos/kokkos/issues/695
  double_pair sum; sum.x = 0; sum.y = 0;

  MDPolicy<2> moment_policy2d({{0, 0}},
                              {{nx, ny}},
                              {{TILE_SIZE0, TILE_SIZE1}}
                             );

  Kokkos::parallel_reduce("moments", moment_policy2d, KOKKOS_LAMBDA (const int ix, const int iy,  double_pair& lsum) {
    const float64 eex = ex(ix, iy);
    const float64 eey = ey(ix, iy);

    lsum.x += rho(ix, iy);
    lsum.y += eex * eex + eey * eey;
  }, sum);

  iter_mass = sum.x;
  iter_nrj = sum.y;

  iter_nrj = sqrt(iter_nrj * dom->dx_[0] * dom->dx_[1]);
  iter_mass *= dom->dx_[0] + dom->dx_[1];

  iter_nrj = iter_nrj > 1.e-30 ? log(iter_nrj) : -1.e9;
  nrj_(iter) = iter_nrj;
  mass_(iter) = iter_mass;
}

void Diags::save(Config *conf) {
  const Domain* dom = &conf->dom_;

  char filename[16];
  sprintf(filename, "nrj.out");

  FILE *fileid = fopen(filename, "w");
  for(int iter=0; iter<=dom->nbiter_; ++iter)
    fprintf(fileid, "%17.13e %17.13e %17.13e\n", iter * dom->dt_, nrj_(iter), mass_(iter));

  fclose(fileid);
}

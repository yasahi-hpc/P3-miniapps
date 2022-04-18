#include "diags.hpp"
#include <omp.h>

Diags::Diags(Config *conf) {
  const int nbiter = conf->dom_.nbiter_ + 1;
  #if defined( ENABLE_OPENMP_OFFLOAD )
    #pragma omp target enter data map(alloc: this[0:1])
  #endif
  nrj_    = RealView1D("nrj",  nbiter);
  mass_   = RealView1D("mass", nbiter);
}

void Diags::compute(Config *conf, Efield *ef, int iter) {
  const Domain *dom = &conf->dom_;
  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];

  assert(iter >= 0 && iter <= dom->nbiter_);
  
  float64 iter_mass = 0.;
  float64 it_nrj = 0.;
  #if defined( ENABLE_OPENMP_OFFLOAD )
    #pragma omp target teams distribute parallel for simd collapse(2) map(tofrom:iter_mass,it_nrj) reduction(+:iter_mass,it_nrj)
  #else
    #pragma omp parallel for reduction(+:iter_mass,it_nrj)
  #endif 
  for(int iy = 0; iy < ny; iy++) {
    LOOP_SIMD
    for(int ix = 0; ix < nx; ix++) {
      const float64 eex = ef->ex_(ix, iy);
      const float64 eey = ef->ey_(ix, iy);
      iter_mass += ef->rho_(ix, iy);
      it_nrj    += eex * eex + eey * eey;
    }
  }

  it_nrj = sqrt(it_nrj * dom->dx_[0] * dom->dx_[1]);
  it_nrj = it_nrj > 1.e-30 ? log(it_nrj) : -1.e9;

  nrj_(iter)  = it_nrj;
  mass_(iter) = iter_mass * dom->dx_[0] * dom->dx_[1];
}

void Diags::save(Config *conf) {
  const Domain* dom = &conf->dom_;

  char filename[16];
  sprintf(filename, "nrj.out");

  FILE* fileid = fopen(filename, "w");
  for(int iter = 0; iter <= dom->nbiter_; ++iter)
      fprintf(fileid, "%17.13e %17.13e %17.13e\n", iter * dom->dt_, nrj_[iter], mass_[iter]);
  fclose(fileid);
}

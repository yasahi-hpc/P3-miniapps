#include "field.hpp"

void lu_solve_poisson(Config *conf, Efield *ef, Diags *dg, int iter);

void field_rho(Config *conf, RealView4D &fn, Efield *ef) {
  const Domain *dom = &(conf->dom_);

  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];
  int nvx = dom->nxmax_[2], nvy = dom->nxmax_[3];
  float64 dvx = dom->dx_[2], dvy = dom->dx_[3];

  #if defined( ENABLE_OPENACC )
    #pragma acc data present(ef[0:1], ef->rho_, fn)
    #pragma acc parallel loop
  #else
    #pragma omp parallel for
  #endif
  for(int iy=0; iy<ny; iy++) {
    LOOP_SIMD
    for(int ix=0; ix<nx; ix++) {
      float64 sum = 0.;
      for(int ivy=0; ivy<nvy; ivy++) {
        for(int ivx=0; ivx<nvx; ivx++) {
          sum += fn(ix, iy, ivx, ivy);
        }
      }
      ef->rho_(ix, iy) = sum * dvx * dvy;
    }
  }
}

void field_poisson(Config *conf, Efield *ef, Diags *dg, int iter) {
  const Domain *dom = &(conf->dom_);
  const int nx = dom->nxmax_[0];
  const int ny = dom->nxmax_[1];
  const float64 minPhyx = dom->minPhy_[0];
  const float64 minPhyy = dom->minPhy_[1];
  const float64 dx = dom->dx_[0];
  const float64 dy = dom->dx_[1];

  switch(dom->idcase_) {
    case 2:
      #if defined( ENABLE_OPENACC )
        #pragma acc data present(ef[0:1], ef->ex_, ef->ey_)
        #pragma acc parallel loop
      #else
        #pragma omp parallel for
      #endif
      for(int iy = 0; iy < ny; iy++) {
        LOOP_SIMD
        for(int ix = 0; ix < nx; ix++) {
          ef->ex_(ix, iy) = -(minPhyx + ix * dx);
          ef->ey_(ix, iy) = 0.;
        }
      }
      break;
    case 6:
      #if defined( ENABLE_OPENACC )
        #pragma acc data present(ef[0:1], ef->ex_, ef->ey_)
        #pragma acc parallel loop
      #else
        #pragma omp parallel for
      #endif
      for(int iy = 0; iy < ny; iy++) {
        LOOP_SIMD
        for(int ix = 0; ix < nx; ix++) {
          ef->ex_(ix, iy) = -(minPhyy + iy * dy);
          ef->ey_(ix, iy) = 0.;
        }
      }
      break;

    case 10:
    case 20:

      #if defined( ENABLE_OPENACC )
        #pragma acc data present(ef[0:1], ef->rho_)
        #pragma acc parallel loop
      #else
        #pragma omp parallel for
      #endif
      for(int iy = 0; iy < ny; iy++) {
        LOOP_SIMD
        for(int ix = 0; ix < nx; ix++) {
          ef->rho_(ix, iy) -= 1.;
        }
      }

      lu_solve_poisson(conf, ef, dg, iter);
      break;

    default:
      lu_solve_poisson(conf, ef, dg, iter);
      break;
  }
}

void lu_solve_poisson(Config *conf, Efield *ef, Diags *dg, int iter) {
  const Domain *dom = &(conf->dom_);
  ef->solve_poisson_fftw(dom->maxPhy_[0], dom->maxPhy_[1]);
  dg->compute(conf, ef, iter);
};

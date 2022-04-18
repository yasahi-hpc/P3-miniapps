#include "Diags.hpp"

Diags::Diags(Config *conf) {
  const int nbiter = conf->dom_.nbiter_ + 1;
  #if defined( ENABLE_OPENACC )
    #pragma acc enter data copyin(this)
  #endif
  nrj_    = RealView1D("nrj",  nbiter);
  nrjx_   = RealView1D("nrjx", nbiter);
  nrjy_   = RealView1D("nrjy", nbiter);
  mass_   = RealView1D("mass", nbiter);
  l2norm_ = RealView1D("l2norm", nbiter);
}

void Diags::compute(Config *conf, Efield *ef, int iter) {
  const Domain *dom = &conf->dom_;
  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];

  assert(iter >= 0 && iter <= dom->nbiter_);
  
  float64 iter_mass = 0.;
  float64 it_nrj = 0., it_nrjx = 0., it_nrjy = 0.;
  #if defined( ENABLE_OPENACC )
    #pragma acc data present(ef[0:1], ef->rho_, ef->ex_, ef->ey_)
    #pragma acc parallel loop reduction(+:iter_mass,it_nrj,it_nrjx,it_nrjy)
  #else
    #pragma omp parallel for reduction(+:iter_mass,it_nrj,it_nrjx,it_nrjy)
  #endif 
  for(int iy = 0; iy < ny; iy++) {
    LOOP_SIMD
    for(int ix = 0; ix < nx; ix++) {
      const float64 eex = ef->ex_(ix, iy);
      const float64 eey = ef->ey_(ix, iy);
      iter_mass += ef->rho_(ix, iy);
      it_nrj    += eex * eex + eey * eey;
      it_nrjx   += eex * eex;
      it_nrjy   += eey * eey;
    }
  }

  it_nrj = sqrt(it_nrj * dom->dx_[0] * dom->dx_[1]);
  it_nrj = it_nrj > 1.e-30 ? log(it_nrj) : -1.e9;

  nrj_(iter)  = it_nrj;
  nrjx_(iter) = sqrt(0.5 * it_nrjx * dom->dx_[0] * dom->dx_[1]);
  nrjy_(iter) = sqrt(0.5 * it_nrjy * dom->dx_[0] * dom->dx_[1]);
  mass_(iter) = iter_mass * dom->dx_[0] * dom->dx_[1];
}

void Diags::computeL2norm(Config *conf, RealView4D &fn, int iter) {
  const Domain *dom = &conf->dom_;
  int nx_min = dom->local_nxmin_[0], ny_min = dom->local_nxmin_[1], nvx_min = dom->local_nxmin_[2], nvy_min = dom->local_nxmin_[3];
  int nx_max = dom->local_nxmax_[0], ny_max = dom->local_nxmax_[1], nvx_max = dom->local_nxmax_[2], nvy_max = dom->local_nxmax_[3];

  float64 l2loc = 0.;
  #if defined( ENABLE_OPENACC )
    #pragma acc data present(fn)
    #pragma acc parallel loop independent collapse(3) reduction(+:l2loc)
  #else
    #pragma omp parallel for collapse(3) reduction(+:l2loc)
  #endif 
  for(int ivy = nvy_min; ivy <= nvy_max; ivy++) {
    for(int ivx = nvx_min; ivx <= nvx_max; ivx++) {
      for(int iy = ny_min; iy <= ny_max; iy++) {
        LOOP_SIMD
        for(int ix = nx_min; ix <= nx_max; ix++) {
          l2loc += fn(ix, iy, ivx, ivy) * fn(ix, iy, ivx, ivy);
        }
      }
    }
  }

  float64 l2glob = 0.;
  MPI_Reduce(&l2loc, &l2glob, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  l2norm_(iter) = sqrt(l2glob * dom->dx_[0] * dom->dx_[1] * dom->dx_[2] * dom->dx_[3]);
}

void Diags::save(Config *conf, Distrib &comm, int cur_iter) {
  const Domain* dom = &conf->dom_;

  char filename[16];

  if(comm.master()) {
    {
      sprintf(filename, "nrj.out");

      FILE *fileid = fopen(filename, (last_iter_ == 0 ? "w": "a"));
      for(int iter=last_iter_; iter<= cur_iter; ++iter)
        fprintf(fileid, "%17.13e %17.13e %17.13e %17.13e %17.13e\n", iter * dom->dt_, nrj_(iter), nrjx_(iter), nrjy_(iter), mass_(iter));

      fclose(fileid);
    }

    {
      sprintf(filename, "l2norm.out");

      FILE *fileid = fopen(filename, (last_iter_ == 0 ? "w": "a"));
      for(int iter=last_iter_; iter<= cur_iter; ++iter)
        fprintf(fileid, "%17.13e %17.13e\n", iter * dom->dt_, l2norm_(iter));

      fclose(fileid);
    }
  }
}

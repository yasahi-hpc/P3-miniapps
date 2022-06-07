#include "Field.hpp"
#include "Parallel_For.hpp"

void lu_solve_poisson(Config *conf, Efield *ef);

void field_rho(Config *conf, RealView4D &fn, Efield *ef) {
  const Domain *dom = &(conf->dom_);

  int nx_min = dom->local_nxmin_[0],     ny_min = dom->local_nxmin_[1],     nvx_min = dom->local_nxmin_[2],     nvy_min = dom->local_nxmin_[3];
  int nx_max = dom->local_nxmax_[0] + 1, ny_max = dom->local_nxmax_[1] + 1, nvx_max = dom->local_nxmax_[2] + 1, nvy_max = dom->local_nxmax_[3] + 1;
  float64 dvx = dom->dx_[2], dvy = dom->dx_[3];

  auto _fn = fn.mdspan();
  auto _rho_loc = ef->rho_loc_.mdspan();

  auto integral = [=](const int ix, const int iy) {
    float64 sum = 0.0;
    for(int ivy=nvy_min; ivy<nvy_max; ivy++) {
      for(int ivx=nvx_min; ivx<nvx_max; ivx++) {
        sum += _fn(ix, iy, ivx, ivy);
      }
    }
    _rho_loc(ix, iy) = sum * dvx * dvy;
  };

  Iterate_policy<2> policy2d({nx_min, ny_min}, {nx_max, ny_max});
  Impl::for_each(policy2d, integral);
}

void field_reduce(Config *conf, Efield *ef) {
  // reduction in velocity space
  const Domain *dom = &(conf->dom_);
  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];
  int nelems = nx * ny;

  float64 *ptr_rho     = ef->rho_.data();
  float64 *ptr_rho_loc = ef->rho_loc_.data();
  MPI_Allreduce(ptr_rho_loc, ptr_rho, nelems, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

void field_poisson(Config *conf, Efield *ef) {
  const Domain *dom = &(conf->dom_);
  const int nx = dom->nxmax_[0];
  const int ny = dom->nxmax_[1];
  const float64 minPhyx = dom->minPhy_[0];
  const float64 minPhyy = dom->minPhy_[1];
  const float64 dx = dom->dx_[0];
  const float64 dy = dom->dx_[1];

  auto _rho = ef->rho_.mdspan();
  auto _ex  = ef->ex_.mdspan();
  auto _ey  = ef->ey_.mdspan();

  Iterate_policy<2> policy2d({0, 0}, {nx, ny});

  switch(dom->idcase_) {
    case 2:
      Impl::for_each(policy2d,
        [=](const int ix, const int iy) {
          _ex(ix, iy) = -(minPhyx + ix * dx);
          _ey(ix, iy) = 0.;
        }
      );
      break;

    case 6:
      Impl::for_each(policy2d,
        [=](const int ix, const int iy) {
          _ex(ix, iy) = 0.;
          _ey(ix, iy) = -(minPhyy + iy * dy);
        }
      );
      break;

    case 10:
    case 20:
      Impl::for_each(policy2d,
        [=](const int ix, const int iy) {
          _rho(ix, iy) -= 1.;
        }
      );

      lu_solve_poisson(conf, ef);
      break;

    default:
      lu_solve_poisson(conf, ef);
      break;
  }
}

void lu_solve_poisson(Config *conf, Efield *ef) {
  const Domain *dom = &(conf->dom_);
  ef->solve_poisson_fftw(dom->maxPhy_[0], dom->maxPhy_[1]);
};

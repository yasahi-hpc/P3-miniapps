#include "field.hpp"
#include "thrust_parallel_for.hpp"

void lu_solve_poisson(Config *conf, Efield *ef, Diags *dg, int iter);

void field_rho(Config *conf, RealView4D &fn, Efield *ef) {
  const Domain *dom = &(conf->dom_);

  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];
  int nvx = dom->nxmax_[2], nvy = dom->nxmax_[3];
  float64 dvx = dom->dx_[2], dvy = dom->dx_[3];

  auto _fn = fn.device_mdspan();
  auto _rho = ef->rho_.device_mdspan();

  auto integral = [=] MDSPAN_FORCE_INLINE_FUNCTION (const unsigned int ix, const unsigned int iy) {
    float64 sum = 0.0;
    for(unsigned int ivy=0; ivy<nvy; ivy++) {
      for(unsigned int ivx=0; ivx<nvx; ivx++) {
        sum += _fn(ix, iy, ivx, ivy);
      }
    }
    _rho(ix, iy) = sum * dvx * dvy;
  };

  //Iterate_policy<2> policy2d({0, 0}, {nx, ny});
  //Impl::for_each(policy2d, integral);
  const int2 begin = make_int2(0, 0);
  const int2 end = make_int2(nx, ny);
 
  Impl::for_each(begin, end, integral);
};

void field_poisson(Config *conf, Efield *ef, Diags *dg, int iter) {
  const Domain *dom = &(conf->dom_);
  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];
  int nvx = dom->nxmax_[2], nvy = dom->nxmax_[3];
  double dx = dom->dx_[0], dy = dom->dx_[1];
  double minPhyx = dom->minPhy_[0], minPhyy = dom->minPhy_[1];

  auto rho = ef->rho_.device_mdspan();
  auto ex  = ef->ex_.device_mdspan();
  auto ey  = ef->ey_.device_mdspan();

  //Iterate_policy<2> policy2d({0, 0}, {nx, ny});
  const int2 begin = make_int2(0, 0);
  const int2 end = make_int2(nx, ny);
 
  switch(dom->idcase_)
  {
    case 2:
        Impl::for_each(begin, end,
          [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix, const int iy){
            ex(ix, iy) = -(minPhyx + ix * dx);
            ey(ix, iy) = 0.;
          });
        break;
    case 6:
        Impl::for_each(begin, end,
          [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix, const int iy){
            ey(ix, iy) = -(minPhyy + iy * dy);
            ex(ix, iy) = 0.;
          });
        break;
    case 10:
    case 20:
        Impl::for_each(begin, end,
          [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix, const int iy){
            rho(ix, iy) -= 1.;
          });
        lu_solve_poisson(conf, ef, dg, iter);
        break;
    default:
        lu_solve_poisson(conf, ef, dg, iter);
        break;
  }
};

void lu_solve_poisson(Config *conf, Efield *ef, Diags *dg, int iter) {
  const Domain *dom = &(conf->dom_);
  ef->solve_poisson_fftw(dom->maxPhy_[0], dom->maxPhy_[1]);
  dg->compute(conf, ef, iter);
};

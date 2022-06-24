#include <cstdio>
#include "diags.hpp"
#include "Parallel_Reduce.hpp"
#include "utils.hpp"

Diags::Diags(Config *conf) {
  const int nbiter = conf->dom_.nbiter_ + 1;

  nrj_ = RealView1D("nrj", nbiter);
  mass_ = RealView1D("mass", nbiter);
}

void Diags::compute(Config *conf, Efield *ef, int iter) {
  const Domain *dom = &conf->dom_;
  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];

  assert(iter >= 0 && iter <= dom->nbiter_);
  auto _rho = ef->rho_.mdspan();
  auto _ex  = ef->ex_.mdspan();
  auto _ey  = ef->ey_.mdspan();

  using moment_type = thrust::tuple<float64, float64>;
  moment_type zeros = {0, 0}, moments = {0, 0};

  auto moment_kernel = 
    [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix, const int iy) {
      const float64 eex = _ex(ix, iy);
      const float64 eey = _ey(ix, iy);
      const float64 rho = _rho(ix, iy);
      const float64 nrj  = eex*eex + eey*eey;
      return moment_type {rho, nrj};
  };

  auto binary_operator =
    [=] MDSPAN_FORCE_INLINE_FUNCTION (const moment_type &left, const moment_type &right) {
      return moment_type {thrust::get<0>(left) + thrust::get<0>(right),
                          thrust::get<1>(left) + thrust::get<1>(right)
                         };
  };

  int2 begin = make_int2(0, 0);
  int2 end   = make_int2(nx, ny);
  Impl::transform_reduce<default_iterate_layout>(begin, end, binary_operator, moment_kernel, moments);
  synchronize();

  float64 iter_mass = thrust::get<0>(moments);
  float64 iter_nrj  = thrust::get<1>(moments);

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

Diags::~Diags() {};

#include "diags.hpp"
#include "Parallel_For.hpp"
#include <cstdio>
#include <numeric>
#include <execution>
#include <algorithm>

Diags::Diags(Config *conf) {
  const int nbiter = conf->dom_.nbiter_ + 1;

  nrj_  = RealView1D("nrj", nbiter);
  mass_ = RealView1D("mass", nbiter);
}

void Diags::compute(Config *conf, Efield *ef, int iter) {
  const Domain *dom = &conf->dom_;
  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];
  const int n = nx * ny;

  assert(iter >= 0 && iter <= dom->nbiter_);
  auto _ex = ef->ex_.mdspan();
  auto _ey = ef->ey_.mdspan();
  auto _rho = ef->rho_.mdspan();
 
  using moment_type = std::tuple<float64, float64>;
  moment_type zeros = {0, 0}, moments = {0, 0};
  if(std::is_same_v<default_iterate_layout, layout_contiguous_at_left>) {
    moments = std::transform_reduce(std::execution::par_unseq,
                                    counting_iterator(0), counting_iterator(n),
                                    zeros,
                                    [=] (const moment_type &left, const moment_type &right) {
                                      return moment_type {std::get<0>(left) + std::get<0>(right),
                                                          std::get<1>(left) + std::get<1>(right)
                                                         };
                                    },
                                    [=] (const int idx) {
                                      const int ix = idx % nx, iy = idx / nx;
                                      const float64 eex = _ex(ix, iy);
                                      const float64 eey = _ey(ix, iy);
                                      const float64 rho = _rho(ix, iy);
                                      const float64 nrj  = eex*eex + eey*eey;
                                      return std::tuple<float64, float64> {rho, nrj};
                                    }
                                   );
  } else {
    moments = std::transform_reduce(std::execution::par_unseq,
                                    counting_iterator(0), counting_iterator(n),
                                    zeros,
                                    [=] (const moment_type &left, const moment_type &right) {
                                      return moment_type {std::get<0>(left) + std::get<0>(right),
                                                          std::get<1>(left) + std::get<1>(right)
                                                         };
                                    },
                                    [=] (const int idx) {
                                      const int iy = idx % ny, ix = idx / ny;
                                      const float64 eex = _ex(ix, iy);
                                      const float64 eey = _ey(ix, iy);
                                      const float64 rho = _rho(ix, iy);
                                      const float64 nrj  = eex*eex + eey*eey;
                                      return std::tuple<float64, float64> {rho, nrj};
                                    }
                                   );
  }

  float64 iter_mass = std::get<0>(moments);
  float64 iter_nrj  = std::get<1>(moments);
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

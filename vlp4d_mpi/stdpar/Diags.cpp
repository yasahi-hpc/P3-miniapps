#include "Diags.hpp"
#include "../Index.hpp"
#include <omp.h>
#include <numeric>
#include <execution>
#include <algorithm>

// This may be useful
// https://docs.microsoft.com/en-us/cpp/parallel/concrt/how-to-perform-map-and-reduce-operations-in-parallel?view=msvc-160

Diags::Diags(Config *conf) {
  const Domain *dom = &(conf->dom_);
  const int nx = dom->nxmax_[0];
  const int ny = dom->nxmax_[1];
  const int nbiter = dom->nbiter_ + 1;
  nrj_    = RealView1D("nrj",  nbiter);
  nrjx_   = RealView1D("nrjx", nbiter);
  nrjy_   = RealView1D("nrjy", nbiter);
  mass_   = RealView1D("mass", nbiter);
  l2norm_ = RealView1D("l2norm", nbiter);
}

void Diags::compute(Config *conf, Efield *ef, int iter) {
  using layout_type = RealView4D::layout_type;
  const Domain *dom = &conf->dom_;
  int nx = dom->nxmax_[0], ny = dom->nxmax_[1];
  const int n = nx*ny;

  assert(iter >= 0 && iter <= dom->nbiter_);

  auto _ex = ef->ex_.mdspan();
  auto _ey = ef->ey_.mdspan();
  auto _rho = ef->rho_.mdspan();

  using moment_type = std::tuple<float64, float64, float64, float64>;
  moment_type zeros = {0, 0, 0, 0}, moments = {0, 0, 0, 0};

  if(std::is_same_v<layout_type, layout_contiguous_at_left>) {
    moments = std::transform_reduce(std::execution::par_unseq,
                                  counting_iterator(0), counting_iterator(n),
                                  zeros,
                                  [=] (const moment_type &left, const moment_type &right) {
                                    return moment_type {std::get<0>(left) + std::get<0>(right),
                                                        std::get<1>(left) + std::get<1>(right),
                                                        std::get<2>(left) + std::get<2>(right),
                                                        std::get<3>(left) + std::get<3>(right)
                                                       };
                                  },
                                  [=] (const int idx) {
                                    const int ix = idx % nx, iy = idx / nx;
                                    const float64 eex = _ex(ix, iy);
                                    const float64 eey = _ey(ix, iy);
                                    const float64 rho = _rho(ix, iy);
                                    const float64 nrjx = eex*eex;
                                    const float64 nrjy = eey*eey;
                                    const float64 nrj  = nrjx + nrjy;
                                    return std::tuple<float64, float64, float64, float64> {rho, nrj, nrjx, nrjy};
                                  }
                                 );
  } else {
    moments = std::transform_reduce(std::execution::par_unseq,
                                  counting_iterator(0), counting_iterator(n),
                                  zeros,
                                  [=] (const moment_type &left, const moment_type &right) {
                                    return moment_type {std::get<0>(left) + std::get<0>(right),
                                                        std::get<1>(left) + std::get<1>(right),
                                                        std::get<2>(left) + std::get<2>(right),
                                                        std::get<3>(left) + std::get<3>(right)
                                                       };
                                  },
                                  [=] (const int idx) {
                                    const int iy = idx % ny, ix = idx / ny;
                                    const float64 eex = _ex(ix, iy);
                                    const float64 eey = _ey(ix, iy);
                                    const float64 rho = _rho(ix, iy);
                                    const float64 nrjx = eex*eex;
                                    const float64 nrjy = eey*eey;
                                    const float64 nrj  = nrjx + nrjy;
                                    return std::tuple<float64, float64, float64, float64> {rho, nrj, nrjx, nrjy};
                                  }
                                 );
  }

  float64 iter_mass = std::get<0>(moments);
  float64 it_nrj    = std::get<1>(moments);
  float64 it_nrjx   = std::get<2>(moments);
  float64 it_nrjy   = std::get<3>(moments);

  it_nrj = sqrt(it_nrj * dom->dx_[0] * dom->dx_[1]);
  it_nrj = it_nrj > 1.e-30 ? log(it_nrj) : -1.e9;

  nrj_(iter)  = it_nrj;
  nrjx_(iter) = sqrt(0.5 * it_nrjx * dom->dx_[0] * dom->dx_[1]);
  nrjy_(iter) = sqrt(0.5 * it_nrjy * dom->dx_[0] * dom->dx_[1]);
  mass_(iter) = iter_mass * dom->dx_[0] * dom->dx_[1];
}

void Diags::computeL2norm(Config *conf, RealView4D &fn, int iter) {
  using layout_type = RealView4D::layout_type;
  const Domain *dom = &conf->dom_;
  int nx_min = dom->local_nxmin_[0], ny_min = dom->local_nxmin_[1], nvx_min = dom->local_nxmin_[2], nvy_min = dom->local_nxmin_[3];
  int nx_max = dom->local_nxmax_[0], ny_max = dom->local_nxmax_[1], nvx_max = dom->local_nxmax_[2], nvy_max = dom->local_nxmax_[3];
  const int nx  = nx_max  - nx_min + 1;
  const int ny  = ny_max  - ny_min + 1;
  const int nvx = nvx_max - nvx_min + 1;
  const int nvy = nvy_max - nvy_min + 1;

  const int n = nx * ny * nvx * nvy;
  auto _fn = fn.mdspan();

  float64 l2loc = 0.0;
  if(std::is_same_v<layout_type, layout_contiguous_at_left>) {
    l2loc = std::transform_reduce(std::execution::par_unseq,
                                  counting_iterator(0), counting_iterator(n),
                                  0,
                                  std::plus<>(),
                                  [=] (const int idx) {
                                    const int ix   = idx % nx + nx_min;
                                    const int iyzw = idx / nx;
                                    const int iy   = iyzw%ny + ny_min;
                                    const int izw  = iyzw/ny;
                                    const int ivx  = izw%nvx + nvx_min;
                                    const int ivy  = izw/nvx + nvy_min;
                                    return _fn(ix, iy, ivx, ivy) * _fn(ix, iy, ivx, ivy);
                                  }
                                 );
  } else {
    l2loc = std::transform_reduce(std::execution::par_unseq,
                                  counting_iterator(0), counting_iterator(n),
                                  0,
                                  std::plus<>(),
                                  [=] (const int idx) {
                                    const int ivy   = idx % nvy + nvy_min;
                                    const int ixyz = idx / nvy;
                                    const int ivx   = ixyz%nvx + nvx_min;
                                    const int ixy  = ixyz/nvx;
                                    const int iy   = ixy%ny + ny_min;
                                    const int ix   = ixy/ny + nx_min;
                                    return _fn(ix, iy, ivx, ivy) * _fn(ix, iy, ivx, ivy);
                                  }
                                 );
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

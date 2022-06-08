#include "Diags.hpp"
#include "Parallel_Reduce.hpp"

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

  assert(iter >= 0 && iter <= dom->nbiter_);

  auto _ex = ef->ex_.mdspan();
  auto _ey = ef->ey_.mdspan();
  auto _rho = ef->rho_.mdspan();

  using moment_type = std::tuple<float64, float64, float64, float64>;
  moment_type moments = {0, 0, 0, 0};

  Iterate_policy<2> policy2d({0, 0}, {nx, ny});

  auto moment_kernel =
    [=](const int ix, const int iy) {
      const float64 eex = _ex(ix, iy);
      const float64 eey = _ey(ix, iy);
      const float64 rho = _rho(ix, iy);
      const float64 nrjx = eex*eex;
      const float64 nrjy = eey*eey;
      const float64 nrj  = nrjx + nrjy;
      return moment_type {rho, nrj, nrjx, nrjy};
  };

  auto binary_operator =
    [=] (const moment_type &left, const moment_type &right) {
      return moment_type {std::get<0>(left) + std::get<0>(right),
                          std::get<1>(left) + std::get<1>(right),
                          std::get<2>(left) + std::get<2>(right),
                          std::get<3>(left) + std::get<3>(right)
                         };
  };

  Impl::transform_reduce(policy2d, binary_operator, moment_kernel, moments);

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
  int nx_max = dom->local_nxmax_[0]+1, ny_max = dom->local_nxmax_[1]+1, nvx_max = dom->local_nxmax_[2]+1, nvy_max = dom->local_nxmax_[3]+1;

  auto _fn = fn.mdspan();

  float64 l2loc = 0.0;
  auto L2norm_kernel =
    [=] (const int ix, const int iy, const int ivx, const int ivy) {
      return _fn(ix, iy, ivx, ivy) * _fn(ix, iy, ivx, ivy);
  };

  Iterate_policy<4> policy4d({nx_min, ny_min, nvx_min, nvy_min}, {nx_max, ny_max, nvx_max, nvy_max});
  Impl::transform_reduce(policy4d, std::plus<float64>(), L2norm_kernel, l2loc);

  float64 l2glob = 0.;
  MPI_Reduce(&l2loc, &l2glob, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  l2norm_(iter) = sqrt(l2glob * dom->dx_[0] * dom->dx_[1] * dom->dx_[2] * dom->dx_[3]);
}

void Diags::save(Config *conf, Distrib &comm) {
  const Domain* dom = &conf->dom_;

  char filename[16];

  if(comm.master()) {
    {
      sprintf(filename, "nrj.out");

      FILE *fileid = fopen(filename, "w");
      for(int iter=0; iter<=dom->nbiter_; ++iter)
        fprintf(fileid, "%17.13e %17.13e %17.13e %17.13e %17.13e\n", iter * dom->dt_, nrj_(iter), nrjx_(iter), nrjy_(iter), mass_(iter));

      fclose(fileid);
    }

    {
      sprintf(filename, "l2norm.out");

      FILE *fileid = fopen(filename, "w");
      for(int iter=0; iter<=dom->nbiter_; ++iter)
        fprintf(fileid, "%17.13e %17.13e\n", iter * dom->dt_, l2norm_(iter));

      fclose(fileid);
    }
  }
}

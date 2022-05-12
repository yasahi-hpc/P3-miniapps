#include "Diags.hpp"
#include "Parallel_Reduce.hpp"
#include "Utils.hpp"

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

  using moment_type = thrust::tuple<float64, float64, float64, float64>;
  moment_type moments = {0, 0, 0, 0};

  auto moment_kernel =
    [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix, const int iy) {
      const float64 eex = _ex(ix, iy);
      const float64 eey = _ey(ix, iy);
      const float64 rho = _rho(ix, iy);
      const float64 nrjx = eex*eex;
      const float64 nrjy = eey*eey;
      const float64 nrj  = nrjx + nrjy;
      return moment_type {rho, nrj, nrjx, nrjy};
  };
 
  auto binary_operator =
    [=] MDSPAN_FORCE_INLINE_FUNCTION (const moment_type &left, const moment_type &right) {
      return moment_type {thrust::get<0>(left) + thrust::get<0>(right),
                          thrust::get<1>(left) + thrust::get<1>(right),
                          thrust::get<2>(left) + thrust::get<2>(right),
                          thrust::get<3>(left) + thrust::get<3>(right)
                         };
  };

  int2 begin = make_int2(0, 0);
  int2 end   = make_int2(nx, ny);
  Impl::transform_reduce<default_iterate_layout>(begin, end, binary_operator, moment_kernel, moments);
  synchronize();

  float64 iter_mass = thrust::get<0>(moments);
  float64 it_nrj    = thrust::get<1>(moments);
  float64 it_nrjx   = thrust::get<2>(moments);
  float64 it_nrjy   = thrust::get<3>(moments);

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

  auto _fn = fn.mdspan();

  float64 l2loc = 0.0;
  auto L2norm_kernel =
    [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix, const int iy, const int ivx, const int ivy) {
      return _fn(ix, iy, ivx, ivy) * _fn(ix, iy, ivx, ivy);
  };

  int4 begin = make_int4(nx_min, ny_min, nvx_min, nvy_min);
  int4 end   = make_int4(nx_max, ny_max, nvx_max, nvy_max);
  Impl::transform_reduce<default_iterate_layout>(begin, end, thrust::plus<float64>(), L2norm_kernel, l2loc);
  synchronize();

  float64 l2glob = 0.0;
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

#ifndef __TUNIHG_HPP__
#define __TUNIHG_HPP__

#include "Efield.hpp"
#include "Diags.hpp"
#include "Types.hpp"
#include "Math.hpp"
#include "Communication.hpp"
#include "Advection.hpp"
#include "Spline.hpp"
#include "Tile_size_tuning.hpp"
#include "Math.hpp"
#include <chrono>
#include <sstream>
#include <iomanip>
#include <functional>

void tileSizeTuning(Config *conf, Distrib &comm, TileSizeTuning &tuning, RealOffsetView4D fn, RealOffsetView4D fnp1, RealOffsetView4D fn_tmp, Efield *ef, Diags *dg, Spline *spline, int iter);

void tileSizeTuning(Config *conf, Distrib &comm, TileSizeTuning &tuning, RealOffsetView4D fn, RealOffsetView4D fnp1, RealOffsetView4D fn_tmp, Efield *ef, Diags *dg, Spline *spline, int iter) {
  const Domain *dom = &(conf->dom_);
  int nx_min = dom->local_nxmin_[0], ny_min = dom->local_nxmin_[1], nvx_min = dom->local_nxmin_[2], nvy_min = dom->local_nxmin_[3];
  int nx_max = dom->local_nxmax_[0], ny_max = dom->local_nxmax_[1], nvx_max = dom->local_nxmax_[2], nvy_max = dom->local_nxmax_[3];

  Impl::deep_copy(fnp1, fn);
  // Tuning for Spline_xy
  auto spline_xy = std::bind(&Spline::computeCoeff_xy, spline, conf, fnp1, std::placeholders::_1);
  tuning.registerKernel("coeff_xy", {nvx_max+1-nvx_min, nvy_max+1-nvy_min, 1, 1});
  tuning.scan("coeff_xy", spline_xy, comm.rank());

  // Run Spline_xy
  spline->computeCoeff_xy(conf, fn);
  Impl::deep_copy(fnp1, fn_tmp, fn);

  // Tuning for Adv2D
  auto adv2d = std::bind(Advection::advect_2D_xy, conf, fnp1, fn_tmp, dom->dt_*0.5, std::placeholders::_1);
  tuning.registerKernel("Adv2D", {nx_max+1-nx_min, ny_max+1-ny_min, nvx_max+1-nvx_min, nvy_max+1-nvy_min});
  tuning.scan("Adv2D", adv2d, comm.rank());

  // Run Adv2D
  Advection::advect_2D_xy(conf, fn, fn_tmp, 0.5 * dom->dt_);

  // Tuning for field_rho
  auto integral = std::bind(field_rho, conf, fn, ef, std::placeholders::_1);
  tuning.registerKernel("integral", {nx_max+1-nx_min, ny_max+1-ny_min, 1, 1});
  tuning.scan("integral", integral, comm.rank());

  // Run fields
  field_rho(conf, fn, ef);
  field_reduce(conf, ef);
  field_poisson(conf, ef, dg, iter);

  // Tuning for Spline_vxvy
  Impl::deep_copy(fnp1, fn);
  auto spline_vxvy = std::bind(&Spline::computeCoeff_vxvy, spline, conf, fnp1, std::placeholders::_1);
  tuning.registerKernel("coeff_vxvy", {nx_max+1-nx_min, ny_max+1-ny_min, 1, 1});
  tuning.scan("coeff_vxvy", spline_vxvy, comm.rank());

  // Run Spline_vxvy
  spline->computeCoeff_vxvy(conf, fnp1);

  // Tuning for Adv4D
  auto adv4d = std::bind(Advection::advect_4D, conf, ef, fnp1, fn, dom->dt_, std::placeholders::_1);
  tuning.registerKernel("Adv4D", {nx_max+1-nx_min, ny_max+1-ny_min, nvx_max+1-nvx_min, nvy_max+1-nvy_min});
  tuning.scan("Adv4D", adv4d, comm.rank());

  // Auto tuning completed!
  #if defined( KOKKOS_ENABLE_CUDA )
    tuning.toCsv("Cuda", comm.rank());
  #else
    tuning.toCsv("OpenMP", comm.rank());
  #endif
}

#endif

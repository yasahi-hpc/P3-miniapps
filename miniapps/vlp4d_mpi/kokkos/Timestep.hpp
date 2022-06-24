#ifndef __TIMESTEP_HPP__
#define __TIMESTEP_HPP__

#include "Efield.hpp"
#include "Diags.hpp"
#include "Types.hpp"
#include "Math.hpp"
#include "Communication.hpp"
#include "Advection.hpp"
#include "../Timer.hpp"
#include "Spline.hpp"
#include "Tile_size_tuning.hpp"

void onetimestep(Config *conf, Distrib &comm, RealOffsetView4D fn, RealOffsetView4D fnp1, RealOffsetView4D fn_tmp,
                 Efield *ef, Diags *dg, Spline *spline, std::vector<Timer*> &timers, int iter);
void onetimestep(Config *conf, Distrib &comm, TileSizeTuning &tuning, RealOffsetView4D fn, RealOffsetView4D fnp1, RealOffsetView4D fn_tmp,
                 Efield *ef, Diags *dg, Spline *spline, std::vector<Timer*> &timers, int iter);

void onetimestep(Config *conf, Distrib &comm, RealOffsetView4D fn, RealOffsetView4D fnp1, RealOffsetView4D fn_tmp,
                 Efield *ef, Diags *dg, Spline *spline, std::vector<Timer*> &timers, int iter) {
  Domain *dom = &(conf->dom_);

  // Exchange halo of the local domain in order to perform
  // the advection afterwards (the interpolation needs the
  // extra points located in the halo region)
  comm.exchangeHalo(conf, fn, timers);

  timers[Splinecoeff_xy]->begin();
  spline->computeCoeff_xy(conf, fn);
  timers[Splinecoeff_xy]->end();

  Impl::deep_copy(fnp1, fn);

  timers[Advec2D]->begin();
  Impl::deep_copy(fn_tmp, fn);
  Advection::advect_2D_xy(conf, fn, fn_tmp, 0.5 * dom->dt_);
  timers[Advec2D]->end();

  timers[TimerEnum::Field]->begin();
  field_rho(conf, fn, ef);
  timers[TimerEnum::Field]->end();

  timers[TimerEnum::AllReduce]->begin();
  field_reduce(conf, ef);
  timers[TimerEnum::AllReduce]->end();

  timers[TimerEnum::Fourier]->begin();
  field_poisson(conf, ef);
  timers[TimerEnum::Fourier]->end();

  timers[Diag]->begin();
  dg->compute(conf, ef, iter);
  timers[Diag]->end();

  timers[Splinecoeff_vxvy]->begin();
  spline->computeCoeff_vxvy(conf, fnp1);
  timers[Splinecoeff_vxvy]->end();

  timers[Advec4D]->begin();
  Advection::advect_4D(conf, ef, fnp1, fn, dom->dt_);
  timers[Advec4D]->end();

  timers[TimerEnum::Field]->begin();
  field_rho(conf, fnp1, ef);
  timers[TimerEnum::Field]->end();

  timers[TimerEnum::AllReduce]->begin();
  field_reduce(conf, ef);
  timers[TimerEnum::AllReduce]->end();

  timers[TimerEnum::Fourier]->begin();
  field_poisson(conf, ef);
  timers[TimerEnum::Fourier]->end();

  timers[Diag]->begin();
  dg->compute(conf, ef, iter);
  dg->computeL2norm(conf, fnp1, iter);

  if(iter % dom->ifreq_ == 0) {
    if(dom->fxvx_) Advection::print_fxvx(conf, comm, fnp1, iter);
  }
  Kokkos::fence();
  timers[Diag]->end();
};

void onetimestep(Config *conf, Distrib &comm, TileSizeTuning &tuning, RealOffsetView4D fn, RealOffsetView4D fnp1, RealOffsetView4D fn_tmp,
                 Efield *ef, Diags *dg, Spline *spline, std::vector<Timer*> &timers, int iter) {
  Domain *dom = &(conf->dom_);

  // Exchange halo of the local domain in order to perform
  // the advection afterwards (the interpolation needs the
  // extra points located in the halo region)
  comm.exchangeHalo(conf, fn, timers);

  timers[Splinecoeff_xy]->begin();
  spline->computeCoeff_xy(conf, fn, tuning.bestTileSize("coeff_xy"));
  timers[Splinecoeff_xy]->end();

  Impl::deep_copy(fnp1, fn_tmp, fn);

  timers[Advec2D]->begin();
  Advection::advect_2D_xy(conf, fn, fn_tmp, 0.5 * dom->dt_, tuning.bestTileSize("Adv2D"));
  timers[Advec2D]->end();

  timers[TimerEnum::Field]->begin();
  field_rho(conf, fn, ef, tuning.bestTileSize("integral"));
  timers[TimerEnum::Field]->end();

  timers[TimerEnum::AllReduce]->begin();
  field_reduce(conf, ef);
  timers[TimerEnum::AllReduce]->end();

  timers[TimerEnum::Fourier]->begin();
  field_poisson(conf, ef);
  timers[TimerEnum::Fourier]->end();

  timers[Diag]->begin();
  dg->compute(conf, ef, iter);
  timers[Diag]->end();

  timers[Splinecoeff_vxvy]->begin();
  spline->computeCoeff_vxvy(conf, fnp1, tuning.bestTileSize("coeff_vxvy"));
  timers[Splinecoeff_vxvy]->end();

  timers[Advec4D]->begin();
  Advection::advect_4D(conf, ef, fnp1, fn, dom->dt_, tuning.bestTileSize("Adv4D"));
  timers[Advec4D]->end();

  timers[TimerEnum::Field]->begin();
  field_rho(conf, fnp1, ef, tuning.bestTileSize("integral"));
  timers[TimerEnum::Field]->end();

  timers[TimerEnum::AllReduce]->begin();
  field_reduce(conf, ef);
  timers[TimerEnum::AllReduce]->end();

  timers[TimerEnum::Fourier]->begin();
  field_poisson(conf, ef);
  timers[TimerEnum::Fourier]->end();

  timers[Diag]->begin();
  dg->compute(conf, ef, iter);
  dg->computeL2norm(conf, fnp1, iter);

  if(iter % dom->ifreq_ == 0) {
    if(dom->fxvx_) Advection::print_fxvx(conf, comm, fnp1, iter); // [May be done]
  }
  Kokkos::fence();
  timers[Diag]->end();
};

#endif

#ifndef __TIMESTEP_HPP__
#define __TIMESTEP_HPP__

/* 
 * Halo exchange on f^n
 * Compute spline coeff. along x, y: f^n -> f_xy^n
 * Advection 2D in space x, y of f^n
 * Compute density f_xy^n -> rho^n+1/2 and solve Poisson in Fourier space phi^n+1/2
 * Diagnostics/outputs on phi^n+1/2
 *
 * Compute spline coeff. along vx, vy: f_xy^n -> f_xy,vxvy^n
 * Estimate 4D displacements: phi^n+1/2 -> delta^n+1/2
 * Advection 4D in x, y, vx, vy of f^n: f_xy,vxvy^n and delta^n+1/2 -> f^n+1
 *
 */

#include "Efield.hpp"
#include "Diags.hpp"
#include "Types.hpp"
#include "Communication.hpp"
#include "Spline.hpp"
#include "Config.hpp"
#include "../Timer.hpp"
#include "Math.hpp"
#include "Advection.hpp"
#include "Field.hpp"
#include "Utils.hpp"

void onetimestep(Config *conf, Distrib &comm, RealView4D &fn, RealView4D &fnp1, RealView4D &fn_tmp,
                 Efield *ef, Diags *dg, Spline *spline, std::vector<Timer*> &timers, int iter);

void onetimestep(Config *conf, Distrib &comm, RealView4D &fn, RealView4D &fnp1, RealView4D &fn_tmp,
                 Efield *ef, Diags *dg, Spline *spline, std::vector<Timer*> &timers, int iter) {
  Domain *dom = &(conf->dom_);

  // Exchange halo of the local domain in order to perform
  // the advection afterwards (the interpolation needs the 
  // extra points located in the halo region)
  comm.exchangeHalo(conf, fn, timers);

  timers[Splinecoeff_xy]->begin();
  spline->computeCoeff_xy(fn);
  Impl::deep_copy(fnp1, fn);
  synchronize();
  timers[Splinecoeff_xy]->end();

  timers[Advec2D]->begin();
  Advection::advect_2D_xy(conf, fn, fn_tmp, 0.5 * dom->dt_);
  synchronize();
  timers[Advec2D]->end();

  timers[TimerEnum::Field]->begin();
  field_rho(conf, fn, ef);
  synchronize();
  timers[TimerEnum::Field]->end();

  timers[TimerEnum::AllReduce]->begin();
  field_reduce(conf, ef);
  timers[TimerEnum::AllReduce]->end();

  timers[TimerEnum::Fourier]->begin();
  field_poisson(conf, ef, iter);
  synchronize();
  timers[TimerEnum::Fourier]->end();

  timers[Diag]->begin();
  dg->compute(conf, ef, iter);
  synchronize();
  timers[Diag]->end();

  timers[Splinecoeff_vxvy]->begin();
  spline->computeCoeff_vxvy(fnp1);
  synchronize();
  timers[Splinecoeff_vxvy]->end();

  timers[Advec4D]->begin();
  Advection::advect_4D(conf, ef, fnp1, fn, dom->dt_);
  synchronize();
  timers[Advec4D]->end();

  timers[TimerEnum::Field]->begin();
  field_rho(conf, fnp1, ef);
  synchronize();
  timers[TimerEnum::Field]->end();

  timers[TimerEnum::AllReduce]->begin();
  field_reduce(conf, ef);
  timers[TimerEnum::AllReduce]->end();

  timers[TimerEnum::Fourier]->begin();
  field_poisson(conf, ef, iter);
  synchronize();
  timers[TimerEnum::Fourier]->end();

  timers[Diag]->begin();
  dg->compute(conf, ef, iter);
  dg->computeL2norm(conf, fnp1, iter);

  if(iter % dom->ifreq_ == 0) {
    if(dom->fxvx_) Advection::print_fxvx(conf, comm, fnp1, iter);
  }
  synchronize();
  timers[Diag]->end();
}

#endif

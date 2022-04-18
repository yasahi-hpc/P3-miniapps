#ifndef __TIMESTEP_HPP__
#define __TIMESTEP_HPP__

#include "efield.hpp"
#include "diags.hpp"
#include "types.hpp"
#include "advection.hpp"
#include "field.hpp"
#include "../timer.hpp"
#include "utils.hpp"

void onetimestep(Config *conf, RealView4D &fn, RealView4D &fnp1, Efield *ef, Diags *dg, std::vector<Timer*> &timers, int iter);

void onetimestep(Config *conf, RealView4D &fn, RealView4D &fnp1, Efield *ef, Diags *dg, std::vector<Timer*> &timers, int iter) {
  const Domain *dom = &(conf->dom_);

  timers[Advec1D_x]->begin();
  Advection::advect_1D_x(conf, fn, fnp1, 0.5 * dom->dt_);
  synchronize();
  timers[Advec1D_x]->end();

  timers[Advec1D_y]->begin();
  Advection::advect_1D_y(conf, fnp1, fn, 0.5 * dom->dt_);
  synchronize();
  timers[Advec1D_y]->end();

  timers[Field]->begin();
  field_rho(conf, fn, ef);
  synchronize();
  timers[Field]->end();

  timers[Fourier]->begin();
  field_poisson(conf, ef, dg, iter);
  synchronize();
  timers[Fourier]->end();

  timers[Advec1D_vx]->begin();
  Advection::advect_1D_vx(conf, fn, fnp1, ef, dom->dt_);
  synchronize();
  timers[Advec1D_vx]->end();

  timers[Advec1D_vy]->begin();
  Advection::advect_1D_vy(conf, fnp1, fn, ef, dom->dt_);
  synchronize();
  timers[Advec1D_vy]->end();

  timers[Advec1D_x]->begin();
  Advection::advect_1D_x(conf, fn, fnp1, 0.5 * dom->dt_);
  synchronize();
  timers[Advec1D_x]->end();

  timers[Advec1D_y]->begin();
  Advection::advect_1D_y(conf, fnp1, fn, 0.5 * dom->dt_);
  synchronize();
  timers[Advec1D_y]->end();

  if(dom->fxvx_)
    Advection::print_fxvx(conf, fn, iter);
}

#endif

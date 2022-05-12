#ifndef __INIT_HPP__
#define __INIT_HPP__

#include "Efield.hpp"
#include "Diags.hpp"
#include "Types.hpp"
#include "Config.hpp"
#include "Communication.hpp"
#include "Spline.hpp"

void init(const char *file, Config *conf, Distrib &comm, RealView4D &fn, RealView4D &fnp1, RealView4D &fn_tmp, Efield **ef, Diags **dg, Spline **spline, std::vector<Timer*> &timers);
void finalize(Efield **ef, Diags **dg, Spline **spline);

#endif

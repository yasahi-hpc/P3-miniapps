#ifndef __INIT_HPP__
#define __INIT_HPP__

#include "Efield.hpp"
#include "Diags.hpp"
#include "Spline.hpp"
#include "Types.hpp"
#include "Config.hpp"
#include "Communication.hpp"

void init(const char *file, Config *conf, Distrib &comm, RealOffsetView4D &fn, RealOffsetView4D &fnp1, RealOffsetView4D &fn_tmp, Efield **ef, Diags **dg, Spline **spline, std::vector<Timer*> &timers);
void initValues(Config *conf, RealOffsetView4D &fn, RealOffsetView4D &fnp1);
void finalize(Efield **ef, Diags **dg, Spline **spline);

#endif

#ifndef __INIT_HPP__
#define __INIT_HPP__

#include "efield.hpp"
#include "diags.hpp"

void init(const char *file, Config *conf, RealView4D &fn, RealView4D &fnp1, Efield **ef, Diags **dg);
void finalize(Config *conf, Efield **ef, Diags **dg);

#endif

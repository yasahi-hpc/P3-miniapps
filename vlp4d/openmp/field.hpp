#ifndef __FIELD_HPP__
#define __FIELD_HPP__

#include "config.hpp"
#include "efield.hpp"
#include "diags.hpp"
#include "types.hpp"

void field_rho(Config *conf, RealView4D &fn, Efield *ef);
void field_poisson(Config *conf, Efield *ef, Diags *dg, int iter);

#endif

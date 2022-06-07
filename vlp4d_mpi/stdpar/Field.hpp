#ifndef __FIELD_HPP__
#define __FIELD_HPP__

#include "Config.hpp"
#include "Efield.hpp"
#include "Diags.hpp"
#include "Types.hpp"
#include "Communication.hpp"

void field_rho(Config *conf, RealView4D &fn, Efield *ef);
void field_reduce(Config *conf, Efield *ef);
void field_poisson(Config *conf, Efield *ef, int iter);

#endif

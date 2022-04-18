#ifndef __FIELD_HPP__
#define __FIELD_HPP__

#include "Config.hpp"
#include "Efield.hpp"
#include "Diags.hpp"
#include "Types.hpp"
#include "Communication.hpp"
#include "tiles.h"

void field_rho(Config *conf, RealOffsetView4D fn, Efield *ef, 
               const std::vector<int> &tiles={TILE_SIZE0, TILE_SIZE1});
void field_reduce(Config *conf, Efield *ef);
void field_poisson(Config *conf, Efield *ef, Diags *dg, int iter);

#endif

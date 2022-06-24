#ifndef __FIELD_HPP__
#define __FIELD_HPP__

#include "config.hpp"
#include "efield.hpp"
#include "diags.hpp"
#include "types.hpp"

/*
 * @param[in] fn
 * @param[out] ef.rho_ (Updated by the integral of fn)
 * @param[out] ef.ex_ (zero initialization)
 * @param[out] ef.ey_ (zero initialization)
 * @param[out] ef.phi_ (zero initialization)
 */
void field_rho(Config *conf, RealView4D &fn, Efield *ef);
void field_poisson(Config *conf, Efield *ef);

#endif

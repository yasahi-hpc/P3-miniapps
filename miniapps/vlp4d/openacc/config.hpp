#ifndef __CONFIG_HPP__
#define __CONFIG_HPP__

#include "types.hpp"

static constexpr size_t DIMENSION = 4;

struct Domain {
  shape_nd<DIMENSION> nxmax_; /* Number of points on the finest grid  */
  int nbiter_, ifreq_; /* number of iteration and diagnostic frequency */
  int idcase_; /* Identifier of the test to run */

  float64 minPhy_[DIMENSION]; /* min and max coordinates in x and v */
  float64 maxPhy_[DIMENSION];
  float64 dt_; /*step in time */
  float64 dx_[DIMENSION]; /* step in x and v */
  char limitCondition_[DIMENSION];
  int fxvx_;
};

struct Physics {
  /*Physic parameter */
  float64 z_; /* position in accelerator */
  float64 S_; /* lattice period */
  float64 omega02_; /* coefficient of applied field */
  float64 vbeam_; /* beam velocity */
  float64 beta_, psi_; /* parameters for applied field */
  float64 eps0_; /*permittivity of free space */
  float64 echarge_; /* charge of particle */
  float64 mass_; /*mass of particle */
};

struct Config {
  Physics phys_;
  Domain dom_;
};

#endif

#ifndef __CONFIG_HPP__
#define __CONFIG_HPP__

#include "Types.hpp"

static constexpr int DIMENSION = 4;
#define HALO_PTS 3
#define MMAX 8

#if LAG_ORDER == 4
  #define LAG_OFFSET 2
  #define LAG_HALO_PTS 3
  #define LAG_PTS 5
#endif

#if LAG_ORDER == 5
  #define LAG_OFFSET 2
  #define LAG_HALO_PTS 3
  #define LAG_PTS 6
  #define LAG_ODD
#endif

#if LAG_ORDER == 7
  #define LAG_OFFSET 3
  #define LAG_HALO_PTS 4
  #define LAG_PTS 8
  #define LAG_ODD
#endif

#ifdef LAG_ORDER
  #define HALO_PTS LAG_HALO_PTS
#endif

struct Domain {
  shape_nd<DIMENSION> nxmax_; // Number of points on the finest grid
  shape_nd<DIMENSION> local_nx_;    // Number of points on the local grid
  range_nd<DIMENSION> local_nxmax_; // Maximum of local grid
  range_nd<DIMENSION> local_nxmin_; // Minimum of local grid
  int nbiter_, ifreq_; // Number of iteration and diagnostic frequency
  int idcase_; // Identifier of the test to run

  float64 minPhy_[DIMENSION]; // min and max coordinates in x and v
  float64 maxPhy_[DIMENSION]; // min and max coordinates in x and v
  float64 dt_; // step in time
  float64 dx_[DIMENSION]; // step in x and v
  char limitCondition_[DIMENSION];
  int fxvx_;
};

struct Physics {
  // Physics parameter
  float64 z_; // position in accelerator
  float64 S_; // lattice period
  float64 omega02_; // coefficient of applied field
  float64 vbeam_; // beam velocity
  float64 beta_, psi_; // parameters for applied field
  float64 eps0_; // permittivity of free space
  float64 echarge_; // charge of particle
  float64 mass_; // mass of particle
};
  
struct Config {
  Physics phys_;
  Domain dom_;
};

#endif

#ifndef __DIAGS_HPP__
#define __DIAGS_HPP__

#include "types.hpp"
#include "config.hpp"
#include "efield.hpp"

struct Diags {
private:
  RealView1D nrj_;
  RealView1D mass_;

public:
  Diags(Config *conf);
  ~Diags(){
    #if defined( ENABLE_OPENMP_OFFLOAD )
      #pragma omp target exit data map(delete: this[0:1])
    #endif
  }

  void compute(Config *conf, Efield *ef, int iter);
  void save(Config *conf);
};

#endif

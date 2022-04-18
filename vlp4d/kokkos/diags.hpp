#ifndef __DIAGS_HPP__
#define __DIAGS_HPP__

#include "types.hpp"
#include "config.hpp"
#include "efield.hpp"

struct Diags {
private:
  using RealHostView1D = RealView1D::HostMirror;
  
  RealHostView1D nrj_;
  RealHostView1D mass_;

public:
  Diags(Config *conf);
  virtual ~Diags() {}

  void compute(Config *conf, Efield *ef, int iter);
  void save(Config *conf);
};

#endif

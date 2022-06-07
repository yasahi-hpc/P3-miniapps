#ifndef __DIAGS_HPP__
#define __DIAGS_HPP__

#include "Types.hpp"
#include "Config.hpp"
#include "Efield.hpp"
#include "Communication.hpp"

struct Diags {
private:
  using RealHostView1D = RealView1D::HostMirror;
  RealHostView1D nrj_;
  RealHostView1D nrjx_;
  RealHostView1D nrjy_;
  RealHostView1D mass_;
  RealHostView1D l2norm_;
  int last_iter_ = 0;

public:
  Diags(Config *conf);
  ~Diags();

  void compute(Config *conf, Efield *ef, int iter);
  void computeL2norm(Config *conf, RealOffsetView4D fn, int iter);
  void save(Config *conf, Distrib &comm);
};

#endif

#ifndef __DIAGS_HPP__
#define __DIAGS_HPP__

#include "types.hpp"
#include "config.hpp"
#include "efield.hpp"
#include "utils.hpp"

struct Diags
{
private:
  RealView1D nrj_;
  RealView1D mass_;

public:
  Diags(Config *conf);
  virtual ~Diags();

  void compute(Config *conf, Efield *ef, int iter);
  void save(Config *conf);

private:
  DISALLOW_COPY_AND_ASSIGN(Diags);
};

#endif

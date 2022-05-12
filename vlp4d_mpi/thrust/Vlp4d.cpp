/*
 * @brief  vlp4d parallelized with std::parallel algorithm
 * @author Yuuichi ASAHI
 * @date   14/April/2022
 *
 */

#include <mpi.h>
#include <iostream>
#include <cstdio>
#include "../Parser.hpp"
#include "../Timer.hpp"
#include "Config.hpp"
#include "Communication.hpp"
#include "Efield.hpp"
#include "Field.hpp"
#include "Diags.hpp"
#include "Init.hpp"
#include "Timestep.hpp"
#include "Spline.hpp"

int main (int argc, char *argv[]) {
  Parser parser(argc, argv);
  Distrib comm(argc, argv);

  Config conf;
  RealView4D fn, fnp1;
  RealView4D fn_tmp;

  Efield *ef = nullptr;
  Diags  *dg = nullptr;
  Spline *spline = nullptr;

  std::vector<Timer*> timers;
  defineTimers(timers);

  // Initialization
  if(comm.master()) printf("reading input file %s\n", parser.file_);
  init(parser.file_, &conf, comm, fn, fnp1, fn_tmp, &ef, &dg, &spline, timers);

  int iter = 0;

  timers[Total]->begin();
  field_rho(&conf, fn, ef);
  field_reduce(&conf, ef);
  field_poisson(&conf, ef, dg, iter);
  dg->computeL2norm(&conf, fn, iter);

  // Main time step loop
  while(iter < conf.dom_.nbiter_) {
    timers[MainLoop]->begin();
    if(comm.master()) {
      printf("iter %d\n", iter);
    }

    iter++;
    onetimestep(&conf, comm, fn, fnp1, fn_tmp, ef, dg, spline, timers, iter);
    fn.swap(fnp1);
    timers[MainLoop]->end();
  }
  timers[Total]->end();

  finalize(&ef, &dg, &spline);

  if(comm.master()) {
    printTimers(timers);
  }
  comm.cleanup();

  freeTimers(timers);
  comm.finalize();

  return 0;
}

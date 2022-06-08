/*
 * @brief The vlp4d code solves Vlasov-Poisson equations in 4D (2d space, 2d velocity). 
 *        From the numerical point of view, vlp4d is based on a semi-lagrangian scheme. 
 *        Vlasov solver is typically based on a directional Strang splitting. 
 *        The Poisson equation is treated with 2D Fourier transforms. 
 *        For the sake of simplicity, all directions are, for the moment, handled with periodic boundary conditions.
 *        The Vlasov solver is based on advection's operators:
 *
 *        2D advection along x, y (Dt/2)
 *        Poisson solver -> compute electric fields Ex and E
 *        4D advection along x, y, vx, vy (Dt)
 *
 *        Interpolation operator within advection is Lagrange polynomial of order 5, 7 depending on a compilation flag (order 5 by default).
 *
 *  @author
 *  @url    https://gitlab.maisondelasimulation.fr/GyselaX/vlp4d/tree/master
 */

#include <Kokkos_Core.hpp>
#include <cstdio>
#include "Types.hpp"
#include "Config.hpp"
#include "Init.hpp"
#include "../Parser.hpp"
#include "Communication.hpp"
#include "Efield.hpp"
#include "Diags.hpp"
#include "Field.hpp"
#include "Math.hpp"
#include "Timestep.hpp"
#include "Tuning.hpp"
#include "../Timer.hpp"
#include "Spline.hpp"

int main (int argc, char* argv[]) {
  Parser parser(argc, argv);
  Distrib comm(argc, argv);

  std::vector<Timer*> timers;
  defineTimers(timers);

  // When initializing Kokkos, you may pass in command-line arguments,
  // just like with MPI_Init().  Kokkos reserves the right to remove
  // arguments from the list that start with '--kokkos-'.
  Kokkos::InitArguments args_kokkos;
  args_kokkos.num_threads = parser.num_threads_;
  args_kokkos.num_numa    = parser.teams_;
  args_kokkos.device_id   = parser.device_;

  Kokkos::initialize (args_kokkos);
  {
    Config conf;
    RealOffsetView4D fn, fnp1;
    RealOffsetView4D fn_tmp;
    Efield *ef = nullptr;
    Diags  *dg = nullptr;
    Spline *spline = nullptr;

    // Initialization
    if(comm.master()) printf("reading input file %s\n", parser.file_);
    init(parser.file_, &conf, comm, fn, fnp1, fn_tmp, &ef, &dg, &spline, timers);
    int iter = 0;

    if(comm.master()) {
      #if defined( TILE_SIZE_TUNING )
        std::cout << "Auto tuning over tile sizes" << std::endl;
      #endif
      #if defined( LAYOUT_LEFT )
        std::cout << "Layout tuning for CPU" << std::endl;
      #endif
    }

    #if defined( TILE_SIZE_TUNING )
      field_rho(&conf, fn, ef);
      field_reduce(&conf, ef);
      field_poisson(&conf, ef);

      comm.exchangeHalo(&conf, fn, timers);

      // Tuning solve the first step of onetimestep repeatedly
      TileSizeTuning tuning;
      tileSizeTuning(&conf, comm, tuning, fn, fnp1, fn_tmp, ef, dg, spline, iter);

      // After tuning init again
      initValues(&conf, fn, fnp1);
    #endif

    Kokkos::fence();
    MPI_Barrier(MPI_COMM_WORLD);
    timers[Total]->begin();

    field_rho(&conf, fn, ef);
    field_reduce(&conf, ef);
    field_poisson(&conf, ef);
    dg->compute(&conf, ef, iter);
    dg->computeL2norm(&conf, fn, iter);

    Kokkos::fence();
    while(iter <conf.dom_.nbiter_) {
      timers[MainLoop]->begin();
      if(comm.master()) {
        printf("iter %d\n", iter);
      }

      iter++;
      #if defined( TILE_SIZE_TUNING )
        onetimestep(&conf, comm, tuning, fn, fnp1, fn_tmp, ef, dg, spline, timers, iter);
      #else
        onetimestep(&conf, comm, fn, fnp1, fn_tmp, ef, dg, spline, timers, iter);
      #endif
      Impl::swap(fn, fnp1);
      timers[MainLoop]->end();
    }
    Kokkos::fence();
    timers[Total]->end();
    finalize(&conf, comm, &ef, &dg, &spline);
    comm.cleanup();
  }
  Kokkos::finalize();
  if(comm.master()) {
    printTimers(timers);
  }
  freeTimers(timers);
  comm.finalize();
  return 0;
}

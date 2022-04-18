/*
//@HEADER
// ************************************************************************
// 
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
// 
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
// 
// ************************************************************************
//@HEADER
*/

/*
 * @brief The vlp4d code solves Vlasov-Poisson equations in 4D (2d space, 2d velocity). 
 *        From the numerical point of view, vlp4d is based on a semi-lagrangian scheme. 
 *        Vlasov solver is typically based on a directional Strang splitting. 
 *        The Poisson equation is treated with 2D Fourier transforms. 
 *        For the sake of simplicity, all directions are, for the moment, handled with periodic boundary conditions.
 *        The Vlasov solver is based on advection's operators:
 *
 *        1D advection along x (Dt/2)
 *        1D advection along y (Dt/2)
 *        Poisson solver -> compute electric fields Ex and E
 *        1D advection along vx (Dt)
 *        1D advection along vy (Dt)
 *        1D advection along x (Dt/2)
 *        1D advection along y (Dt/2)
 *
 *        Interpolation operator within advection is Lagrange polynomial of order 5, 7 depending on a compilation flag (order 5 by default).
 *
 *  @author
 *  @url    https://github.com/yasahi-hpc/vlp4d
 */

#include <Kokkos_Core.hpp>
#include "types.hpp"
#include "config.hpp"
#include "efield.hpp"
#include "field.hpp"
#include "diags.hpp"
#include "timestep.hpp"
#include "init.hpp"
#include "../timer.hpp"
#include <cstdio>

int main (int argc, char* argv[]) {
  std::vector<Timer*> timers;
  defineTimers(timers);

  // When initializing Kokkos, you may pass in command-line arguments,
  // just like with MPI_Init().  Kokkos reserves the right to remove
  // arguments from the list that start with '--kokkos-'.
  Kokkos::initialize (argc, argv);
  {
    Config conf;
    RealView4D fn, fnp1;
    Efield *ef = nullptr;
    Diags *dg = nullptr;

    // Initialization
    if(argc == 2) {
      /* A file is given in parameter */
      printf("reading input file %s\n", argv[1]);
      fflush(stdout);

      init(argv[1], &conf, fn, fnp1, &ef, &dg);
    }
    else {
      printf("argc != 2, reading 'data.dat' by default\n");
      fflush(stdout);

      init("data.dat", &conf, fn, fnp1, &ef, &dg);
    }
    int iter = 0;

    #if defined( MDRange3D )
      printf("Using 3D MDRangePolicy\n");
    #endif
    Kokkos::fence();

    timers[Total]->begin();
    field_rho(&conf, fn, ef);
    field_poisson(&conf, ef, dg, iter);

    Kokkos::fence();
    while(iter < conf.dom_.nbiter_) {
      timers[MainLoop]->begin();
      printf("iter %d\n", iter);

      iter++;
      onetimestep(&conf, fn, fnp1, ef, dg, timers, iter);
      timers[MainLoop]->end();
    }
    timers[Total]->end();
    Kokkos::fence();

    finalize(&conf, &ef, &dg);
  }
  Kokkos::finalize();
  printTimers(timers);
  freeTimers(timers);

  return 0;
}

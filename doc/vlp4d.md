# vlp4d code

## Brief description
The vlp4d code solves Vlasov-Poisson equations in 4D (2d space, 2d velocity). 
From the numerical point of view, vlp4d is based on a semi-lagrangian scheme. 
Vlasov solver is typically based on a directional Strang splitting. The Poisson equation is treated with 2D Fourier transforms. 
For the sake of simplicity, all directions are, for the moment, handled with periodic boundary conditions. 
We have prepared the non-MPI and MPI versions, where the non-MPI version uses the Lagrange interpolation and MPI version uses the Spline interpolation.

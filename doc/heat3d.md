# About
Solving the 3D heat equation ![heat equation](https://latex.codecogs.com/svg.latex?u_t=\kappa\nabla^2u) with a finite difference method. The computatinal domain is a cube with the size of ![domain size](https://latex.codecogs.com/svg.latex?L^3). Animation shows the simulation results.
![heat3d](figs/heat_xy_anime.gif) 

At the simulation results are compared with the analytical solution for sanity check.
The difference `L2_norm` will be printed out in the standard output.

# Numerical settings
## Boundary conditions
We apply preidoic boundary conditions in each direction as 
- ![boundary x](https://latex.codecogs.com/svg.latex?u(0,y,z,t)=u(L,y,z,t)) 
- ![boundary y](https://latex.codecogs.com/svg.latex?u(x,0,z,t)=u(x,L,z,t))  
- ![boundary z](https://latex.codecogs.com/svg.latex?u(x,y,0,t)=u(x,y,L,t))   

## Initial condition
In order to compare with the analytical solution, we apply the following initial condition:  
![initial condition](https://latex.codecogs.com/svg.latex?u(x,y,z,t=0)=\cos(2\pi(x/L+y/L+z/L))) 

## Analytical solution
With the boundary and initial conditions given above, we have the analytical solution as
![analytical solution](https://latex.codecogs.com/svg.latex?u(x,y,z,t)=\cos(2\pi(x/L+y/L+z/L))\exp(-3\kappa*(2\pi/L)^2t).)  
The numerical solution is checked against this solution.

# Parallelization
heat3d application is parallelized with _stdpar_, OpenMP, OpenACC, OpenMP4.5, Kokkos, thrust, CUDA and HIP. 
Since our main focus is _stdpar_, we briefly describe the parallel kernels and data structures. 
To represent the 3D variable ![u](https://latex.codecogs.com/svg.latex?u), we use the 3D data structure `View` consisting of 
`std::vector` and `stdex::mdspan` (see [implementation](https://github.com/yasahi-hpc/P3-miniapps/blob/main/lib/stdpar/View.hpp) for detail). 
The parallel computations are performed with `std::for_each_n` and `std::transform_reduce`.

## Data structure
We define ![u](https://latex.codecogs.com/svg.latex?u) in the following manner.
```c++
u  = RealView3D("u", std::array<size_t, 3>{nx_halo, ny_halo, nz_halo}, std::array<int, 3>{-1, -1, -1});
```
Where the second and third arguments are extents and the starting indices of this view, respectively. 
For both CPUs and GPUs, we employ Fortran style data layout (`stdex::layout_left`). 
In the host code, the data `u` can be accessed through its `operator()` like
```c++
for(int iz=0; iz<nz; iz++) {
    for(int iy=0; iy<ny; iy++) {
      for(int ix=0; ix<nx; ix++) {
        const real_type xtmp = static_cast<real_type>(ix - nx/2) * dx;
        const real_type ytmp = static_cast<real_type>(iy - ny/2) * dy;
        const real_type ztmp = static_cast<real_type>(iz - nz/2) * dz;

        x(ix) = xtmp;
        y(iy) = ytmp;
        z(iz) = ztmp;
        u(ix, iy, iz) = conf.umax
          * cos(xtmp / Lx * 2.0 * M_PI + ytmp / Ly * 2.0 * M_PI + ztmp / Lz * 2.0 * M_PI); 
      }
    }
  }
```
We need to access the data through `mdspan` in the accelerated region. Which can be accessed by the `mdspan()` method.

## Parallel operations
Since the multi-dimensional parallel operations are not fully supported before cartesian-product in c++23, 
we have wrapped `std::for_each_n` and `std::transform_reduce` manually. 
The 3D heat equation can be performed as 
```c++
Iterate_policy<3> policy3d({0, 0, 0}, {nx, ny, nz});
Impl::for_each(policy3d, heat_functor(conf, u, un));
```
with the `heat_functor` defined by
```c++
struct heat_functor {
  using mdspan3d_type = RealView3D::mdspan_type;
  mdspan3d_type u_, un_;
  float64 coef_;
 
  heat_functor(Config &conf, RealView3D &u, RealView3D &un) {
    u_  = u.mdspan();
    un_ = un.mdspan();
 
    coef_ = conf.Kappa * conf.dt / (conf.dx*conf.dx);
  }

  void operator()(const int ix, const int iy, const int iz) const {
    un_(ix, iy, iz) = u_(ix, iy, iz)
                    + coef_ * ( u_(ix+1, iy, iz) + u_(ix-1, iy, iz)
                              + u_(ix, iy+1, iz) + u_(ix, iy-1, iz)
                              + u_(ix, iy, iz+1) + u_(ix, iy, iz-1)
                              - 6. * u_(ix, iy, iz) );
  }
};
```
As explained, we accessed the data `u` and `un` through `mdspan`, which are captured by _values_ in the functor for `std::for_each_n`.

In the end, the numerical solution is compared against this analytical solution, where we perform parallel reduction with `std::transform_reduce`. 
```c++
float64 l2loc = 0.0;
auto _u  = u.mdspan();
auto _un = un.mdspan();

auto L2norm_kernel = [=] (const int ix, const int iy, const int iz) {
  auto diff = _un(ix, iy, iz) - _u(ix, iy, iz);
  return diff * diff;
};

Iterate_policy<3> policy3d({0, 0, 0}, {conf.nx, conf.ny, conf.nz});
Impl::transform_reduce(policy3d, std::plus<float64>(), L2norm_kernel, l2loc);
```

For both cases, we firstly define the parallel operations over `nx * ny * nz`. The 3D indices are computed from flattend 1D index manually inside `Impl::for_each` and `Impl::transform_reduce`. 

# Run
## heat3d
### Example Compile Command (with stdpar for Nvidia GPUs)
```
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DPROGRAMMING_MODEL=STDPAR -DBACKEND=CUDA -DAPPLICATION=heat3d ..
cmake --build .
```

### Example Run Command
```./heat3d --nx 512 --ny 512 --nz 512 --nbiter 50000 --freq_diag 1000```

### Input parameters
You can set the problem size (resolution), the number of iteration and the frequency of diagnostics.  
- `nx`: The number of grid points in x direction. (default: 128)  
- `ny`: The number of grid points in y direction. (default: 128)    
- `nz`: The number of grid points in z direction. (default: 128)  
- `nbiter`: The number of iterations of simulations. (default: 1000)  
- `freq_diag`: The diagnostics is made for every `freq_diag` step. (default: 10)

For example, we have executed the following command to make the animation.  
```./heat3d --nx 512 --ny 512 --nz 512 --nbiter 50000 --freq_diag 1000```

### Expected output (stdout)
With the run command 
```bash
./heat3d --nx 512 --ny 512 --nz 512 --nbiter 1000 --freq_diag 0
```
, you will get the following standard output (performed on V100 GPUs).
```
(nx, ny, nz) = 512, 512, 512

L2_norm: 0.00355178
Programming model: stdpar
Backend: CUDA
Elapsed time: 6.53835 [s]
Bandwidth: 328.444 [GB/s]
Flops: 184.75 [GFlops]

total 6.53835 [s], 1 calls
MainLoop 6.53832 [s], 1000 calls
Heat 6.53817 [s], 1000 calls
IO 0 [s], 0 calls
```

## heat3d_mpi
### Example Compile Command (with stdpar for Nvidia GPUs)
```
mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_BUILD_TYPE=Release -DPROGRAMMING_MODEL=STDPAR -DBACKEND=CUDA -DAPPLICATION=heat3d_mpi ..
cmake --build .
```

### Example Run Command (in a job script)
To enforce the device to device MPI communication, it is important to set the environmental variable 
```bash
export UCX_RNDV_FRAG_MEM_TYPE=cuda
```
This environmental variable is avilable with HPC-X (OpenMPI 4.1.4 + UCX 1.13.0) under [Nvidia HPC SDK v22.5](https://docs.nvidia.com/hpc-sdk/archive/22.5/index.html) (or later). The run command in a job script is as follows:
```bash
export UCX_RNDV_FRAG_MEM_TYPE=cuda
mpirun -np 2 ./wrapper.sh ../build/heat3d_mpi/stdpar/heat3d_mpi --px 2 --py 2 --pz 2 --nx 256 --ny 256 --nz 256 --nbiter 1000 --freq_diag 10
```  

GPU mapping inside a node should be made before ```MPI_Init``` with ```wrapper.sh``` (for OpenMPI).
```bash
#!/bin/sh

NGPUS=`nvidia-smi -L | wc -l`
export CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_LOCAL_RANK % NGPUS))
exec $*
```

### Input parameters
You can set the problem size (resolution), the MPI domain decomposition, the number of iteration and the frequency of diagnostics.  
- `nx`: The number of grid points in x direction. (default: 128)  
- `ny`: The number of grid points in y direction. (default: 128)    
- `nz`: The number of grid points in z direction. (default: 128)  
- `px`: The number of MPI processes in x direction. (default: 2)  
- `py`: The number of MPI processes in y direction. (default: 2)    
- `pz`: The number of MPI processes in z direction. (default: 2)  
- `nbiter`: The number of iterations of simulations. (default: 1000)  
- `freq_diag`: The diagnostics is made for every `freq_diag` step. (default: 10)

Contrary to the `heat3d` case, `nx`, `ny`, and `nz` are the number of grid points in each MPI process. In other words,
the total number of grid points in each direction are `nx * px`, `ny * py`, and `nz * pz`.
It should be noted that the total MPI processes `nb_procs` must be eqaul to `px * py * pz`.
If we set `px = 1`, then MPI communication along x direction is suppressed and replaced by swapping halo regions. 

### Expected output (stdout)
With the run command 
```bash
./heat3d_mpi --px 1 --py 1 --pz 2 --nx 512 --ny 512 --nz 256 --nbiter 1000 --freq_diag 0
```
, you will get the following standard output (performed on V100 GPUs). 

```
Parallelization (px, py, pz) = 1, 1, 2
Local (nx, ny, nz) = 512, 512, 256
Global (nx, ny, nz) = 512, 512, 512

L2_norm: 0.00355178
STDPAR backend
Elapsed time: 3.62221 [s]
Bandwidth/GPU: 296.433 [GB/s]
Flops/GPU: 166.744 [GFlops]

total 3.62221 [s], 1 calls
MainLoop 3.62217 [s], 1000 calls
Heat 3.20131 [s], 1000 calls
HaloPack 0.101119 [s], 3000 calls
HaloUnpack 0.172946 [s], 3000 calls
HaloComm 0.146246 [s], 3000 calls
IO 0 [s], 0 calls
```


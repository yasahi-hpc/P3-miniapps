# P3-miniapps
P3-miniapps are designed to evalute Performance, Portability and Productivity (P3) of mini-applications with C++ parallel algorithms (stdpar). 
We have implemented 3D heat equation solver and 4D (2D space and 2D velocity space) Vlasov-Poisson solver. These mini-apps are parallelized with MPI + "X" programming model in C++. "X" includes _stdpar_, OpenMP, OpenACC, OpenMP4.5, [Kokkos](https://github.com/kokkos/kokkos), [thrust](https://github.com/NVIDIA/thrust), CUDA, and HIP. As well as the language standard parallelization (stdpar), we also focus on the language standard high dimensional array support [_mdspan_](https://github.com/kokkos/mdspan). 

For questions or comments, please find us in the AUTHORS file.

# Usage
## Preparation
Firstly, you need to git clone on your environment as
```
git clone https://github.com/yasahi-hpc/P3-miniapps.git
```

In order to try [Kokkos](https://github.com/kokkos/kokkos) version, you need to install Kokkos on your environment following the [instructions](https://github.com/kokkos/kokkos). Please make sure that we only support the Kokkos installation through CMake.

## Compile
We rely on CMake to build the applications. 4 mini applications `heat3d`, `heat3d_mpi`, `vlp4d`, and `vlp4d_mpi` are provided. You will compile with the following CMake command. For `-DAPPLICATION` option, `<app_name>` should be choosen from the application names provided above.
```bash
cmake -DCMAKE_CXX_COMPILER=<compiler_name> \
      -DCMAKE_BUILD_TYPE=<build_type> \
      -DPROGRAMMING_MODEL=<programming_model> \
      -DBACKEND=<backend> \
      -DAPPLICATION=<app_name>
```

|  DEVICE |  programming_model  |  compiler_name  | backend  | 
| :-: | :-: | :-: | :-: |
|  IceLake  | OPENMP <br> THRUST <br> KOKKOS <br> STDPAR  | icpc <br> NONE <br> icpc <br> nvc++ | OPENMP | 
|  P100 <br> V100 <br> A100 | CUDA <br> OPENMP <br> OPENACC <br> THRUST <br> KOKKOS <br> STDPAR | NONE <br> nvc++ <br> nvc++ <br> NONE <br> nvcc_wrapper <br> nvc++ | CUDA |
|  MI100 | HIP <br> OPENMP <br> THRUST <br> KOKKOS | hipcc <br> clang++ <br> hipcc <br> hipcc | HIP |

## Run
To run the applications, several command line arguments are necessary. Following table includes the list of commad line arguments for each mini-app. 
| app_name | Arguments | Examples |
| --- | --- | --- |
| heat3d | nx, ny, nz, nbiter | ```heat3d --nx 512 --ny 512 --nz 512 --nbiter 1000``` |
| heat3d_mpi | px, py, pz, nx, ny, nz, nbiter | ```heat3d_mpi --px 2 --py 2 --pz 2 --nx 256 --ny 256 --nz 256 --nbiter 1000``` |
| vlp4d | (file_name) | ```vlp4d SLD10_large.dat``` |
| vlp4d_mpi | f | ```vlp4d_mpi -f SLD10.dat``` |

For the Kokkos version, you additionally need to give "num_threads", "teams", "device", "num_gpus", and "device_map", e.g. ```vlp4d_mpi --num_threads 1 --teams 1 --device 0 --num_gpus 8 --device_map 1 -f SLD10.dat```.

# Citations
```bibtex
@INPROCEEDINGS{Asahi2021, 
      author={Asahi, Yuuichi and Latu, Guillaume and Bigot, Julien and Grandgirard, Virginie},
      booktitle={2021 International Workshop on Performance, Portability and Productivity in HPC (P3HPC)},
      title={Optimization strategy for a performance portable Vlasov code},
      year={2021},
      volume={},
      number={},
      pages={79-91},
      doi={10.1109/P3HPC54578.2021.00011}}
```

```bibtex
@INPROCEEDINGS{Asahi2019,
    author = {Asahi, Yuuichi and Latu, Guillaume and Grandgirard, Virginie and Bigot, Julien}, 
    title = {Performance Portable Implementation of a Kinetic Plasma Simulation Mini-App}, 
    booktitle = {Accelerator Programming Using Directives}, 
    year = {2020},
    editor = {Wienke, Sandra and Bhalachandra, Sridutt}, 
    series = {series},
    pages = {117--139},
    address = {Cham},
    publisher = {Springer International Publishing}, 
}
```

# P3-miniapps
P3-miniapps are designed to evalute Performance, Portability and Productivity (P3) of mini-applications with C++ parallel algorithms (stdpar). 
We have implemented 3D heat equation solver and 4D (2D space and 2D velocity space) Vlasov-Poisson solver. These mini-apps are parallelized with MPI + "X" programming model in C++. "X" includes _stdpar_, OpenMP, OpenACC, OpenMP4.5, [Kokkos](https://github.com/kokkos/kokkos), [Thrust](https://github.com/NVIDIA/thrust), CUDA, and HIP. As well as the language standard parallelization (stdpar), we also focus on the language standard high dimensional array support [_mdspan_](https://github.com/kokkos/mdspan). 

This repository includes the following mini-apps:
* [heat3d](docs/heat3d.md)
* [heat3d_mpi](docs/heat3d.md)
* [vlp4d](docs/vlp4d.md)
* [vlp4d_mpi](docs/vlp4d.md)

For questions or comments, please find us in the AUTHORS file.

# Usage
## Preparation
Firstly, you need to git clone on your environment as
```
git clone --recursive https://github.com/yasahi-hpc/P3-miniapps.git
```

This software relies on external libraries including [Kokkos](https://github.com/kokkos/kokkos), [mdspan](https://github.com/kokkos/mdspan) and 
[fftw](http://www.fftw.org). Kokkos and mdspan are included as submodules. CUDA-Aware-MPI or ROCm-Aware-MPI are also needed for Nvidia and AMD GPUs. 
In the following, we assume that fftw and MPI libraries are appropriately installed.

## Compile
We rely on CMake to build the applications. 4 mini applications `heat3d`, `heat3d_mpi`, `vlp4d`, and `vlp4d_mpi` are provided. You will compile with the following CMake command. For `-DAPPLICATION` option, `<app_name>` should be choosen from the application names provided above. To enable test, you should set `-DBUILD_TESTING=ON`. Following table summarizes the allowed combinations of `<programming_model>`, `<compiler_name>`, and `<backend>` for each `DEVICE`.
```bash
cmake -DCMAKE_CXX_COMPILER=<compiler_name> \
      -DCMAKE_BUILD_TYPE=<build_type> \
      -DBUILD_TESTING=OFF \
      -DPROGRAMMING_MODEL=<programming_model> \
      -DBACKEND=<backend> \
      -DAPPLICATION=<app_name> 
```

|  DEVICE |  programming_model  |  compiler_name  | backend  | 
| :-: | :-: | :-: | :-: |
|  IceLake  | OPENMP <br> THRUST <br> KOKKOS <br> STDPAR  | icpc <br> g++ <br> icpc <br> nvc++ | OPENMP | 
|  P100 <br> V100 <br> A100 | CUDA <br> OPENMP <br> OPENACC <br> THRUST <br> KOKKOS <br> STDPAR | g++ <br> nvc++ <br> nvc++ <br> g++ <br> nvcc_wrapper <br> nvc++ | CUDA |
|  MI100 | HIP <br> OPENMP <br> THRUST <br> KOKKOS | hipcc <br> hipcc <br> hipcc <br> hipcc | HIP |

* For Nvidia Devices, you need to add ```-DCMAKE_CUDA_ARCHITECTURES=<device>``` (`device`=`60`, `70`, `80` for P100, V100 and A100, respetively) to compile for `<programmind_model>=[THRUST, CUDA]`. For `<programmind_model>=KOKKOS` and `<backend>=CUDA`, one further needs to add Kokkos options for installation like
```bash
-DKokkos_ENABLE_CUDA=On \
-DKokkos_ENABLE_CUDA_LAMBDA=On \
-DKokkos_ARCH_AMPERE80=On
```

* For AMD Devices, you also need to pass ```-DCMAKE_HIP_ARCHITECTURES=<device>``` (`device`=`gfx908` for MI100) to compile for `<programming_model>=OPENMP` and `<backend>=HIP`. For `<programmind_model>=KOKKOS` and `<backend>=HIP`, one further needs to add Kokkos options for installation like
```bash
-DKokkos_ENABLE_HIP=On \
-DKokkos_ARCH_VEGA908=On
```

## Run
To run the applications, several command line arguments are necessary. Following table summarizes the list of commad line arguments for each mini-app. 
| app_name | Arguments | Examples |
| --- | --- | --- |
| heat3d | nx, ny, nz, nbiter, freq_diag | ```heat3d --nx 512 --ny 512 --nz 512 --nbiter 1000 --freq_diag 0``` |
| heat3d_mpi | px, py, pz, nx, ny, nz, nbiter, freq_diag | ```heat3d_mpi --px 2 --py 2 --pz 2 --nx 256 --ny 256 --nz 256 --nbiter 1000 --freq_diag 0``` |
| vlp4d | (file_name) | ```vlp4d SLD10_large.dat``` |
| vlp4d_mpi | f | ```vlp4d_mpi -f SLD10.dat``` |

For the Kokkos version, you additionally need to give "num_threads", "teams", "device", "num_gpus", and "device_map", e.g. ```vlp4d_mpi --num_threads 1 --teams 1 --device 0 --num_gpus 8 --device_map 1 -f SLD10.dat```.

More details are given in the docs for [heat3d](docs/heat3d.md) and [vlp4d](docs/vlp4d.md).

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

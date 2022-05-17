# About


# Usage
## Preparation
Firstly, you need to git clone on your environment as
```
git clone https://github.com/yasahi-hpc/P3-miniapps.git
```

## Compile
We rely on CMake to build the applications. You will compile with the following CMake command.
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

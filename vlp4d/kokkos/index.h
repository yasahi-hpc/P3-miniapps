#ifndef __INDEX_H__
#define __INDEX_H__

#include <Kokkos_Core.hpp>

namespace Index {
  #if defined( KOKKOS_ENABLE_CUDA ) || defined( KOKKOS_ENABLE_HIP )
    // For Layout left (Fortan layout)
    KOKKOS_INLINE_FUNCTION 
    int2 int2coord_2D(int i, int n1, int n2) {
      int j1 = i%n1, j2 = i/n1;
      return make_int2(j1, j2);
    }
  
    KOKKOS_INLINE_FUNCTION 
    int3 int2coord_3D(int i, int n1, int n2, int n3) {
      int j12 = i%(n1*n2), j3 = i/(n1*n2);
      int j1 = j12%n1, j2 = j12/n1;
      return make_int3(j1, j2, j3);
    }
    
    KOKKOS_INLINE_FUNCTION 
    int4 int2coord_4D(int i, int n1, int n2, int n3, int n4) {
      int j123 = i%(n1*n2*n3), j4 = i/(n1*n2*n3);
      int j12 = j123%(n1*n2), j3 = j123/(n1*n2);
      int j1 = j12%n1, j2 = j12/n1;
      return make_int4(j1, j2, j3, j4);
    }
  #else
    // For Layout right (C layout)
    KOKKOS_INLINE_FUNCTION 
    int2 int2coord_2D(int i, int n1, int n2) {
      int j1 = i/n2, j2 = i%n2;
      return make_int2(j1, j2);
    }
  
    KOKKOS_INLINE_FUNCTION 
    int3 int2coord_3D(int i, int n1, int n2, int n3) {
      int j12 = i/n3, j3 = i%n3;
      int j1 = j12/n2, j2 = j12%n2;
      return make_int3(j1, j2, j3);
    }
    
    KOKKOS_INLINE_FUNCTION 
    int4 int2coord_4D(int i, int n1, int n2, int n3, int n4) {
      int j123 = i/n4, j4 = i%n4;
      int j12 = j123/n3, j3 = j123%n3;
      int j1 = j12/n2, j2 = j12%n2;
      return make_int4(j1, j2, j3, j4);
    }
  #endif
};

#endif

#ifndef __TILES_HPP__
#define __TILES_HPP__

#if defined( KOKKOS_ENABLE_CUDA ) 
  constexpr int TILE_SIZE0 = 32;
  constexpr int TILE_SIZE1 = 4;
  constexpr int TILE_SIZE2 = 2;
  constexpr int TILE_SIZE3 = 1;
#elif defined( KOKKOS_ENABLE_HIP )
  constexpr int TILE_SIZE0 = 64;
  constexpr int TILE_SIZE1 = 4;
  constexpr int TILE_SIZE2 = 2;
  constexpr int TILE_SIZE3 = 1;
#else
  constexpr int TILE_SIZE0 = 4;
  constexpr int TILE_SIZE1 = 4;
  constexpr int TILE_SIZE2 = 4;
  constexpr int TILE_SIZE3 = 4;
#endif

#endif

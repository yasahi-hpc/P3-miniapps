#include <gtest/gtest.h>
#include "Types.hpp"
#include "Transpose.hpp"

TEST( TRANSPOSE, EVEN ) {
  const int rows = 16;
  const int cols = rows * 2;
  double eps = 1.e-8;
   
  RealView2D a("a", rows, cols);
  RealView2D b("b", rows, cols);
  RealView2D c("c", cols, rows);
  RealView2D d("d", cols, rows);
 
  // Reference
  for(int i=0; i<cols; i++) {
    for(int j=0; j<rows; j++) {
      a(j, i) = i * 0.12 + j * 0.5;
      c(i, j) = a(j, i);
    }
  }
  a.updateDevice();

  // Execute transpose on CPUs or GPUs
  using value_type = RealView2D::value_type;
  using layout_type = RealView2D::layout_type;
  Impl::Transpose<value_type, layout_type> transpose(rows, cols);
  transpose.forward(a.data(), d.data());
  transpose.backward(d.data(), b.data());
 
  b.updateSelf();
  d.updateSelf();

  // Test
  for(int i=0; i<cols; i++) {
    for(int j=0; j<rows; j++) {
      ASSERT_NEAR( b(j, i), a(j, i), eps );
      ASSERT_NEAR( d(i, j), c(i, j), eps );
    }
  }
}

TEST( TRANSPOSE, ODD ) {
  const int rows = 15;
  const int cols = rows * 2;
  double eps = 1.e-8;
   
  RealView2D a("a", rows, cols);
  RealView2D b("b", rows, cols);
  RealView2D c("c", cols, rows);
  RealView2D d("d", cols, rows);
 
  // Reference
  for(int i=0; i<cols; i++) {
    for(int j=0; j<rows; j++) {
      a(j, i) = i * 0.12 + j * 0.5;
      c(i, j) = a(j, i);
    }
  }
  a.updateDevice();

  // Execute transpose on CPUs or GPUs
  using value_type = RealView2D::value_type;
  using layout_type = RealView2D::layout_type;
  Impl::Transpose<value_type, layout_type> transpose(rows, cols);
  transpose.forward(a.data(), d.data());
  transpose.backward(d.data(), b.data());
 
  b.updateSelf();
  d.updateSelf();

  // Test
  for(int i=0; i<cols; i++) {
    for(int j=0; j<rows; j++) {
      ASSERT_NEAR( b(j, i), a(j, i), eps );
      ASSERT_NEAR( d(i, j), c(i, j), eps );
    }
  }
}

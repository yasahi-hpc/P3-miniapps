#ifndef __EFIELD_HPP__
#define __EFIELD_HPP__

#include "config.hpp"
#include "types.hpp"
#include "FFT.hpp"
#include "Utils.hpp"

struct Efield {
  using value_type = RealView2D::value_type;
  using layout_type = RealView2D::layout_type;

  RealView2D rho_;
  RealView2D ex_;
  RealView2D ey_;
  RealView2D phi_;

  Impl::FFT<value_type, layout_type> *fft_;

private:
  RealView1D filter_; // [YA Added] In order to avoid conditional to keep (0, 0) component 0

  // 2D complex buffers of size nx1h * nx2 (renamed)
  ComplexView2D rho_hat_;
  ComplexView2D ex_hat_;
  ComplexView2D ey_hat_;
  
public:
  Efield(Config *conf, shape_nd<2> dim);
  virtual ~Efield();
  
  void solve_poisson_fftw(float64 xmax, float64 ymax);

private:
  DISALLOW_COPY_AND_ASSIGN(Efield);
};

#endif

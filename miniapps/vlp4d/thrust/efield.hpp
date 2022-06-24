#ifndef __EFIELD_HPP__
#define __EFIELD_HPP__

#include "types.hpp"
#include "config.hpp"
#include "FFT.hpp"
#include "utils.hpp"

struct Efield {
  using value_type = RealView2D::value_type;
  using layout_type = RealView2D::layout_type;

  RealView2D rho_;
  RealView2D ex_;
  RealView2D ey_;
  RealView2D phi_;

  RealView1D filter_; // [YA added] In order to avoid conditional to keep (0, 0) component 0

  Impl::FFT<value_type, layout_type> *fft_;

  // a 2D complex buffer of size nx1h * nx2 (renamed)
private:
  ComplexView2D rho_hat_;
  ComplexView2D ex_hat_;
  ComplexView2D ey_hat_;

public:
  Efield(Config *conf, shape_nd<2> dim);
  virtual ~Efield();

  void solve_poisson_fftw(double xmax, double ymax);

private:
  DISALLOW_COPY_AND_ASSIGN(Efield);
  
};

#endif

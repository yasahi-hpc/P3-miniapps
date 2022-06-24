#ifndef __EFIELD_HPP__
#define __EFIELD_HPP__

#include "Types.hpp"
#include "Config.hpp"
#include "FFT.hpp"

struct Efield {
  using array_layout = typename RealView2D::array_layout;
  RealView2D rho_;
  RealView2D rho_loc_; // Before all reduce
  RealView2D ex_;
  RealView2D ey_;
  RealView2D phi_;

  // Filter to avoid conditional to keep (0, 0) component 0
  RealView1D filter_;

  Impl::FFT<float64, array_layout> *fft_;

  // a 2D complex buffer of size nx1h * nx2 (renamed)
private:
  ComplexView2D rho_hat_;
  ComplexView2D ex_hat_;
  ComplexView2D ey_hat_;

public:
  Efield(Config *conf, shape_t<2> dim);
  virtual ~Efield();

  void solve_poisson_fftw(float64 xmax, float64 ymax);
};

#endif

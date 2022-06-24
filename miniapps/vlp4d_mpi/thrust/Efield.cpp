#include "Efield.hpp"
#include "Parallel_For.hpp"

Efield::Efield(Config *conf, shape_nd<2> dim)
  : fft_(nullptr) {
  rho_     = RealView2D("rho",     dim[0], dim[1]);
  rho_loc_ = RealView2D("rho_loc", dim[0], dim[1]);
  ex_      = RealView2D("ex",      dim[0], dim[1]);
  ey_      = RealView2D("ey",      dim[0], dim[1]);
  phi_     = RealView2D("phi",     dim[0], dim[1]);

  rho_.fill(0.);
  rho_loc_.fill(0.);
  ex_.fill(0.);
  ey_.fill(0.);
  phi_.fill(0.);

  // Initialize fft helper
  const Domain *dom = &(conf->dom_);
  int nx = dom->nxmax_[0];
  int ny = dom->nxmax_[1];
  float64 xmax = dom->maxPhy_[0];
  float64 kx0 = 2 * M_PI / xmax;

  fft_ = new FFT_type(nx, ny, 1);
  int nx1h = fft_->nx1h();
  rho_hat_ = ComplexView2D("rho_hat", nx1h, ny);
  ex_hat_  = ComplexView2D("ex_hat",  nx1h, ny);
  ey_hat_  = ComplexView2D("ey_hat",  nx1h, ny);

  rho_hat_.fill(0.);
  ex_hat_.fill(0.);
  ey_hat_.fill(0.);

  filter_  = RealView1D("filter", nx1h);
  
  // Initialize filter (0,1/k2,1/k2,1/k2,...)
  // In order to avoid zero division in vectorized way
  // filter[0] == 0, and filter[0:] == 1./(ix1*kx0)**2
  filter_(0) = 0.;
  for(int ix = 1; ix < nx1h; ix++) {
    float64 kx = ix * kx0;
    float64 k2 = kx * kx;
    filter_(ix) = 1./k2;
  }
  filter_.updateDevice();
}

Efield::~Efield() {
  if(fft_ != nullptr) delete fft_;
}

void Efield::solve_poisson_fftw(float64 xmax, float64 ymax) {
  float64 kx0 = 2 * M_PI / xmax;
  float64 ky0 = 2 * M_PI / ymax;
  int nx1  = fft_->nx1();
  int nx1h = fft_->nx1h();
  int nx2  = fft_->nx2();
  int nx2h = fft_->nx2h();
  float64 normcoeff = fft_->normcoeff();
  const complex128 I = complex128(0., 1.);

  auto _filter  = filter_.mdspan();
  auto _ex_hat  = ex_hat_.mdspan();
  auto _ey_hat  = ey_hat_.mdspan();
  auto _rho_hat = rho_hat_.mdspan();

  // Forward 2D FFT (Real to complex)
  fft_->rfft2(rho_.data(), rho_hat_.data());

  // Solve Poisson equation in Fourier space
  auto solve_poisson = [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix1) {
    float64 kx = ix1 * kx0;
    {
      int ix2 = 0;
      float64 kx = ix1 * kx0;
      _ex_hat(ix1, ix2) = -kx * I * _rho_hat(ix1, ix2) * _filter(ix1) * normcoeff;
      _ey_hat(ix1, ix2) = 0.;
      _rho_hat(ix1, ix2) = _rho_hat(ix1, ix2) * _filter(ix1) * normcoeff;
    }

    for(int ix2=1; ix2<nx2h; ix2++) {
      float64 ky = ix2 * ky0;
      float64 k2 = kx * kx + ky * ky;

      _ex_hat(ix1, ix2) = -(kx/k2) * I * _rho_hat(ix1, ix2) * normcoeff;
      _ey_hat(ix1, ix2) = -(ky/k2) * I * _rho_hat(ix1, ix2) * normcoeff;
      _rho_hat(ix1, ix2) = _rho_hat(ix1, ix2) / k2 * normcoeff;
    }

    for(int ix2=nx2h; ix2<nx2; ix2++) {
      float64 ky = (ix2-nx2) * ky0;
      float64 k2 = kx*kx + ky*ky;

      _ex_hat(ix1, ix2) = -(kx/k2) * I * _rho_hat(ix1, ix2) * normcoeff;
      _ey_hat(ix1, ix2) = -(ky/k2) * I * _rho_hat(ix1, ix2) * normcoeff;
      _rho_hat(ix1, ix2) = _rho_hat(ix1, ix2) / k2 * normcoeff;
    }
  };

  Impl::for_each(nx1h, solve_poisson);

  // Backward 2D FFT (Complex to Real)
  fft_->irfft2(rho_hat_.data(), rho_.data());
  fft_->irfft2(ex_hat_.data(),  ex_.data());
  fft_->irfft2(ey_hat_.data(),  ey_.data());
}

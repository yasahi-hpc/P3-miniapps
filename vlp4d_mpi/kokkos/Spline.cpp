#include "Spline.hpp"

void Spline::computeCoeff_xy(Config *conf, RealOffsetView4D fn,
                             const std::vector<int> &tiles) {

  const int TX = tiles[0], TY = tiles[1];
  MDPolicy<2> spline_xy_policy2d({{0,  0}},
                                 {{nvx_, nvy_}},
                                 {{TX, TY}}
                                );
  #if defined ( LAYOUT_LEFT )
    Kokkos::parallel_for("spline_coef_xy", spline_xy_policy2d, spline_coef_2d(conf, fn));
  #else
    transpose_->forward(fn.data(), fn_trans_.data());
    Kokkos::parallel_for("spline_coef_xy", spline_xy_policy2d, spline_coef_2d(conf, fn_trans_));
    transpose_->backward(fn_trans_.data(), fn.data()); 
  #endif
  Kokkos::fence();
}

void Spline::computeCoeff_vxvy(Config *conf, RealOffsetView4D fn,
                               const std::vector<int> &tiles) {
  const int TX = tiles[0], TY = tiles[1];
   MDPolicy<2>spline_vxvy_policy2d({{0,  0}},
                                   {{nx_, ny_}},
                                   {{TX, TY}}
                                  );
  #if defined ( LAYOUT_LEFT )
    transpose_->forward(fn.data(), fn_trans_.data());
    Kokkos::parallel_for("spline_coef_vxvy", spline_vxvy_policy2d, spline_coef_2d(conf, fn_trans_));
    transpose_->backward(fn_trans_.data(), fn.data());
  #else
    Kokkos::parallel_for("spline_coef_vxvy", spline_vxvy_policy2d, spline_coef_2d(conf, fn));
  #endif
  Kokkos::fence();
}

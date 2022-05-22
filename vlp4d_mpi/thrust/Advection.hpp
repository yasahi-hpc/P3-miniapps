#ifndef __ADVECTION_HPP__
#define __ADVECTION_HPP__

#include "Config.hpp"
#include "Efield.hpp"
#include "Communication.hpp"
#include "Types.hpp"
#include "Math.hpp"
#include "Parallel_For.hpp"
#include <cmath>
using std::max; using std::min;

namespace Advection {
  // prototypes
  void advect_2D_xy(Config *conf, RealView4D &fn, RealView4D &fn_tmp, float64 dt);
  void advect_4D(Config *conf, Efield *ef, RealView4D &fn, RealView4D &tmp_fn, float64 dt);
  void print_fxvx(Config *conf, Distrib &comm, RealView4D &fn, int iter);

  // Internal functions
  static void testError(device_vector<int> &err) {
    host_vector<int> h_err = err;
    if(h_err[0] != 0) {
      fprintf(stderr, "Time step is too large, exiting\n");
      exit(0);
    }
  }

  // Internal functions
  MDSPAN_FORCE_INLINE_FUNCTION
  static inline void lag3_basis(double posx, double coef[3]) {
    coef[0] = .5  * (posx - 1.) * (posx - 2.);
    coef[1] = -1. * (posx)      * (posx - 2.);
    coef[2] = .5  * (posx)      * (posx - 1.);
  }

  MDSPAN_FORCE_INLINE_FUNCTION
  static inline void lag5_basis(double posx, double coef[5]) {
    double const loc[] = {1. / 24., -1. / 6., 1. / 4., -1. / 6., 1. / 24.};
    coef[0] = loc[0] * (posx - 1.) * (posx - 2.) * (posx - 3.) * (posx - 4.);
    coef[1] = loc[1] * (posx) * (posx - 2.) * (posx - 3.) * (posx - 4.);
    coef[2] = loc[2] * (posx) * (posx - 1.) * (posx - 3.) * (posx - 4.);
    coef[3] = loc[3] * (posx) * (posx - 1.) * (posx - 2.) * (posx - 4.);
    coef[4] = loc[4] * (posx) * (posx - 1.) * (posx - 2.) * (posx - 3.);
  }

  #if LAG_ORDER == 4
    MDSPAN_FORCE_INLINE_FUNCTION
    static inline void lag_basis(float64 posx, float64 coef[LAG_PTS]) {
      const float64 loc[] = {1. / 24., -1. / 6., 1. / 4., -1. / 6., 1. / 24.};
      coef[0] = loc[0] * (posx - 1.) * (posx - 2.) * (posx - 3.) * (posx - 4.);
      coef[1] = loc[1] * (posx) * (posx - 2.) * (posx - 3.) * (posx - 4.);
      coef[2] = loc[2] * (posx) * (posx - 1.) * (posx - 3.) * (posx - 4.);
      coef[3] = loc[3] * (posx) * (posx - 1.) * (posx - 2.) * (posx - 4.);
      coef[4] = loc[4] * (posx) * (posx - 1.) * (posx - 2.) * (posx - 3.);
    }
  #endif

  #if LAG_ORDER == 5
    MDSPAN_FORCE_INLINE_FUNCTION
    static inline void lag_basis(float64 posx, float64 coef[LAG_PTS]) {
      const float64 loc[] = {-1. / 24, 1. / 24., -1. / 12., 1. / 12., -1. / 24., 1. / 24.};
      const float64 pxm2 = px - 2.;
      const float64 sqrpxm2 = pxm2 * pxm2;
      const float64 pxm2_01 = pxm2 * (pxm2 - 1.);
                              
      coef[0] = loc[0] * pxm2_01 * (pxm2 + 1.) * (pxm2 - 2.) * (pxm2 - 1.);
      coef[1] = loc[1] * pxm2_01 * (pxm2 - 2.) * (5 * sqrpxm2 + pxm2 - 8.);
      coef[2] = loc[2] * (pxm2 - 1.) * (pxm2 - 2.) * (pxm2 + 1.) * (5 * sqrpxm2 - 3 * pxm2 - 6.);
      coef[3] = loc[3] * pxm2 * (pxm2 + 1.) * (pxm2 - 2.) * (5 * sqrpxm2 - 7 * pxm2 - 4.);
      coef[4] = loc[4] * pxm2_01 * (pxm2 + 1.) * (5 * sqrpxm2 - 11 * pxm2 - 2.);
      coef[5] = loc[5] * pxm2_01 * pxm2 * (pxm2 + 1.) * (pxm2 - 2.);
    }
  #endif

  #if LAG_ORDER == 7
    MDSPAN_FORCE_INLINE_FUNCTION
    static inline void lag_basis(float64 px, float64 coef[LAG_PTS]) {
      const float64 loc[] = {-1. / 720, 1. / 720, -1. / 240, 1. / 144, -1. / 144, 1. / 240, -1. / 720, 1. / 720};
      const float64 pxm3 = px - 3.;
      const float64 sevenpxm3sqr = 7. * pxm3 * pxm3;
      const float64 f1t3 = (px - 1.) * (px - 2.) * (px - 3.);
      const float64 f4t6 = (px - 4.) * (px - 5.) * (px - 6.);
                                                          
      coef[0] = loc[0] * f1t3 * f4t6 * (px - 4.);
      coef[1] = loc[1] * (px - 2.) * pxm3 * f4t6 * (sevenpxm3sqr + 8. * pxm3 - 18.);
      coef[2] = loc[2] * (px - 1.) * pxm3 * f4t6 * (sevenpxm3sqr + 2. * pxm3 - 15.);
      coef[3] = loc[3] * (px - 1.) * (px - 2.) * f4t6 * (sevenpxm3sqr - 4. * pxm3 - 12.);
      coef[4] = loc[4] * f1t3 * (px - 5.) * (px - 6.) * (sevenpxm3sqr - 10. * pxm3 - 9.);
      coef[5] = loc[5] * f1t3 * (px - 4.) * (px - 6.) * (sevenpxm3sqr - 16. * pxm3 - 6.);
      coef[6] = loc[6] * f1t3 * (px - 4.) * (px - 5.) * (sevenpxm3sqr - 22. * pxm3 - 3.);
      coef[7] = loc[7] * f1t3 * f4t6 * pxm3;
    }
  #endif

  void advect_2D_xy(Config *conf, RealView4D &fn, RealView4D &fn_tmp, float64 dt) {
    Domain *dom = &(conf->dom_);
    const float64 minPhyx  = dom->minPhy_[0];
    const float64 minPhyy  = dom->minPhy_[1];
    const float64 minPhyvx = dom->minPhy_[2];
    const float64 minPhyvy = dom->minPhy_[3];
    const float64 dx  = dom->dx_[0];
    const float64 dy  = dom->dx_[1];
    const float64 dvx = dom->dx_[2];
    const float64 dvy = dom->dx_[3];
    const int nx_min  = dom->local_nxmin_[0];
    const int ny_min  = dom->local_nxmin_[1];
    const int nvx_min = dom->local_nxmin_[2];  
    const int nvy_min = dom->local_nxmin_[3];
    const int nx_max  = dom->local_nxmax_[0] + 1;
    const int ny_max  = dom->local_nxmax_[1] + 1;
    const int nvx_max = dom->local_nxmax_[2] + 1;
    const int nvy_max = dom->local_nxmax_[3] + 1;

    const float64 inv_dx[2] = {1./dx, 1./dy};
    const float64 minPhy[2] = {dom->minPhy_[0], dom->minPhy_[1]};
    const int xmin[2]   = {dom->local_nxmin_[0], dom->local_nxmin_[1]};
    const int xmax[2]   = {dom->local_nxmax_[0], dom->local_nxmax_[1]};
    shape_nd<DIMENSION> shape_halo;
    shape_nd<DIMENSION> nxmin_halo;
    for(int i=0; i<DIMENSION; i++)
      nxmin_halo[i] = dom->local_nxmin_[i] - HALO_PTS;
    for(int i=0; i<DIMENSION; i++)
      shape_halo[i] = dom->local_nxmax_[i] - dom->local_nxmin_[i] + HALO_PTS*2 + 1;

    // Allocate 4D data structures with Offsets
    Impl::deep_copy(fn_tmp, fn);

    device_vector<int> err(1, 0);
    int *ptr_err = (int*)thrust::raw_pointer_cast(err.data());

    auto _fn = fn.mdspan();
    auto _fn_tmp = fn_tmp.mdspan();

    auto advect_2d = [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix, const int iy, const int ivx, const int ivy) {
      const float64 x  = minPhyx  + ix  * dx;
      const float64 y  = minPhyy  + iy  * dy;
      const float64 vx = minPhyvx + ivx * dvx;
      const float64 vy = minPhyvy + ivy * dvy;
      const float64 depx = dt * vx;
      const float64 depy = dt * vy;
      const float64 xstar[2] = {x - depx, y - depy};

      #ifdef LAG_ORDER
        #ifdef LAG_ODD
          int ipos1 = floor((xstar[0] - minPhy[0]) * inv_dx[0]);
          int ipos2 = floor((xstar[1] - minPhy[1]) * inv_dx[1]);
        #else
          int ipos1 = round((xstar[0] - minPhy[0]) * inv_dx[0]);
          int ipos2 = round((xstar[1] - minPhy[1]) * inv_dx[1]);
        #endif
        const float64 d_prev1 = LAG_OFFSET + inv_dx[0] * (xstar[0] - (minPhy[0] + ipos1 * inv_dx[0]));
        const float64 d_prev2 = LAG_OFFSET + inv_dx[1] * (xstar[1] - (minPhy[1] + ipos2 * inv_dx[1]));

        float64 coefx[LAG_PTS];
        float64 coefy[LAG_PTS];
        float64 ftmp = 0.;

        ipos1 -= LAG_OFFSET;
        ipos2 -= LAG_OFFSET;
        lag_basis(d_prev1, coefx);
        lag_basis(d_prev2, coefy);

        if(ipos1 < xmin[0] - HALO_PTS || ipos1 > xmax[0] + HALO_PTS - LAG_ORDER)
          ptr_err[0] += 1;
        if(ipos2 < xmin[1] - HALO_PTS || ipos2 > xmax[1] + HALO_PTS - LAG_ORDER)
          ptr_err[0] += 1;

        for(int k2 = 0; k2 <= LAG_ORDER; k2++) {
          for(int k1 = 0; k1 <= LAG_ORDER; k1++) {
            ftmp += coefx[k1] * coefy[k2] * _fn_tmp(ipos1 + k1, ipos2 + k2, ivx, ivy);
          }
        }
      #else
        const float64 fx = (xstar[0] - minPhy[0]) * inv_dx[0];
        const float64 fy = (xstar[1] - minPhy[1]) * inv_dx[1];
        const int ipos1 = floor(fx);
        const int ipos2 = floor(fy);
        const float64 wx = fx - static_cast<float64>(ipos1);
        const float64 wy = fy - static_cast<float64>(ipos2);
 
        const float64 etax3 = (1./6.) * wx * wx * wx;
        const float64 etax0 = (1./6.) + 0.5  * wx * (wx - 1.) - etax3;
        const float64 etax2 = wx + etax0 - 2. * etax3;
        const float64 etax1 = 1. - etax0 - etax2 - etax3;
        const float64 etax[4] = {etax0, etax1, etax2, etax3};
 
        const float64 etay3 = (1./6.) * wy * wy * wy;
        const float64 etay0 = (1./6.) + 0.5 * wy * (wy - 1.) - etay3;
        const float64 etay2 = wy + etay0 - 2. * etay3;
        const float64 etay1 = 1. - etay0 - etay2 - etay3;
        const float64 etay[4] = {etay0, etay1, etay2, etay3};
 
        if(ipos1 < xmin[0] - 1 || ipos1 > xmax[0]) ptr_err[0] += 1;
        if(ipos2 < xmin[1] - 1 || ipos2 > xmax[1]) ptr_err[0] += 1;
        float64 ftmp = 0.;
 
        for(int k2 = 0; k2 <= 3; k2++) {
          float64 sum = 0.;
          for(int k1 = 0; k1 <= 3; k1++) {
            sum += etax[k1] *  _fn_tmp(ipos1-1+k1, ipos2-1+k2, ivx, ivy);
          }
          ftmp += etay[k2] * sum;
        }
      #endif

      _fn(ix, iy, ivx, ivy) = ftmp;
    };

    int4 begin = make_int4(nx_min, ny_min, nvx_min, nvy_min);
    int4 end   = make_int4(nx_max, ny_max, nvx_max, nvy_max);
    Impl::for_each<default_iterate_layout>(begin, end, advect_2d);

    #if ! defined(NO_ERROR_CHECK)
      testError(err);
    #endif
  }

  void advect_4D(Config *conf, Efield *ef, RealView4D &fn, RealView4D &tmp_fn, float64 dt) {
    Domain *dom = &(conf->dom_);
    const int nx_min  = dom->local_nxmin_[0];
    const int ny_min  = dom->local_nxmin_[1];
    const int nvx_min = dom->local_nxmin_[2];
    const int nvy_min = dom->local_nxmin_[3];
    const int nx_max  = dom->local_nxmax_[0] + 1;
    const int ny_max  = dom->local_nxmax_[1] + 1;
    const int nvx_max = dom->local_nxmax_[2] + 1;
    const int nvy_max = dom->local_nxmax_[3] + 1;

    float64 xmin[DIMENSION], inv_dx[DIMENSION];
    float64 rxwidth[DIMENSION], dx[DIMENSION];
    float64 locrxmindx[DIMENSION], locrxmaxdx[DIMENSION];
    int nxmax[DIMENSION];
    for(int j=0; j<DIMENSION; j++) {
      nxmax[j]  = dom->nxmax_[j];
      xmin[j]  = dom->minPhy_[j];
      rxwidth[j] = dom->maxPhy_[j] - dom->minPhy_[j];
      inv_dx[j] = 1. / dom->dx_[j];
      dx[j]     = dom->dx_[j];
      locrxmindx[j] = dom->minPhy_[j] + dom->local_nxmin_[j] * dom->dx_[j] - dom->dx_[j];
      locrxmaxdx[j] = dom->minPhy_[j] + dom->local_nxmax_[j] * dom->dx_[j] + dom->dx_[j];
    }
     
    device_vector<int> err(1, 0);
    int *ptr_err = (int*)thrust::raw_pointer_cast(err.data());

    Impl::deep_copy(tmp_fn, fn);

    auto _fn = fn.mdspan();
    auto _tmp_fn = tmp_fn.mdspan();
    auto _ex = ef->ex_.mdspan();
    auto _ey = ef->ey_.mdspan();

    auto computeFeet = [=] MDSPAN_FORCE_INLINE_FUNCTION (float64 xstar[DIMENSION], const int pos[DIMENSION]) {
      const int s_nxmax = nxmax[0];
      const int s_nymax = nxmax[1];
      const float64 x  = xmin[0] + pos[0] * dx[0];
      const float64 y  = xmin[1] + pos[1] * dx[1];
      const float64 vx = xmin[2] + pos[2] * dx[2];
      const float64 vy = xmin[3] + pos[3] * dx[3];
  
      float64 xtmp[DIMENSION];
      xtmp[0] = x  - 0.5 * dt * vx;
      xtmp[1] = y  - 0.5 * dt * vy;
      xtmp[2] = vx - 0.5 * dt * _ex(pos[0], pos[1]);
      xtmp[3] = vy - 0.5 * dt * _ey(pos[0], pos[1]);
  
      float64 ftmp1 = 0., ftmp2 = 0.;

      for(int count = 0; count < 1; count++) {
        int ipos[2];
        float64 coefx[2][3];
 
        for(int j = 0; j <= 1; j++) {
          xtmp[j] = inv_dx[j] * (xtmp[j] - xmin[j]);
          ipos[j] = round(xtmp[j]) - 1;
          const float64 posx = xtmp[j] - ipos[j];
          lag3_basis(posx, coefx[j]);
        }
 
        ftmp1 = 0.;
        ftmp2 = 0.;
 
        for(int k1 = 0; k1 <= 2; k1++) {
          for(int k0 = 0; k0 <= 2; k0++) {
            int ix = (s_nxmax + ipos[0] + k0) % s_nxmax;
            int iy = (s_nymax + ipos[1] + k1) % s_nymax;
            ftmp1 += coefx[0][k0] * coefx[1][k1] * _ex(ix, iy);
            ftmp2 += coefx[0][k0] * coefx[1][k1] * _ey(ix, iy);
          }
        }
 
        xtmp[2] = vx - 0.5 * dt * ftmp1;
        xtmp[3] = vy - 0.5 * dt * ftmp2;
        xtmp[0] = x  - 0.5 * dt * xtmp[2];
        xtmp[1] = y  - 0.5 * dt * xtmp[3];
      }
 
      xstar[0] = x - dt * xtmp[2];
      xstar[1] = y - dt * xtmp[3];
      xstar[2] = max(min(vx - dt * ftmp1, xmin[2] + rxwidth[2]), xmin[2]);
      xstar[3] = max(min(vy - dt * ftmp2, xmin[3] + rxwidth[3]), xmin[3]);
    };

    auto interp_4d = [=] MDSPAN_FORCE_INLINE_FUNCTION (float64 xstar[DIMENSION]) {
      int ipos[DIMENSION];
      for(int j = 0; j < DIMENSION; j++) {
        xstar[j] = inv_dx[j] * (xstar[j] - xmin[j]);
      }

      #ifdef LAG_ORDER
        float64 coefx0[LAG_PTS];
        float64 coefx1[LAG_PTS];
        float64 coefx2[LAG_PTS];
        float64 coefx3[LAG_PTS];

        ipos[0] = floor(xstar[0]) - LAG_OFFSET;
        ipos[1] = floor(xstar[1]) - LAG_OFFSET;
        ipos[2] = floor(xstar[2]) - LAG_OFFSET;
        ipos[3] = floor(xstar[3]) - LAG_OFFSET;
        lag_basis((xstar[0] - ipos[0]), coefx0);
        lag_basis((xstar[1] - ipos[1]), coefx1);
        lag_basis((xstar[2] - ipos[2]), coefx2);
        lag_basis((xstar[3] - ipos[3]), coefx3);

        float64 ftmp1 = 0.;
        for(int k3 = 0; k3 <= LAG_ORDER; k3++) {
          for(int k2 = 0; k2 <= LAG_ORDER; k2++) {
            float64 ftmp2 = 0.;

            for(int k1 = 0; k1 <= LAG_ORDER; k1++) {
              float64 ftmp3 = 0.;

              for(int k0 = 0; k0 <= LAG_ORDER; k0++) {
                ftmp3 += coefx0[k0] * _tmp_fn(ipos[0] + k0, ipos[1] + k1, ipos[2] + k2, ipos[3] + k3);
              }
              ftmp2 += ftmp3 * coefx1[k1];
            }
            ftmp1 += ftmp2 * coefx2[k2] * coefx3[k3];
          }
        }

        return ftmp1;
      #else
        //LAG_ORDER
        float64 eta[DIMENSION][DIMENSION];
        for(int j = 0; j < DIMENSION; j++) {
          ipos[j] = floor(xstar[j]);
          const float64 wx = xstar[j] - ipos[j];
          const float64 etax3 = (1./6.) * wx * wx * wx;
          const float64 etax0 = (1./6.) + 0.5 * wx * (wx - 1.) - etax3;
          const float64 etax2 = wx + etax0 - 2. * etax3;
          const float64 etax1 = 1. - etax0 - etax2 - etax3;
          eta[j][0] = etax0;
          eta[j][1] = etax1;
          eta[j][2] = etax2;
          eta[j][3] = etax3;
        }

        float64 ftmp1 = 0.;
        for(int k3 = 0; k3 <= 3; k3++) {
          for(int k2 = 0; k2 <= 3; k2++) {
            float64 ftmp2 = 0.;
            for(int k1 = 0; k1 <= 3; k1++) {
              float64 ftmp3 = 0.;

              for(int k0 = 0; k0 <= 3; k0++) {
                ftmp3 += eta[0][k0]
                       * _tmp_fn(ipos[0]+k0-1, ipos[1]+k1-1, ipos[2]+k2-1, ipos[3]+k3-1);
              }
              ftmp2 += ftmp3 * eta[1][k1];
            }
            ftmp1 += ftmp2 * eta[2][k2] * eta[3][k3];
          }
        }

        return ftmp1;
      #endif
    };

    auto advect_4d = [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix, const int iy, const int ivx, const int ivy) {
      float64 xstar[DIMENSION];
      int indices[4] = {ix, iy, ivx, ivy};

      computeFeet(xstar, indices);
      #if ! defined(NO_ERROR_CHECK)
        for(int j = 0; j < DIMENSION; j++) {
          ptr_err[0] += (xstar[j] < locrxmindx[j] || xstar[j] > locrxmaxdx[j]);
        }
      #endif

      _fn(ix, iy, ivx, ivy) = interp_4d(xstar);
    };

    int4 begin = make_int4(nx_min, ny_min, nvx_min, nvy_min);
    int4 end   = make_int4(nx_max, ny_max, nvx_max, nvy_max);
    Impl::for_each<default_iterate_layout>(begin, end, advect_4d);

    #if ! defined(NO_ERROR_CHECK)
      testError(err);
    #endif
  }

  void print_fxvx(Config *conf, Distrib &comm, RealView4D &fn, int iter) {
    Domain *dom = &(conf->dom_);
    const int nx  = dom->nxmax_[0], ny  = dom->nxmax_[1];
    const int nvx = dom->nxmax_[2], nvy = dom->nxmax_[3];
    const int nxmin  = dom->local_nxmin_[0], nymin  = dom->local_nxmin_[1];
    const int nvxmin = dom->local_nxmin_[2], nvymin = dom->local_nxmin_[3];
    const int nxmax  = dom->local_nxmax_[0], nymax  = dom->local_nxmax_[1];
    const int nvxmax = dom->local_nxmax_[2], nvymax = dom->local_nxmax_[3];
    RealView2D fnxvx("fnxvx", nx, nvx);
    RealView2D fnxvxres("fnxvxres", nx, nvx);
    for(int ivx = 0; ivx < nvx; ivx++)  {
      for(int ix = 0; ix < nx; ix++)  {
        fnxvx(ix, ivx)    = 0.;
        fnxvxres(ix, ivx) = 0.;
      }
    }

    RealView2D fnyvy("fnyvy", ny, nvy);
    RealView2D fnyvyres("fnyvyres", ny, nvy);
    for(int ivy = 0; ivy < nvy; ivy++)  {
      for(int iy = 0; iy < ny; iy++)  {
        fnyvy(iy, ivy)    = 0.;
        fnyvyres(iy, ivy) = 0.;
      }
    }

    fn.updateSelf();

    // At (iy, ivy) = (0, nvy/2) cross section
    const int iy  = 0;
    const int ivy = ny / 2;
    if(nvymin <= ivy && ivy <= nvymax && 
       nymin <= iy && iy && nymax) {
      for(int ivx = nvxmin; ivx <= nvxmax; ivx++) {
        for(int ix = nxmin; ix <= nxmax; ix++) {
          fnxvx(ix, ivx) = fn(ix, iy, ivx, ivy);
        }
      }
    }

    // At (ix, ivx) = (0, nvx/2) cross section
    const int ix = 0;
    const int ivx = nvx/2;
    if(nvxmin <= ivy && ivy <= nvymax && 
       nxmin <= iy && iy && nymax) {
      for(int ivy = nvymin; ivy <= nvymax; ivy++) {
        for(int iy = nymin; iy <= nymax; iy++) {
          fnyvy(iy, ivy) = fn(ix, iy, ivx, ivy);
        }
      }
    }

    int nelems = nx * nvx;
    MPI_Reduce(fnxvx.data(), fnxvxres.data(), nelems, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    nelems = ny * nvy;
    MPI_Reduce(fnyvy.data(), fnyvyres.data(), nelems, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(comm.master()) {
      char filename[128];
      printf("print_fxvx %d\n",  iter);

      {
        sprintf(filename, "data/vlp4d_mpi/fxvx_it%06d.csv", iter);
        FILE* fileid = fopen(filename, "w");

        for(int ivx = 0; ivx < dom->nxmax_[2]; ivx++) {
          for(int ix = 0; ix < dom->nxmax_[0]; ix++) {
            if(ix == dom->nxmax_[0]-1) {
              fprintf(fileid, "%20.13le\n", fnxvxres(ix, ivx));
            } else {
              fprintf(fileid, "%20.13le, ", fnxvxres(ix, ivx));
            }
          }
        }
        fclose(fileid);
      }

      {
        sprintf(filename, "data/vlp4d_mpi/fyvy_it%06d.csv", iter);
        FILE* fileid = fopen(filename, "w");

        for(int ivy = 0; ivy < dom->nxmax_[3]; ivy++) {
          for(int iy = 0; iy < dom->nxmax_[1]; iy++) {
            if(iy == dom->nxmax_[1]-1) {
              fprintf(fileid, "%20.13le\n", fnyvyres(iy, ivy));
            } else {
              fprintf(fileid, "%20.13le, ", fnyvyres(iy, ivy));
            }
          }
        }
        fclose(fileid);
      }
    }
  }
};

#endif

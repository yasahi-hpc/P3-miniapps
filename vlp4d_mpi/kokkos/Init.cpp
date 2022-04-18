#include "Init.hpp"
#include "Math.hpp"

// Prototypes
void import(const char *f, Config *conf);
void print(Config *conf);
void initcase(Config *conf, RealOffsetView4D fn);
void testcaseCYL02(Config* conf, RealOffsetView4D fn);
void testcaseCYL05(Config* conf, RealOffsetView4D fn);
void testcaseSLD10(Config* conf, RealOffsetView4D fn);
void testcaseTSI20(Config* conf, RealOffsetView4D fn);

void import(const char *f, Config *conf) {
  char idcase[8], tmp;
  Physics* phys = &(conf->phys_);
  Domain* dom = &(conf->dom_);
  FILE* stream = fopen(f, "r");

  if(stream == (FILE*)NULL) {
    printf("import: error file not found\n");
    abort();
  }

  /*Physical parameters */
  phys->eps0_ = 1.; /* permittivity of free space */
  phys->echarge_ = 1.; /* charge of particle */
  phys->mass_ = 1; /* mass of particle */
  phys->omega02_ = 0; /* coefficient of applied field */
  phys->vbeam_ = 0.; /* beam velocity */
  phys->S_ = 1; /* lattice period */
  phys->psi_ = 0;

  for(int i = 0; i < DIMENSION; i++) {
    do
      tmp = fgetc(stream);
    while(tmp != ':');
    fscanf(stream, " %d\n", &(dom->nxmax_[i]));
  }

  for(int i = 0; i < DIMENSION; i++) {
    do
      tmp = fgetc(stream);
    while(tmp != ':');
    fscanf(stream, " %le\n", &(dom->minPhy_[i]));
    do
      tmp = fgetc(stream);
    while(tmp != ':');
    fscanf(stream, " %le\n", &(dom->maxPhy_[i]));
  }

  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fgets(idcase, 7, stream);

  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fscanf(stream, " %le\n", &(dom->dt_));

  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fscanf(stream, " %d\n", &(dom->nbiter_));
  
  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fscanf(stream, " %d\n", &(dom->ifreq_));
   
  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fscanf(stream, " %d\n", &(dom->fxvx_));

  dom->idcase_ = atoi(idcase);
  
  for(int i = 0; i < DIMENSION; i++) {
    dom->dx_[i] = (dom->maxPhy_[i] - dom->minPhy_[i]) / dom->nxmax_[i];
  }
  
  fclose(stream);
};

void print(Config *conf) {
  Domain* dom = &(conf->dom_);

  printf("** Definition of mesh\n");
  printf("Number of points in  x with the coarse mesh : %d\n", dom->nxmax_[0]);
  printf("Number of points in  y with the coarse mesh : %d\n", dom->nxmax_[1]);
  printf("Number of points in Vx with the coarse mesh : %d\n", dom->nxmax_[2]);
  printf("Number of points in Vy with the coarse mesh : %d\n", dom->nxmax_[3]);
  
  printf("\n** Defintion of the geometry of the domain\n");
  printf("Minimal value of Ex : %lf\n", dom->minPhy_[0]);
  printf("Maximal value of Ex : %lf\n", dom->maxPhy_[0]);
  printf("Minimal value of Ey : %lf\n", dom->minPhy_[1]);
  printf("Maximal value of Ey : %lf\n", dom->maxPhy_[1]);
  
  printf("\nMinimal value of Vx : %lf\n", dom->minPhy_[2]);
  printf("Maximal value of Vx   : %lf\n", dom->maxPhy_[2]);
  printf("Minimal value of Vy   : %lf\n", dom->minPhy_[3]);
  printf("Maximal value of Vy   : %lf\n", dom->maxPhy_[3]);
  
  printf("\n** Considered test cases");
  printf("\n-10- Landau Damping");
  printf("\n-11- Landau Damping 2");
  printf("\n-20- Two beam instability");
  printf("\nNumber of the chosen test case : %d\n", dom->idcase_);
  
  printf("\n** Iterations in time and diagnostics\n");
  printf("Time step : %lf\n", dom->dt_);
  printf("Number of total iterations : %d\n", dom->nbiter_);
  printf("Frequency of diagnostics : %d\n", dom->ifreq_);
  
  printf("Diagnostics of fxvx : %d\n", dom->fxvx_);
}

void initcase(Config* conf, RealOffsetView4D fn) {
  Domain* dom = &(conf->dom_);
 
  switch(dom->idcase_) {
    case 2:
      testcaseCYL02(conf, fn);
      break; // CYL02 ;
    case 5:
      testcaseCYL05(conf, fn);
      break; // CYL05 ;
    case 10:
      testcaseSLD10(conf, fn);
      break; // SLD10 ;
    case 20:
      testcaseTSI20(conf, fn);
      break; // TSI20 ;
    default:
      printf("Unknown test case !\n");
      abort();
      break;
  }
}

void testcaseCYL02(Config* conf, RealOffsetView4D fn) {
  Domain* dom = &(conf->dom_);
  const float64 PI = M_PI;
  const float64 AMPLI = 4;
  const float64 PERIOD = 0.5 * PI;
  const float64 cc = 0.50 * (6. / 16.);
  const float64 rc = 0.50 * (4. / 16.);

  auto h_fn = Kokkos::create_mirror_view(fn);
  for(int ivy = dom->local_nxmin_[3]; ivy <= dom->local_nxmax_[3]; ivy++) {
    for(int ivx = dom->local_nxmin_[2]; ivx <= dom->local_nxmax_[2]; ivx++) {
      float64 vx = dom->minPhy_[2] + ivx * dom->dx_[2];
      for(int iy = dom->local_nxmin_[1]; iy <= dom->local_nxmax_[1]; iy++) {
        for(int ix = dom->local_nxmin_[0]; ix <= dom->local_nxmax_[0]; ix++) {
          float64 x = dom->minPhy_[0] + ix * dom->dx_[0];
          float64 xx = x;
          float64 vv = vx;
          
          float64 hv = 0.0;
          float64 hx = 0.0;
          
          if((vv <= cc + rc) && (vv >= cc - rc)) {
            hv = cos(PERIOD * ((vv - cc) / rc));
          } else if((vv <= -cc + rc) && (vv >= -cc - rc)) {
            hv = -cos(PERIOD * ((vv + cc) / rc));
          }
          
          if((xx <= cc + rc) && (xx >= cc - rc)) {
            hx = cos(PERIOD * ((xx - cc) / rc));
          } else if((xx <= -cc + rc) && (xx >= -cc - rc)) {
            hx = -cos(PERIOD * ((xx + cc) / rc));
          }
          
          h_fn(ix, iy, ivx, ivy) = (AMPLI * hx * hv);
        }
      }
    }
  }
  Kokkos::deep_copy(fn, h_fn);
}

void testcaseCYL05(Config* conf, RealOffsetView4D fn) {
  Domain * dom = &(conf->dom_);
  const float64 PI = M_PI;
  const float64 AMPLI = 4;
  const float64 PERIOD = 0.5 * PI;
  const float64 cc = 0.50 * (6. / 16.);
  const float64 rc = 0.50 * (4. / 16.);

  auto h_fn = Kokkos::create_mirror_view(fn);
  for(int ivy = dom->local_nxmin_[3]; ivy <= dom->local_nxmax_[3]; ivy++) {
    for(int ivx = dom->local_nxmin_[2]; ivx < dom->local_nxmax_[2]; ivx++) {
      float64 vy = dom->minPhy_[3] + ivy * dom->dx_[3];
      for(int iy = dom->local_nxmin_[1]; iy < dom->local_nxmax_[1]; iy++) {
        float64 y = dom->minPhy_[1] + iy * dom->dx_[1];
        for(int ix = dom->local_nxmin_[0]; ix < dom->local_nxmax_[0]; ix++) {
          float64 xx = y;
          float64 vv = vy;
          
          float64 hv = 0.0;
          float64 hx = 0.0;
          
          if((vv <= cc + rc) && (vv >= cc - rc)) {
            hv = cos(PERIOD * ((vv - cc) / rc));
          } else if((vv <= -cc + rc) && (vv >= -cc - rc)) {
            hv = -cos(PERIOD * ((vv + cc) / rc));
          }
          
          if((xx <= cc + rc) && (xx >= cc - rc)) {
            hx = cos(PERIOD * ((xx - cc) / rc));
          } else if((xx <= -cc + rc) && (xx >= -cc - rc)) {
            hx = -cos(PERIOD * ((xx + cc) / rc));
          }
          
          h_fn(ix, iy, ivx, ivy) = AMPLI * hx * hv;
        }
      }
    }
  }
  Kokkos::deep_copy(fn, h_fn);
}

void testcaseSLD10(Config* conf, RealOffsetView4D fn) {
  Domain * dom = &(conf->dom_);

  auto h_fn = Kokkos::create_mirror_view(fn);
  for(int ivy = dom->local_nxmin_[3]; ivy <= dom->local_nxmax_[3]; ivy++) {
    for(int ivx = dom->local_nxmin_[2]; ivx <= dom->local_nxmax_[2]; ivx++) {
      float64 vy = dom->minPhy_[3] + ivy * dom->dx_[3];
      float64 vx = dom->minPhy_[2] + ivx * dom->dx_[2];
      for(int iy = dom->local_nxmin_[1]; iy <= dom->local_nxmax_[1]; iy++) {
        float64 y = dom->minPhy_[1] + iy * dom->dx_[1];
        for(int ix = dom->local_nxmin_[0]; ix <= dom->local_nxmax_[0]; ix++) {
          float64 x = dom->minPhy_[0] + ix * dom->dx_[0];
            
          float64 sum = (vx * vx + vy * vy);
          h_fn(ix, iy, ivx, ivy) = (1. / (2 * M_PI)) * exp(-0.5 * (sum)) * (1 + 0.05 * (cos(0.5 * x) * cos(0.5 * y)));
        }
      }
    }
  }
  Kokkos::deep_copy(fn, h_fn);
}

void testcaseTSI20(Config* conf, RealOffsetView4D fn) {
  Domain * dom = &(conf->dom_);

  float64 vd  = 2.4;
  float64 vthx  = 1.0;
  float64 vthy  = 0.5;
  float64 alphax  = 0.05;
  float64 alphay  = 0.25;
  float64 kx  = 0.2;
  float64 ky  = 0.2;
    
  auto h_fn = Kokkos::create_mirror_view(fn);
  for(int ivy = dom->local_nxmin_[3]; ivy <= dom->local_nxmax_[3]; ivy++) {
    for(int ivx = dom->local_nxmin_[2]; ivx <= dom->local_nxmax_[2]; ivx++) {
      double vy = dom->minPhy_[3] + ivy * dom->dx_[3];
      double vx = dom->minPhy_[2] + ivx * dom->dx_[2];
      for(int iy = dom->local_nxmin_[1]; iy <= dom->local_nxmax_[1]; iy++) {
        for(int ix = dom->local_nxmin_[0]; ix <= dom->local_nxmax_[0]; ix++) {
          double yy = dom->minPhy_[1] + iy * dom->dx_[1];
          double xx = dom->minPhy_[0] + ix * dom->dx_[0];

          h_fn(ix, iy, ivx, ivy) = 
            1./(4*M_PI*vthx*vthy)*
            (1-alphax*sin(kx*xx)-alphay*sin(ky*yy))*
            (exp(-.5*(pow((vx-vd)/vthx,2)+pow(vy/vthy,2)))+
                exp(-.5*(pow((vx+vd)/vthx,2)+pow(vy/vthy,2))));
        }
      }
    }
  }
  Kokkos::deep_copy(fn, h_fn);
}

/* @brief pack some numbers into halo_fn for checking the communication routines
 */
void testcase_ptest_init(Config *conf, Distrib &comm, RealOffsetView4D halo_fn) {
  const Domain *dom = &(conf->dom_);

  auto h_halo_fn = Kokkos::create_mirror_view(halo_fn);

  for(int ivy = dom->local_nxmin_[3]; ivy <= dom->local_nxmax_[3]; ivy++) {
    for(int ivx = dom->local_nxmin_[2]; ivx <= dom->local_nxmax_[2]; ivx++) {
      for(int iy = dom->local_nxmin_[1]; iy <= dom->local_nxmax_[1]; iy++) {
        for(int ix = dom->local_nxmin_[0]; ix <= dom->local_nxmax_[0]; ix++) {
          h_halo_fn(ix, iy, ivx, ivy) = (double)((ivy + 111 * ivx) * 111 + iy) * 111 + ix;
        }
      }
    }
  }

  Kokkos::deep_copy(halo_fn, h_halo_fn);
}

void testcase_ptest_check(Config* conf, Distrib &comm, RealOffsetView4D halo_fn) {
  Domain *dom = &(conf->dom_);
  Urbnode *node = comm.node();
  int offp = HALO_PTS - 1, offm = HALO_PTS - 1;

  auto h_halo_fn = Kokkos::create_mirror_view(halo_fn);
  Kokkos::deep_copy(h_halo_fn, halo_fn);
  for(int ivy = node->xmin_[3] - offm; ivy <= node->xmax_[3] + offp; ivy++) {
    for(int ivx = node->xmin_[2] - offm; ivx <= node->xmax_[2] + offp; ivx++) {
      const int jvy = (dom->nxmax_[3] + ivy) % dom->nxmax_[3];
      const int jvx = (dom->nxmax_[2] + ivx) % dom->nxmax_[2];
   
      for(int iy = node->xmin_[1] - offm; iy <= node->xmax_[1] + offp; iy++) {
        const int jy = (dom->nxmax_[1] + iy) % dom->nxmax_[1];
                                                                                                                          
        for(int ix = node->xmin_[0] - offm; ix <= node->xmax_[0] + offp; ix++) {
          const int jx = (dom->nxmax_[0] + ix) % dom->nxmax_[0];
          double fval = h_halo_fn(ix, iy, ivx, ivy);
          double ref = (((double)jvy + 111. * (double)jvx) * 111. + (double)jy) * 111. + (double)jx;
          double diff = fval - ref;
          if(fabs(diff) > .1) {
            printf("[%d] Pb %d %d %d %d: %lf %lf %lf\n", comm.pid(), ivy, ivx, iy, ix, fval, ref, diff);
          } else {
            //printf("[%d] OK %d %d %d %d: %lf %lf\n", comm.pid(),ivy,ivx,iy,ix, fval,ref);
          }
        }
      }
    }
  }
}

void init(const char *file, Config *conf, Distrib &comm, RealOffsetView4D &fn, RealOffsetView4D &fnp1, RealOffsetView4D &fn_tmp, Efield **ef, Diags **dg, Spline **spline, std::vector<Timer*> &timers) {
  Domain* dom = &conf->dom_; 

  import(file, conf);
  if(comm.master()) print(conf);

  // Initialize communication manager
  comm.createDecomposition(conf);
  Urbnode *mynode = comm.node();

  shape_t<DIMENSION> nxmin_halo;
  shape_t<DIMENSION> nxmax_halo;
  for(int i=0; i<DIMENSION; i++)
    nxmin_halo[i] = mynode->xmin_[i] - HALO_PTS;
  for(int i=0; i<DIMENSION; i++)
    nxmax_halo[i] = mynode->xmax_[i] + HALO_PTS;

  // Allocate 4D data structures
  int nxmin = nxmin_halo[0], nymin = nxmin_halo[1], nvxmin = nxmin_halo[2], nvymin = nxmin_halo[3];
  int nxmax = nxmax_halo[0], nymax = nxmax_halo[1], nvxmax = nxmax_halo[2], nvymax = nxmax_halo[3];
  fn   = RealOffsetView4D("fn",   {nxmin, nxmax}, {nymin, nymax}, {nvxmin, nvxmax}, {nvymin, nvymax});
  fnp1 = RealOffsetView4D("fnp1", {nxmin, nxmax}, {nymin, nymax}, {nvxmin, nvxmax}, {nvymin, nvymax});
  fn_tmp = RealOffsetView4D("fn_tmp", {nxmin, nxmax}, {nymin, nymax}, {nvxmin, nvxmax}, {nvymin, nvymax});

  comm.neighboursList(conf, fn);
  comm.bookHalo(conf);

  testcase_ptest_init(conf, comm, fn);
  comm.exchangeHalo(conf, fn, timers);
  testcase_ptest_check(conf, comm, fn);

  timers[TimerEnum::pack]->reset();
  timers[TimerEnum::comm]->reset();
  timers[TimerEnum::unpack]->reset();

  // allocate and initialize field solver
  *ef = new Efield(conf, {dom->nxmax_[0], dom->nxmax_[1]});

  // allocate and initialize diagnostics data structures
  *dg = new Diags(conf);

  // allocated and initiallize spline helper
  *spline = new Spline(conf);

  // Initialize distribution function with zeros
  Impl::fill(fn, 0);
  initcase(conf, fn);
}

void initValues(Config *conf, RealOffsetView4D &fn, RealOffsetView4D &fnp1) {
  // Initialize distribution function with zeros
  Impl::fill(fn, 0);
  Impl::fill(fnp1, 0);
  initcase(conf, fn);
}

void finalize(Efield **ef, Diags **dg, Spline **spline) {
  // Store diagnostics
  delete *ef;
  delete *dg;
  delete *spline;
}

#ifndef __HELPER_HPP__
#define __HELPER_HPP__

#include <iomanip>
#include <sstream>
#include "Parallel_For.hpp"

template < class ViewType >
void L2norm(const ViewType &data, const int rank) {
  auto extents = data.extents();
  int nx = extents.extent(0), ny = extents.extent(1);
  const int n = nx * ny;
  using layout_type = typename ViewType::layout_type;
  using value_type = typename ViewType::value_type;
  float64 l2loc = 0.0;
  auto data_ = data.mdspan();
  if(std::is_same_v<layout_type, stdex::layout_left>) {
    l2loc = std::transform_reduce(std::execution::par_unseq,
                                  counting_iterator(0), counting_iterator(n),
                                  0.0,
                                  std::plus<value_type>(),
                                  [=] (const int idx) {
                                    const int ix = idx % nx;
                                    const int iy = idx / nx;
                                    return data_(ix, iy) * data_(ix, iy);
                                  });
  } else {
    l2loc = std::transform_reduce(std::execution::par_unseq,
                                  counting_iterator(0), counting_iterator(n),
                                  0.0,
                                  std::plus<value_type>(),
                                  [=] (const int idx) {
                                    const int ix = idx / nx;
                                    const int iy = idx % nx;
                                    return data_(ix, iy) * data_(ix, iy);
                                  });
  }
  std::stringstream ss;
  ss << "L2 norm of 2 dimensional view " << data.name() << " @ rank " << rank << ": " << std::scientific << std::setprecision(15) << l2loc;
  std::cout << ss.str() << std::endl;
}

template < class ViewType >
void print(const ViewType &data, const int rank) {
  auto extents = data.extents();
  int nx = extents.extent(0), ny = extents.extent(1);
  const int n = nx * ny;
  using layout_type = typename ViewType::layout_type;
  using value_type = typename ViewType::value_type;
  std::stringstream ss;
  ss << "elements of 2 dimensional view " << data.name() << " @ rank " << rank << ": \n";

  for(int iy = 0; iy < ny; iy++) {
    for(int ix = 0; ix < nx; ix++) {
      ss << std::scientific << std::setprecision(15) << data(ix, iy) << ", ";
    }
    ss << "\n";
  }
  std::cout << ss.str() << std::endl;
}

#endif

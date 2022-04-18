#ifndef __ITERATION_HPP__
#define __ITERATION_HPP__

#include <cassert>
#include <algorithm>
#include <numeric>
#include <array>

template <size_t ND>
using shape_nd = std::array<int, ND>;

template <class LayoutPolicy, size_t ND, size_t SIMD_WIDTH=1>
class IteratePolicy {
private:
  size_t size_;
  size_t SIMD_WIDTH_ = SIMD_WIDTH;
  shape_nd<ND> strides_;
  static constexpr size_t rank_ = ND;
  shape_nd<ND> start_;

public:
  using layout_type = LayoutPolicy;

public:
  // Disable default constructor
  IteratePolicy() = delete;

  // Constructor instanized with {stop}
  IteratePolicy(const shape_nd<ND>& stop) : start_{0} {
    init(start_, stop);
  }

  // Constructor instanized with {start, stop}
  IteratePolicy(const shape_nd<ND>& start, const shape_nd<ND>& stop) : start_(start) {
    init(start, stop);
  }

  // Constructor only for 1D
  template <typename... I>
  IteratePolicy(const I... indices) {
    static_assert(ND == 1, "This should only be used for 1D case");
    static_assert(sizeof...(I) <= 2, "range should be given in {start, stop} or {stop}");

    shape_nd<1> start, stop;
    using index_type = std::tuple_element_t<0, std::tuple<I...>>;
    index_type indices_tmp[sizeof...(I)] = {indices...};
    if(sizeof...(I) == 1) {
      start[0] = 0;
      stop[0] = indices_tmp[0];
    } else {
      start[0] = indices_tmp[0];
      stop[0] = indices_tmp[1];
    }

    init(start, stop);
  }

  static constexpr size_t rank() { return rank_; }
  constexpr size_t size() const { return size_; }
  const shape_nd<ND> start() const { return start_; }
  const shape_nd<ND> strides() const { return strides_; }

private:
  void init(const shape_nd<ND>& start, const shape_nd<ND>& stop) {
    start_ = start;
    std::transform(stop.begin(), stop.end(), start.begin(), strides_.begin(), std::minus<int>());
    size_t size = 1;
    for(auto stride : strides_) {
      assert(stride > 0);
      size *= stride;
    }
    size_ = size;
  }
};

#endif

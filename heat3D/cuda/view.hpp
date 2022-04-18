#ifndef __VIEW_HPP__
#define __VIEW_HPP__

#include <experimental/mdspan>
#include <type_traits>
#include <vector>
#include <string>

#if defined (ENABLE_CUDA) || defined (ENABLE_HIP)
  #include <thrust/host_vector.h>
  #include <thrust/device_vector.h>
  template <typename ElementType>
    using host_vector = typename thrust::host_vector<ElementType>;
  template <typename ElementType>
    using device_vector = typename thrust::device_vector<ElementType>;
#else
  template <typename ElementType>
    using host_vector = typename std::vector<ElementType>;
  template <typename ElementType>
    using device_vector = typename std::vector<ElementType>;
#endif

namespace stdex = std::experimental;

template <
  class ElementType,
  class Extents,
  class LayoutPolicy = std::experimental::layout_right
>
class View {
public:
  using mdspan_type = stdex::mdspan<ElementType, Extents, LayoutPolicy>;
  using host_vector_type = host_vector<ElementType>;
  using device_vector_type = device_vector<ElementType>;
  using value_type = typename mdspan_type::value_type;
  using extents_type = typename mdspan_type::extents_type;
  using size_type = typename mdspan_type::size_type;

private:
  std::string name_;
  bool is_empty_;
  size_type size_;

  host_vector_type host_vector_;
  device_vector_type device_vector_;
  mdspan_type host_mdspan_, device_mdspan_;

public:
  View() : name_("empty"), is_empty_(true) {}
  View(std::string name, std::array<size_type, extents_type::rank()> extents) : name_(name), is_empty_(false) {
    init(extents);
  }
  
  template <typename... I>
  View(std::string name, I... indices) : name_(name), is_empty_(false) {
    std::array<size_type, extents_type::rank()> extents = {static_cast<size_type>(indices)...};
    init(extents);
  }

  void init(std::array<size_type, extents_type::rank()> extents) {
    size_type size = 1;
    for(auto&& extent: extents)
      size *= extent;

    // Allocate data by std::vector and then map it to mdspan
    size_ = size;
    host_vector_.resize(size);
    host_mdspan_ = mdspan_type((ElementType *)thrust::raw_pointer_cast(host_vector_.data()), extents);
    device_vector_.resize(size);
    device_mdspan_ = mdspan_type((ElementType *)thrust::raw_pointer_cast(device_vector_.data()), extents);
  }

  void swap(View &rhs) {
    std::string name = this->name();
    bool is_empty = this->is_empty();

    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());

    rhs.setName(name);
    rhs.setIsEmpty(is_empty);
    std::swap(this->host_mdspan_, rhs.host_mdspan_);
    std::swap(this->device_mdspan_, rhs.device_mdspan_);
  }

public:
  const std::string name() const noexcept {return name_;}
  bool is_empty() const noexcept { return is_empty_; }
  constexpr int rank() noexcept { return extents_type::rank(); }
  constexpr int rank_dynamic() noexcept { return extents_type::rank_dynamic(); }
  constexpr size_type size() const noexcept { return size_; }
  value_type *host_data() { return host_mdspan_.data(); }
  const  value_type *host_data() const { return host_mdspan_.data(); }
  value_type *device_data() { return device_mdspan_.data(); }
  const value_type *device_data() const { return device_mdspan_.data(); }
  mdspan_type host_mdspan() const { return host_mdspan_; }
  mdspan_type &host_mdspan() { return host_mdspan_; }
  mdspan_type device_mdspan() const { return device_mdspan_; }
  mdspan_type &device_mdspan() { return device_mdspan_; }

  inline void setName(const std::string &name) { name_ = name; }
  inline void setIsEmpty(bool is_empty) { is_empty_ = is_empty; }

  void updateDevice() { device_vector_ = host_vector_; }
  void updateSelf() { host_vector_ = device_vector_; }

  template <typename... I>
  inline ElementType& operator()(I... indices) noexcept {
    static_assert(sizeof...(I) == extents_type::rank(), "The number of indices must be equal to rank");
    return host_mdspan_(indices...);
  }
  template <typename... I>
  inline ElementType operator()(I... indices) const noexcept {
    static_assert(sizeof...(I) == extents_type::rank(), "The number of indices must be equal to rank");
    return host_mdspan_(indices...);
  }

  template <typename... I>
  inline ElementType& operator[](I... indices) noexcept {
    static_assert(sizeof...(I) == extents_type::rank(), "The number of indices must be equal to rank");
    return host_mdspan_(indices...);
  }

  template <typename... I>
  inline ElementType& operator[](I... indices) const noexcept {
    static_assert(sizeof...(I) == extents_type::rank(), "The number of indices must be equal to rank");
    return host_mdspan_(indices...);
  }
};

#endif

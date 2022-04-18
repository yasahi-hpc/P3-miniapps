#ifndef __VIEW_HPP__
#define __VIEW_HPP__

#include <experimental/mdspan>
#include <type_traits>
#include <vector>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/* [TO DO] Check the behaviour of thrust::device_vector if it is configured for CPUs */
template <typename ElementType>
  using host_vector = typename thrust::host_vector<ElementType>;
#if defined (ENABLE_CUDA) || defined (ENABLE_HIP)
  template <typename ElementType>
    using device_vector = typename thrust::device_vector<ElementType>;
#else
  template <typename ElementType>
    using device_vector = typename thrust::host_vector<ElementType>;
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
  using int_type = int64_t;
  using layout_type = typename mdspan_type::layout_type;

private:
  std::string name_;
  bool is_empty_;
  size_type size_;
  int_type total_offset_;
  std::array<int_type, extents_type::rank()> offsets_;

  value_type *host_data_;
  value_type *device_data_;
  host_vector_type host_vector_;
  device_vector_type device_vector_;
  mdspan_type host_mdspan_;
  mdspan_type device_mdspan_;

public:
  View() : name_("empty"), is_empty_(true), total_offset_(0), offsets_{0} {}
  View(std::string name, std::array<size_type, extents_type::rank()> extents)
    : name_(name), is_empty_(false), total_offset_(0), offsets_{0} {
    init(extents, offsets_);
  }
  
  // Kokkos like constructor
  template <typename... I>
  View(std::string name, I... indices)
    : name_(name), is_empty_(false), total_offset_(0), offsets_{0} {
    std::array<size_type, extents_type::rank()> extents = {static_cast<size_type>(indices)...};
    init(extents, offsets_);
  }

  // Offset View
  View(std::string name,
       std::array<size_type, extents_type::rank()> extents,
       std::array<int_type, extents_type::rank()> offsets
      ) : name_(name), is_empty_(false), offsets_(offsets) {
 
    init(extents, offsets);
  }

  ~View() {}

  // Copy constructor and assignment operators
  View(const View &rhs) {
    shallow_copy(rhs);
  }
 
  View& operator=(const View &rhs) {
    if (this == &rhs) return *this;
    deep_copy(rhs);
    return *this;
  }

  // Move and Move assignment
  View(View &&rhs) noexcept {
    deep_copy(std::forward<View>(rhs));
  }

  View& operator=(View &&rhs) {
    if (this == &rhs) return *this;
    deep_copy(std::forward<View>(rhs));
    return *this;
  }

private:
  void init(std::array<size_type, extents_type::rank()> extents,
            std::array<int_type, extents_type::rank()> offsets) {
    auto size = std::accumulate(extents.begin(), extents.end(), 1, std::multiplies<>());

    // Compute offsets
    int_type total_offset = 0;
    size_type total_extents = 1;
    if(std::is_same_v<layout_type, layout_contiguous_at_left>) {
      for(size_type i=0; i<extents_type::rank(); i++) {
        total_offset -= offsets[i] * total_extents;
        total_extents *= extents[i];
      }
    } else {
      for(size_type i=extents_type::rank()-1; i >= 0; i--) {
        total_offset -= offsets[i] * total_extents;
        total_extents *= extents[i];
      }
    }

    // Allocate data by std::vector and then map it to mdspan
    size_ = size;
    total_offset_ = total_offset;

    host_vector_.resize(size);
    host_data_ = (value_type *)thrust::raw_pointer_cast(host_vector_.data());
    host_mdspan_ = mdspan_type(host_data_ + total_offset_, extents);
    device_vector_.resize(size);
    device_data_ = (value_type *)thrust::raw_pointer_cast(device_vector_.data());
    device_mdspan_ = mdspan_type(device_data_ + total_offset_, extents);
  }

  // Only copying meta data
  void shallow_copy(const View &rhs) {
    this->setName(rhs.name()+"_copy");
    this->setIsEmpty(rhs.is_empty());
    size_ = rhs.size();
    total_offset_ = rhs.total_offset();
    offsets_ = rhs.offsets();

    host_data_ = rhs.host_data();
    device_data_ = rhs.device_data();
    host_mdspan_ = rhs.host_mdspan();
    device_mdspan_ = rhs.device_mdspan();
  }

  void shallow_copy(View &&rhs) {
    this->setName(rhs.name()+"_copy");
    this->setIsEmpty(rhs.is_empty());
    size_ = rhs.size();
    total_offset_ = rhs.total_offset();
    offsets_ = rhs.offsets();

    host_data_ = rhs.host_data();
    device_data_ = rhs.device_data();
    host_mdspan_ = rhs.host_mdspan();
    device_mdspan_ = rhs.device_mdspan();
  }

  void deep_copy(const View &rhs) {
    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());
    size_ = rhs.size();
    total_offset_ = rhs.total_offset();
    offsets_ = rhs.offsets();

    host_vector_ = rhs.host_vector_; // not a move
    device_vector_ = rhs.device_vector_; // not a move
    host_data_ = (value_type *)thrust::raw_pointer_cast(host_vector_.data());
    device_data_ = (value_type *)thrust::raw_pointer_cast(device_vector_.data());
    host_mdspan_ = mdspan_type(host_data_ + total_offset_, rhs.extents());
    device_mdspan_ = mdspan_type(device_data_ + total_offset_, rhs.extents());
  }

  void deep_copy(View &&rhs) {
    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());
    size_ = rhs.size();
    total_offset_ = rhs.total_offset();
    offsets_ = rhs.offsets();

    host_vector_ = std::move(rhs.host_vector_);
    device_vector_ = std::move(rhs.device_vector_);
    host_data_ = (value_type *)thrust::raw_pointer_cast(host_vector_.data());
    device_data_ = (value_type *)thrust::raw_pointer_cast(device_vector_.data());
    host_mdspan_ = mdspan_type(host_data_ + total_offset_, rhs.extents());
    device_mdspan_ = mdspan_type(device_data_ + total_offset_, rhs.extents());
  }

  void swap(View &rhs) {
    std::string name = this->name();
    bool is_empty = this->is_empty();
    size_type size = this->size();
    int_type total_offset = this->total_offset();
    auto offsets = this->offsets();
    //value_type *host_data = this->host_data();
    //value_type *device_data = this->device_data();

    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());
    this->setSize(rhs.size());
    this->setTotalOffset(rhs.total_offset());
    this->setOffsets(rhs.offsets());
    //this->setHostData(rhs.host_data());
    //this->setDeviceData(rhs.device_data());

    rhs.setName(name);
    rhs.setIsEmpty(is_empty);
    rhs.setSize(size);
    rhs.setTotalOffset(total_offset);
    rhs.setOffsets(offsets);


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

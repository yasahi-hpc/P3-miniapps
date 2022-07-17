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
  using int_type = int;
  using layout_type = typename mdspan_type::layout_type;

private:
  std::string name_;
  bool is_empty_;
  int_type total_offset_;
  std::array<int_type, extents_type::rank()> offsets_;
  std::array<int_type, extents_type::rank()> ends_;

  host_vector_type host_vector_;
  device_vector_type device_vector_;
  mdspan_type host_mdspan_;
  mdspan_type device_mdspan_;

public:
  View() : name_("empty"), is_empty_(true), total_offset_(0), offsets_{0} {}
  View(const std::string name, std::array<size_type, extents_type::rank()> extents)
    : name_(name), is_empty_(false), total_offset_(0), offsets_{0} {
    init(extents, offsets_);
  }
  
  // Kokkos like constructor
  template <typename... I,
             std::enable_if_t<
               std::is_integral_v<
                 std::tuple_element_t<0, std::tuple<I...>>
               >, std::nullptr_t> = nullptr>
  View(const std::string name, I... indices)
    : name_(name), is_empty_(false), total_offset_(0), offsets_{0} {
    std::array<size_type, extents_type::rank()> extents = {static_cast<size_type>(indices)...};
    init(extents, offsets_);
  }

  // Offset View
  template <typename SizeType>
  View(const std::string name,
       const std::array<SizeType, extents_type::rank()>& extents,
       const std::array<int_type, extents_type::rank()>& offsets
      ) : name_(name), is_empty_(false), offsets_(offsets) {
    // Cast to size_t explicitly
    std::array<size_type, extents_type::rank()> _extents;
    std::transform(extents.begin(), extents.end(), _extents.begin(),
                   [](const SizeType e) -> size_type { return static_cast<size_type>(e);} );
    init(_extents, offsets);
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
    std::transform(offsets.begin(), offsets.end(), extents.begin(), ends_.begin(), std::plus<int_type>());

    // Compute offsets
    int_type total_offset = 0;
    size_type total_extents = 1;
    if(std::is_same_v<layout_type, stdex::layout_left>) {
      for(size_type i=0; i<extents_type::rank(); i++) {
        total_offset -= offsets[i] * total_extents;
        total_extents *= extents[i];
      }
    } else {
      for(size_type i=0; i<extents_type::rank(); i++) {
        total_offset -= offsets[extents_type::rank()-1-i] * total_extents;
        total_extents *= extents[extents_type::rank()-1-i];
      }
    }

    // Allocate data by std::vector and then map it to mdspan
    total_offset_ = total_offset;

    host_vector_.resize(size);
    host_mdspan_ = mdspan_type((value_type *)thrust::raw_pointer_cast(host_vector_.data()) + total_offset_, extents);

    #if defined (ENABLE_CUDA) || defined (ENABLE_HIP)
      device_vector_.resize(size);
      device_mdspan_ = mdspan_type((value_type *)thrust::raw_pointer_cast(device_vector_.data()) + total_offset_, extents);
    #else
      // In the host configuration, device_mdspan_ also points the host_vector
      device_mdspan_ = mdspan_type((value_type *)thrust::raw_pointer_cast(host_vector_.data()) + total_offset_, extents);
    #endif
  }

  // Only copying meta data
  void shallow_copy(const View &rhs) {
    this->setName(rhs.name()+"_copy");
    this->setIsEmpty(rhs.is_empty());
    total_offset_ = rhs.total_offset();
    offsets_ = rhs.offsets();
    ends_ = rhs.end();

    host_mdspan_ = rhs.host_mdspan();
    device_mdspan_ = rhs.device_mdspan();
  }

  void shallow_copy(View &&rhs) {
    this->setName(rhs.name()+"_copy");
    this->setIsEmpty(rhs.is_empty());
    total_offset_ = rhs.total_offset();
    offsets_ = rhs.offsets();
    ends_ = rhs.end();

    host_mdspan_ = rhs.host_mdspan();
    device_mdspan_ = rhs.device_mdspan();
  }

  void deep_copy(const View &rhs) {
    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());
    total_offset_ = rhs.total_offset();
    offsets_ = rhs.offsets();
    ends_ = rhs.end();

    host_vector_ = rhs.host_vector_; // not a move
    host_mdspan_ = mdspan_type((value_type *)thrust::raw_pointer_cast(host_vector_.data()) + total_offset_, rhs.extents());

    #if defined (ENABLE_CUDA) || defined (ENABLE_HIP)
      device_vector_ = rhs.device_vector_; // not a move
      device_mdspan_ = mdspan_type((value_type *)thrust::raw_pointer_cast(device_vector_.data()) + total_offset_, rhs.extents());
    #else
      device_mdspan_ = mdspan_type((value_type *)thrust::raw_pointer_cast(host_vector_.data()) + total_offset_, rhs.extents());
    #endif
  }

  void deep_copy(View &&rhs) {
    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());
    total_offset_ = rhs.total_offset();
    offsets_ = rhs.offsets();
    ends_ = rhs.end();

    host_vector_ = std::move(rhs.host_vector_);
    host_mdspan_ = mdspan_type((value_type *)thrust::raw_pointer_cast(host_vector_.data()) + total_offset_, rhs.extents());
    #if defined (ENABLE_CUDA) || defined (ENABLE_HIP)
      device_vector_ = std::move(rhs.device_vector_);
      device_mdspan_ = mdspan_type((value_type *)thrust::raw_pointer_cast(device_vector_.data()) + total_offset_, rhs.extents());
    #else
      device_mdspan_ = mdspan_type((value_type *)thrust::raw_pointer_cast(host_vector_.data()) + total_offset_, rhs.extents());
    #endif
  }

public:
  void swap(View &rhs) {
    std::string name = this->name();
    bool is_empty = this->is_empty();
    int_type total_offset = this->total_offset();
    auto offsets = this->offsets();
    auto ends = this->end();

    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());
    this->setTotalOffset(rhs.total_offset());
    this->setOffsets(rhs.offsets());
    this->setEnds(rhs.end());

    rhs.setName(name);
    rhs.setIsEmpty(is_empty);
    rhs.setTotalOffset(total_offset);
    rhs.setOffsets(offsets);
    rhs.setEnds(ends);

    thrust::swap(this->host_vector_, rhs.host_vector_);
    #if defined (ENABLE_CUDA) || defined (ENABLE_HIP)
      thrust::swap(this->device_vector_, rhs.device_vector_);
    #endif
    std::swap(this->host_mdspan_, rhs.host_mdspan_);
    std::swap(this->device_mdspan_, rhs.device_mdspan_);
  }

public:
  const std::string name() const noexcept {return name_;}
  bool is_empty() const noexcept { return is_empty_; }
  constexpr int rank() noexcept { return extents_type::rank(); }
  constexpr int rank_dynamic() noexcept { return extents_type::rank_dynamic(); }
  constexpr size_type size() const noexcept { return host_mdspan_.size(); }
  constexpr extents_type extents() const noexcept { return host_mdspan_.extents(); }
  constexpr size_type extent(size_type r) const noexcept { return host_mdspan_.extents().extent(r); }
  int_type total_offset() const noexcept { return total_offset_; }
  std::array<int_type, extents_type::rank()> offsets() const noexcept { return offsets_; }
  int_type offset(size_type r) const noexcept { return offsets_[r]; }

  value_type *data() { return device_mdspan_.data_handle() - total_offset_; }
  const value_type *data() const { return device_mdspan_.data_handle() - total_offset_; }
  value_type *host_data() { return host_mdspan_.data_handle() - total_offset_; }
  const value_type *host_data() const { return host_mdspan_.data_handle() - total_offset_; }
  value_type *device_data() { return device_mdspan_.data_handle() - total_offset_; }
  const value_type *device_data() const { return device_mdspan_.data_handle() - total_offset_; }

  mdspan_type mdspan() const { return device_mdspan_; }
  mdspan_type &mdspan() { return device_mdspan_; }
  mdspan_type host_mdspan() const { return host_mdspan_; }
  mdspan_type &host_mdspan() { return host_mdspan_; }
  mdspan_type device_mdspan() const { return device_mdspan_; }
  mdspan_type &device_mdspan() { return device_mdspan_; }

  std::array<int_type, extents_type::rank()> begin() const noexcept { return offsets_; }
  std::array<int_type, extents_type::rank()> end() const noexcept { return ends_; }
  int_type begin(size_type r) const noexcept { return offsets_[r]; }
  int_type end(size_type r) const noexcept { return ends_[r]; }

  inline void setName(const std::string &name) { name_ = name; }
  inline void setIsEmpty(bool is_empty) { is_empty_ = is_empty; }
  inline void setTotalOffset(int_type total_offset) { total_offset_ = total_offset; }
  inline void setOffsets(std::array<int_type, extents_type::rank()> offsets) { offsets_ = offsets; }
  inline void setEnds(std::array<int_type, extents_type::rank()> ends) { ends_ = ends; }

  void updateDevice() {
    #if defined (ENABLE_CUDA) || defined (ENABLE_HIP)
      device_vector_ = host_vector_; 
    #endif
  }

  void updateSelf() {
    #if defined (ENABLE_CUDA) || defined (ENABLE_HIP)
      host_vector_ = device_vector_; 
    #endif
  }

  void fill(const ElementType value = 0) {
    #if defined (ENABLE_CUDA) || defined (ENABLE_HIP)
      thrust::fill(device_vector_.begin(), device_vector_.end(), value);
      updateSelf();
    #else 
      thrust::fill(host_vector_.begin(), host_vector_.end(), value);
    #endif
  }

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

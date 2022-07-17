#ifndef __VIEW_HPP__
#define __VIEW_HPP__

#include <experimental/mdspan>
#include <type_traits>
#include <vector>
#include <string>
#include <numeric>
#include <execution>
#include <algorithm>

namespace stdex = std::experimental;

template <typename... Args>
struct Traits {
  using tuple = std::tuple<Args...>;
  static constexpr size_t size = sizeof...(Args);
  template <std::size_t N>
    using Nth = typename std::tuple_element<N, tuple>::type;
  using first = Nth<0>;
  using Last  = Nth<size - 1>;
};


template <
  class ElementType,
  class Extents,
  class LayoutPolicy = std::experimental::layout_right
>
class View {
public:
  using mdspan_type = stdex::mdspan<ElementType, Extents, LayoutPolicy>;
  using vector_type = std::vector<ElementType>;
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

  vector_type vector_;
  mdspan_type mdspan_;

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

    vector_.resize(size);
    mdspan_ = mdspan_type(vector_.data() + total_offset_, extents);
  }

  // Only copying meta data
  void shallow_copy(const View &rhs) {
    this->setName(rhs.name()+"_copy");
    this->setIsEmpty(rhs.is_empty());
    total_offset_ = rhs.total_offset();
    offsets_ = rhs.offsets();
    ends_ = rhs.end();

    mdspan_ = rhs.mdspan();
  }

  void shallow_copy(View &&rhs) {
    this->setName(rhs.name()+"_copy");
    this->setIsEmpty(rhs.is_empty());
    total_offset_ = rhs.total_offset();
    offsets_ = rhs.offsets();
    ends_ = rhs.end();

    mdspan_ = rhs.mdspan();
  }

  void deep_copy(const View &rhs) {
    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());
    total_offset_ = rhs.total_offset();
    offsets_ = rhs.offsets();
    ends_ = rhs.end();

    vector_ = rhs.vector_; // not a move
    mdspan_ = mdspan_type(vector_.data() + total_offset_, rhs.extents());
  }

  void deep_copy(View &&rhs) {
    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());
    total_offset_ = rhs.total_offset();
    offsets_ = rhs.offsets();
    ends_ = rhs.end();

    vector_ = std::move(rhs.vector_);
    mdspan_ = mdspan_type(vector_.data() + total_offset_, rhs.extents());
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

    std::swap(this->vector_, rhs.vector_);
    std::swap(this->mdspan_, rhs.mdspan_);
  }

public:
  const std::string name() const noexcept {return name_;}
  bool is_empty() const noexcept { return is_empty_; }
  constexpr size_t rank() noexcept { return extents_type::rank(); }
  constexpr size_t rank_dynamic() noexcept { return extents_type::rank_dynamic(); }
  constexpr size_type size() const noexcept { return mdspan_.size(); }
  constexpr extents_type extents() const noexcept { return mdspan_.extents(); }
  constexpr size_type extent(size_t r) const noexcept { return mdspan_.extents().extent(r); }
  int_type total_offset() const noexcept { return total_offset_; }
  std::array<int_type, extents_type::rank()> offsets() const noexcept { return offsets_; }

  value_type *data() { return mdspan_.data_handle() - total_offset_; }
  const value_type *data() const { return mdspan_.data_handle() - total_offset_; }
  mdspan_type mdspan() const { return mdspan_; }
  mdspan_type &mdspan() { return mdspan_; }

  std::array<int_type, extents_type::rank()> begin() const noexcept { return offsets_; }
  std::array<int_type, extents_type::rank()> end() const noexcept { return ends_; }
  int_type begin(size_t i) const noexcept { return offsets_[i]; }
  int_type end(size_t i) const noexcept { return ends_[i]; }

  inline void setName(const std::string &name) { name_ = name; }
  inline void setIsEmpty(bool is_empty) { is_empty_ = is_empty; }
  inline void setTotalOffset(int_type total_offset) { total_offset_ = total_offset; }
  inline void setOffsets(std::array<int_type, extents_type::rank()> offsets) { offsets_ = offsets; }
  inline void setEnds(std::array<int_type, extents_type::rank()> ends) { ends_ = ends; }

  // Do nothing, in order to inform compiler "vector_" is on device by launching a gpu kernel
  void updateDevice() {
    auto tmp = vector_;
    std::copy(std::execution::par_unseq, tmp.begin(), tmp.end(), vector_.begin());
  }

  // Do nothing, in order to inform compiler "vector_" is on host by launching a host kernel
  void updateSelf() {
    auto tmp = vector_;
    std::copy(tmp.begin(), tmp.end(), vector_.begin());
  } 

  void fill(const ElementType value = 0) {
    std::fill(vector_.begin(), vector_.end(), value);
  }

  template <typename... I>
  inline ElementType& operator()(I... indices) noexcept {
    static_assert(sizeof...(I) == extents_type::rank(), "The number of indices must be equal to rank");
    return mdspan_(indices...);
  }
  template <typename... I>
  inline ElementType operator()(I... indices) const noexcept {
    static_assert(sizeof...(I) == extents_type::rank(), "The number of indices must be equal to rank");
    return mdspan_(indices...);
  }

  template <typename... I>
  inline ElementType& operator[](I... indices) noexcept {
    static_assert(sizeof...(I) == extents_type::rank(), "The number of indices must be equal to rank");
    return mdspan_(indices...);
  }

  template <typename... I>
  inline ElementType& operator[](I... indices) const noexcept {
    static_assert(sizeof...(I) == extents_type::rank(), "The number of indices must be equal to rank");
    return mdspan_(indices...);
  }
};

#endif

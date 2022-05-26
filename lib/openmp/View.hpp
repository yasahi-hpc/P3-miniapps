/*
 *  Simplified View to hand multidimensional array inside OpenMP offload region
 */

#ifndef __VIEW_HPP__
#define __VIEW_HPP__

#include <string>
#include <type_traits>
#include <array>
#include <omp.h>
#include <experimental/mdspan>

namespace stdex = std::experimental;

/*
 * Reference
 * Daley, Christopher & Ahmed, Hadia & Williams, Samuel & Wright, N.J.. (2020).
 * A Case Study of Porting HPGMG from CUDA to OpenMP Target Offload. 10.1007/978-3-030-58144-2_3.
 * url: https://crd.lbl.gov/assets/Uploads/p24-daley.pdf
 */
inline void omp_attach(void **ptr) {
  #if defined( ENABLE_OPENMP_OFFLOAD )
    void *dptr = *ptr;
    if(dptr) {
      #pragma omp target data use_device_ptr(dptr)
      {
        #pragma omp target is_device_ptr(dptr)
        {
          *ptr = dptr;
        }
      }
    }
  #endif
}

inline void omp_detach(void **ptr) {
  /*
  #if defined( ENABLE_OPENMP_OFFLOAD )
    void *dptr = *ptr;
    if(dptr) {
      #pragma omp target is_device_ptr(dptr)
      {
        *ptr = nullptr;
      }
    }
  #endif
  */
}

template <typename ScalarType, size_t ND, class LayoutPolicy=stdex::layout_left>
class View {
public:
  using value_type = ScalarType;
  using size_type = size_t;
  using int_type = int;
  using layout_type = LayoutPolicy;

private:
  std::string name_;

  // Do not delete the pointers, if this instance is a copy
  bool is_copied_;

  // In case instanized with default constructor
  bool is_empty_;

  // Raw data
  value_type *data_;

  // Meta data used in offload region
  int_type *strides_;
  int_type total_offset_;

  // Used outside offload region
  std::array<size_type, ND> extents_;
  std::array<int_type, ND> offsets_;
  std::array<int_type, ND> ends_;
  size_type size_; // total data size
  static constexpr size_type rank_ = ND;

public:
  // Default constructor, define an empty view
  View() : name_("empty"), total_offset_(0), size_(0), data_(nullptr), strides_(nullptr),
           extents_ {0}, offsets_ {0}, ends_ {0}, is_copied_(false), is_empty_(true)
  {}

  // Constructor instanized with shape_nd
  View(const std::string name, const std::array<size_type, ND>& extents)
    : name_(name), total_offset_(0), extents_(extents), offsets_ {0}, ends_ {0}, is_copied_(false), is_empty_(false) {
    init(extents_, offsets_);
  }

  // Kokkos like constructor
  template <typename... I,
              std::enable_if_t<
                std::is_integral_v<
                  std::tuple_element_t<0, std::tuple<I...>>
                >, std::nullptr_t> = nullptr>
  View(const std::string name, I... indices)
    : name_(name), total_offset_(0), offsets_ {0}, ends_ {0}, is_copied_(false), is_empty_(false) {
    static_assert(sizeof...(I) == ND, "The number of indices must be equal to ND");
    // expand parameter packs in the initializer
    std::array<size_type, ND> extents = {static_cast<size_type>(indices)...};

    init(extents, offsets_);
  }

  // Offset view
  template <typename I>
  View(const std::string name,
       const std::array<I, ND>& extents,
       const std::array<int_type, ND>& offsets)
    : name_(name), total_offset_(0), offsets_(offsets), is_copied_(false), is_empty_(false) {
    // Cast to size_t explicitly
    std::array<size_type, ND> _extents;
    std::transform(extents.begin(), extents.end(), _extents.begin(),
                   [](const I e) -> size_type { return static_cast<size_type>(e);} );
    init(_extents, offsets_);
  }

  ~View() { free(); }

  // Copy construct performs shallow copy
  View(const View &rhs)
    : name_(rhs.name_), is_copied_(true), is_empty_(rhs.is_empty_), size_(rhs.size_), data_(rhs.data_), strides_(rhs.strides_),
    total_offset_(rhs.total_offset_), extents_(rhs.extents_), offsets_(rhs.offsets_), ends_(rhs.ends_) {

    // Attach pointers
    #if defined(ENABLE_OPENMP_OFFLOAD)
      #pragma omp target enter data map(alloc: this[0:1])
      #pragma omp target enter data map(alloc: total_offset_)
      #pragma omp target update to(total_offset_)
      omp_attach( (void**) &data_ );
      omp_attach( (void**) &strides_ );
    #endif
  }

  // Assignment operator used only for data allocation, values are also copied
  View& operator=(const View &rhs) {
    if (this == &rhs) return *this;
    free();

    this->is_empty_  = rhs.is_empty();
    this->is_copied_ = false;
    this->name_      = rhs.name();
    this->extents_   = rhs.extents();
    this->offsets_   = rhs.offsets();

    init(extents_, offsets_);

    // Copy data
    std::copy(rhs.data(), rhs.data() + size_, data_);
    updateDevice();

    return *this;
  }

  // Move constructor
  View(View &&rhs) noexcept
    : name_(rhs.name_), is_copied_(rhs.is_copied_), is_empty_(rhs.is_empty_), size_(rhs.size_), data_(rhs.data_), strides_(rhs.strides_),
    total_offset_(rhs.total_offset_), extents_(rhs.extents_), offsets_(rhs.offsets_), ends_(rhs.ends_) {

    #if defined(ENABLE_OPENMP_OFFLOAD)
      #pragma omp target enter data map(alloc: this[0:1])
      #pragma omp target enter data map(alloc: total_offset_)
      #pragma omp target update to(total_offset_)
      omp_attach( (void**) &data_ );
      omp_attach( (void**) &strides_ );
    #endif

    // Set rhs to be copied state in order not to deallocate the rhs data (detach only)
    rhs.setIsCopied(true);
    rhs.free();
  }

  // Move assignment operator
  View& operator=(View &&rhs) noexcept {
    if (this == &rhs) return *this;
    free(); // freeing empty

    this->is_empty_     = rhs.is_empty();
    this->is_copied_    = rhs.is_copied();
    this->name_         = rhs.name();
    this->size_         = rhs.size();
    this->total_offset_ = rhs.total_offset();
    this->extents_      = rhs.extents();
    this->offsets_      = rhs.offsets();
    this->ends_         = rhs.ends();
    this->data_         = rhs.data();
    this->strides_      = rhs.strides();

    #if defined(ENABLE_OPENMP_OFFLOAD)
      #pragma omp target enter data map(alloc: this[0:1])
      #pragma omp target enter data map(alloc: total_offset_)
      #pragma omp target update to(total_offset_)
      omp_attach( (void**) &data_ );
      omp_attach( (void**) &strides_ );
    #endif

    // Set rhs to be copied state in order not to deallocate the rhs data (detach only)
    rhs.setIsCopied(true);
    rhs.free();

    return *this;
  }

public:
  template <typename... I>
  inline value_type& operator()(I... indices) const noexcept {
    static_assert(sizeof...(I) == ND, "The number of indices must be equal to ND");
    return access(indices...);
  }

  template <typename... I>
  inline value_type& operator[](I... indices) const noexcept {
    static_assert(sizeof...(I) == ND, "The number of indices must be equal to ND");
    return access(indices...);
  }

  // methods for device/host data transfer
  void updateSelf() {
    #if defined( ENABLE_OPENMP_OFFLOAD )
      #pragma omp target update from(data_[0:size_])
    #endif
  };
 
  void updateDevice() {
    #if defined( ENABLE_OPENMP_OFFLOAD )
      #pragma omp target update to(data_[0:size_])
    #endif
  };

  void fill(const value_type value = 0) {
    std::fill(data_, data_+size_, value);
    updateDevice();
  }

  // Exchange Meta data with another view (no deep copy)
  void swap(View &rhs) {
    // Temporal data
    std::string name      = this->name_;
    bool is_empty         = this->is_empty_;
    bool is_copied        = this->is_copied_;
    int_type *strides     = this->strides_;
    value_type *data      = this->data_;
    int_type total_offset = this->total_offset_;
    auto extents          = this->extents();
    auto offsets          = this->offsets();
    auto ends             = this->ends();
    size_type size        = this->size_;

    // Update this
    this->setName(rhs.name());
    this->setIsEmpty(rhs.is_empty());
    this->setIsCopied(rhs.is_copied());
    this->setData(rhs.data()); // attach the data pointer
    this->setStrides(rhs.strides()); // attach the strides pointer
    this->setTotalOffset(rhs.total_offset());
    this->setExtents(rhs.extents());
    this->setOffsets(rhs.offsets()); 
    this->setEnds(rhs.ends());
    this->setSize(rhs.size());
    
    // Update the rhs
    rhs.setName(name);
    rhs.setIsEmpty(is_empty);
    rhs.setIsCopied(is_copied);
    rhs.setSize(size);
    rhs.setData(data); // attach the data pointer
    rhs.setStrides(strides); // attach the strides pointer
    rhs.setTotalOffset(total_offset);
    rhs.setExtents(extents);
    rhs.setOffsets(offsets);
    rhs.setEnds(ends);
  }

public:

  // Getters
  const std::string name() const noexcept {return name_;}
  bool is_empty() const {return is_empty_;}
  bool is_copied() const {return is_copied_;}
  size_type size() const {return size_;}
  size_type rank() const {return rank_;}
  value_type *&data() {return data_;}
  value_type *data() const {return data_;}
  int_type *&strides() {return strides_;}
  int_type *strides() const noexcept {return strides_;}

  const std::array<size_type, ND> extents() const noexcept { return extents_; }
  size_type extent(size_type r) const noexcept { return extents_[r]; }
  int_type total_offset() const noexcept { return total_offset_; }
  std::array<int_type, ND> offsets() const noexcept { return offsets_; }
  int_type offsets(size_type r) const noexcept { return offsets_[r]; }

  std::array<int_type, ND> begins() const noexcept { return offsets_; }
  std::array<int_type, ND> ends() const noexcept { return ends_; }
  int_type begin(size_type r) const noexcept { return offsets_[r]; }
  int_type end(size_type r) const noexcept { return ends_[r]; }

  // Setters
  inline void setName(const std::string &name) { name_ = name; }
  inline void setSize(const size_type size) { size_ = size; }
  inline void setIsEmpty(bool is_empty) { is_empty_ = is_empty; }
  inline void setIsCopied(bool is_copied) { is_copied_ = is_copied; }
  inline void setTotalOffset(const int_type total_offset) { total_offset_ = total_offset; }
  inline void setExtents(const std::array<size_type, ND>& extents) { extents_ = extents; }
  inline void setOffsets(const std::array<int_type, ND>& offsets) { offsets_ = offsets; }
  inline void setEnds(const std::array<int_type, ND>& ends) { ends_ = ends; }
  inline void setData(value_type* data) {
    data_ = data;
    omp_attach( (void**) &data_ );
  }

  inline void setStrides(int_type* strides) {
    strides_ = strides;
    omp_attach( (void**) &strides_ );
  }

private:
  void init(std::array<size_type, ND> extents,
            std::array<int_type, ND> offsets) {
    extents_ = extents;
    size_ = std::accumulate(extents.begin(), extents.end(), 1, std::multiplies<>());
    std::transform(offsets.begin(), offsets.end(), extents.begin(), ends_.begin(), std::plus<int_type>());

    // Allocate and initialize strides
    strides_ = new int_type[rank_];

    // Compute offsets
    int_type total_offset = 0;
    size_type total_extents = 1;
    if(std::is_same_v<layout_type, stdex::layout_left>) {
      for(size_type i=0; i<rank_; i++) {
        total_offset -= offsets[i] * total_extents;
        total_extents *= extents[i];
        strides_[i] = total_extents;
      }
    } else {
      for(size_type i=0; i<rank_; i++) {
        total_offset -= offsets[rank_-1-i] * total_extents;
        total_extents *= extents[rank_-1-i];
        strides_[rank_-1-i] = total_extents;
      }
    }

    total_offset_ = total_offset;

    // Allocate the data
    data_ = new value_type[size_];
    
    #if defined( ENABLE_OPENMP_OFFLOAD )
      #pragma omp target enter data map(alloc: this[0:1])
      #pragma omp target enter data map(alloc: data_[0:size_], strides_[0:rank_], total_offset_)
      #pragma omp target update to(strides_[0:rank_], total_offset_)
    #endif
  }

  void free() {
    if( !is_empty_ ) {
      if( is_copied_) {
        #if defined( ENABLE_OPENMP_OFFLOAD )
          omp_detach( (void**) &data_ );
          omp_detach( (void**) &strides_ );
        #endif
      } else {
        #if defined( ENABLE_OPENMP_OFFLOAD )
          #pragma omp target exit data map(delete: data_[0:size_], strides_[0:rank_], total_offset_)
        #endif
        if(data_    != nullptr) delete [] data_;
        if(strides_ != nullptr) delete [] strides_;
      }
      name_ = "";
      is_empty_ = true;
      is_copied_ = false;
      data_    = nullptr;
      strides_ = nullptr;
      total_offset_ = 0;
      size_ = 0;
      extents_.fill(0);
      offsets_.fill(0);
      ends_.fill(0);
      #if defined( ENABLE_OPENMP_OFFLOAD )
        #pragma omp target exit data map(delete: this[0:1])
      #endif
    }
  }

  // Naive accessors to ease the compiler optimizations
  // For LayoutLeft
  template <typename I0, class L=LayoutPolicy>
  inline typename std::enable_if_t<std::is_same_v<L, stdex::layout_left>, ScalarType&>
  access(I0 i0) const noexcept {
    return data_[total_offset_ + i0];
  }

  template <typename I0, typename I1, class L=LayoutPolicy>
  inline typename std::enable_if_t<std::is_same_v<L, stdex::layout_left>, ScalarType&>
  access(I0 i0, I1 i1) const noexcept {
    int_type idx = total_offset_ + i0 + i1 * strides_[0];
    return data_[idx];
  }
  
  template <typename I0, typename I1, typename I2, class L=LayoutPolicy>
  inline typename std::enable_if_t<std::is_same_v<L, stdex::layout_left>, ScalarType&>
  access(I0 i0, I1 i1, I2 i2) const noexcept {
    int_type idx = total_offset_ + i0 + i1 * strides_[0] + i2 * strides_[1];
    return data_[idx];
  }
  
  template <typename I0, typename I1, typename I2, typename I3, class L=LayoutPolicy>
  inline typename std::enable_if_t<std::is_same_v<L, stdex::layout_left>, ScalarType&>
  access(I0 i0, I1 i1, I2 i2, I3 i3) const noexcept {
    int_type idx = total_offset_ + i0 + i1 * strides_[0] + i2 * strides_[1] + i3 * strides_[2];
    return data_[idx];
  }

  // For LayoutRight
  template <typename I0, class L=LayoutPolicy>
  inline typename std::enable_if_t<std::is_same_v<L, stdex::layout_right>, ScalarType&>
  access(I0 i0) const noexcept {
    return data_[total_offset_ + i0];
  }

  template <typename I0, typename I1, class L=LayoutPolicy>
  inline typename std::enable_if_t<std::is_same_v<L, stdex::layout_right>, ScalarType&>
  access(I0 i0, I1 i1) const noexcept {
    int_type idx = total_offset_ + i1 + i0 * strides_[1];
    return data_[idx];
  }

  template <typename I0, typename I1, typename I2, class L=LayoutPolicy>
  inline typename std::enable_if_t<std::is_same_v<L, stdex::layout_right>, ScalarType&>
  access(I0 i0, I1 i1, I2 i2) const noexcept {
    int_type idx = total_offset_ + i2 + i1 * strides_[2] + i0 * strides_[1];
    return data_[idx];
  }

  template <typename I0, typename I1, typename I2, typename I3, class L=LayoutPolicy>
  inline typename std::enable_if_t<std::is_same_v<L, stdex::layout_right>, ScalarType&>
  access(I0 i0, I1 i1, I2 i2, I3 i3) const noexcept {
    int_type idx = total_offset_ + i3 + i2 * strides_[3] + i1 * strides_[2] + i0 * strides_[1];
    return data_[idx];
  }
};

#endif

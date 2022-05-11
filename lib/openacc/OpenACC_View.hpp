/*
 *  Simplified View to hand multidimensional array inside OpenMP offload region
 */

#ifndef __OPENACC_VIEW_HPP__
#define __OPENACC_VIEW_HPP__

#include <string>
#include <type_traits>
#include <array>
#include <experimental/mdspan>

#if defined( ENABLE_OPENACC )
  #include <openacc.h>
#endif

namespace stdex = std::experimental;

template <size_t ND>
using shape_nd = std::array<int, ND>;

template <typename ScalarType, size_t ND, class LayoutPolicy=stdex::layout_left>
class View {
  std::string name_;

  // Do not delete the pointers, if this instance is a copy
  bool is_copied_;

  // In case instanized with default constructor
  bool is_empty_;

  // Raw data
  ScalarType *data_;

  // Meta data used in offload region
  int *strides_;
  int total_offset_;

  // Used outside offload region
  shape_nd<ND> strides_meta_;
  shape_nd<ND> offsets_meta_;
  shape_nd<ND> end_meta_;
  size_t size_; // total data size
  size_t dims_ = ND;

public:
  using value_type = ScalarType;
  using layout_type = LayoutPolicy;

public:
  // Default constructor, define an empty view
  View() : name_("empty"), total_offset_(0), size_(0), data_(nullptr), strides_(nullptr),
           strides_meta_ {0}, offsets_meta_ {0}, end_meta_ {0}, is_copied_(false), is_empty_(true)
  {}

  // Constructor instanized with shape_nd
  View(const std::string name, const shape_nd<ND>& strides)
    : name_(name), total_offset_(0), strides_meta_(strides), is_copied_(false), is_empty_(false) {
    offsets_meta_.fill(0);
    init();
  }

  // Kokkos like constructor
  template <typename... I>
  View(const std::string name, I... indices)
    : name_(name), total_offset_(0), is_copied_(false), is_empty_(false) {
    static_assert(sizeof...(I) == ND, "The number of indices must be equal to ND");
    offsets_meta_.fill(0);

    // expand parameter packs in the initializer
    int indices_tmp[ND] = {indices...};
    for(int i=0; i<ND; i++) {
      strides_meta_[i] = indices_tmp[i];
    }

    init();
  }

  // View with offsets
  View(const std::string name, const shape_nd<ND>& strides, const shape_nd<ND>& offsets)
    : name_(name), total_offset_(0), strides_meta_(strides), offsets_meta_(offsets), is_copied_(false), is_empty_(false) {
    init();
  }

  ~View() { free(); }

  // Copy construct performs shallow copy
  View(const View &rhs) { copy(rhs); }

  // Assignment operator used only for data allocation, values are not copied
  View& operator=(const View &rhs) {
    this->is_empty_  = false;
    this->is_copied_ = false;
    this->strides_  = new int[rhs.dims()];
    this->data_     = new ScalarType[rhs.size()];
    this->size_     = rhs.size();
    this->dims_     = rhs.dims();
    this->name_     = rhs.name() + "_assign";
    this->total_offset_ = rhs.total_offsets();

    // copy meta data
    for(int i=0; i<ND; i++) {
      this->strides_meta_[i] = rhs.strides_meta()[i];
      this->offsets_meta_[i] = rhs.offsets()[i];
      this->end_meta_[i]     = rhs.end()[i];
      this->strides_[i]      = rhs.strides()[i];
    }

    #if defined( ENABLE_OPENACC )
      #pragma acc enter data copyin(this)
      #pragma acc enter data create(data_[0:size_], strides_[0:dims_])
      #pragma acc update device(strides_[0:dims_])
    #endif

    return *this;
  }

private:
  void init() {
    // allocate and initialize strides
    strides_ = new int[ND];
    for(int i=0; i<ND; i++) {
      end_meta_[i] = strides_meta_[i] + offsets_meta_[i];
    }

    if(std::is_same_v<layout_type, stdex::layout_left>) {
      int total_strides = 1;
      for(int i=0; i<ND; i++) {
        total_strides *= strides_meta_[i];
        strides_[i] = total_strides;
      }
    } else {
      int total_strides = 1;
      for(int i=0; i<ND; i++) {
        total_strides *= strides_meta_[ND-1-i];
        strides_[ND-1-i] = total_strides;
      }
    }

    // allocate raw pointer
    size_t sz = 1;
    for(auto&& dim: strides_meta_) {
      sz *= dim;
    }
    size_ = sz;
    data_ = new ScalarType[sz];

    // compute the total offsets
    int offset = 0;
    if(std::is_same_v<layout_type, stdex::layout_left>) {
      offset -= offsets_meta_[0];
      int total_strides = 1;
      for(int i=0; i<ND-1; i++) {
        total_strides *= strides_meta_[i];
        offset -= offsets_meta_[i+1] * total_strides;
      }
    } else {
      offset -= offsets_meta_[ND-1];
      int total_strides = 1;
      for(int i=1; i<ND; i++) {
        total_strides *= strides_meta_[ND-i];
        offset -= offsets_meta_[ND-1-i] * total_strides;
      }
    }
    total_offset_ = offset;
    #if defined( ENABLE_OPENACC )
      #pragma acc enter data copyin(this)
      #pragma acc enter data create(data_[0:size_], strides_[0:dims_])
      #pragma acc update device(strides_[0:dims_])
    #endif
  }

  void free() {
    if(! (is_empty_ || is_copied_) ){
      // In case, this is not a copy, deallocate the data
      #if defined( ENABLE_OPENACC )
        #pragma acc exit data delete(data_[0:size_], strides_[0:dims_]) // detach data
      #endif
      if(data_    != nullptr) delete [] data_;
      if(strides_ != nullptr) delete [] strides_;

      data_    = nullptr;
      strides_ = nullptr;

      #if defined( ENABLE_OPENACC )
        #pragma acc exit data delete(this) // delete this pointer
      #endif
    }
  }

public:
  template <typename... I>
  inline ScalarType& operator()(I... indices) const noexcept {
    static_assert(sizeof...(I) == ND, "The number of indices must be equal to ND");
    return access(indices...);
  }

  template <typename... I>
  inline ScalarType& operator[](I... indices) const noexcept {
    static_assert(sizeof...(I) == ND, "The number of indices must be equal to ND");
    return access(indices...);
  }

  // methods for device/host data transfer
  void updateSelf() {
    #if defined( ENABLE_OPENACC  )
      #pragma acc update self(data_[0:size_])
    #endif
  };
 
  void updateDevice() {
    #if defined( ENABLE_OPENACC )
      #pragma acc update device(data_[0:size_])
    #endif
  };

  void fill(const ScalarType value = 0) {
    for(int i=0; i<size_; i++) {
      data_[i] = value;
    }
    updateDevice();
  }

  void copyData(ScalarType *data) {
    for(int i=0; i<size_; i++) {
      data_[i] = data[i];
    }
  }

  // Shallow copy the rhs value
  // If destructor called, delete the reference only
  void copy(const View &rhs) {
    this->is_copied_ = true;
    setSize(rhs.size());
    setDims(rhs.dims());
    setData(rhs.data()); // attach the data pointer
    setStrides(rhs.strides()); // attach the strides pointer
    setStridesMeta(rhs.strides_meta());
    setOffsetsMeta(rhs.offsets());
    setTotalOffsets(rhs.total_offsets());
    setEndMeta(rhs.end());
    setName(rhs.name() + "_copy");
  }

  // Exchange Meta data with another view (no deep copy)
  void swap(View &rhs) {
    // Temporal data
    bool is_copied       = this->is_copied_;
    std::string name     = this->name_;
    int *strides         = this->strides_;
    ScalarType *data     = this->data_;
    int total_offset     = this->total_offset_;
    shape_nd<ND> strides_meta = this->strides_meta_;
    shape_nd<ND> offsets_meta = this->offsets_meta_;
    shape_nd<ND> end_meta     = this->end_meta_;
    size_t size = this->size_;
    size_t dims = this->dims_;

    // Update this
    this->setIsCopied(rhs.is_copied());
    this->setSize(rhs.size());
    this->setDims(rhs.dims());
    this->setData(rhs.data()); // attach the data pointer
    this->setStrides(rhs.strides()); // attach the strides pointer
    this->setStridesMeta(rhs.strides_meta());
    this->setOffsetsMeta(rhs.offsets()); 
    this->setTotalOffsets(rhs.total_offsets());
    this->setEndMeta(rhs.end());
    this->setName(rhs.name());
    
    // Update the rhs
    rhs.setIsCopied(is_copied);
    rhs.setSize(size);
    rhs.setDims(dims);
    rhs.setData(data); // attach the data pointer
    rhs.setStridesMeta(strides_meta);
    rhs.setStrides(strides); // attach the strides pointer
    rhs.setOffsetsMeta(offsets_meta);
    rhs.setTotalOffsets(total_offset);
    rhs.setEndMeta(end_meta);
    rhs.setName(name);
  }
  
  // This method overwrites the original view by rhs
  void clone(View &rhs) {
    // Delete all the data
    this->free();

    // Update this
    this->setIsCopied(rhs.is_copied());
    this->setStrides_meta(rhs.strides_meta());

    // allocate data
    this->init();

    // Copy and memcpy H2D
    this->copyData(rhs.data());
    this->updateDevice();
  }

public:

  // Getters
  bool is_empty() const {return is_empty_;}
  bool is_copied() const {return is_copied_;}
  size_t size() const {return size_;}
  size_t dims() const {return dims_;}
  ScalarType *&data() {return data_;}
  ScalarType *data() const {return data_;}
  int *strides() const noexcept {return strides_;}

  inline const shape_nd<ND>& strides_meta() const noexcept {return strides_meta_;}
  inline const shape_nd<ND>& offsets() const noexcept {return offsets_meta_;}
  inline const shape_nd<ND>& begin() const noexcept {return offsets_meta_;}
  inline const shape_nd<ND>& end() const noexcept {return end_meta_;}
  inline int strides(size_t i) const noexcept {return strides_meta_[i];}
  inline int offsets(size_t i) const noexcept {return offsets_meta_[i];}
  inline int begin(size_t i) const noexcept {return offsets_meta_[i];}
  inline int end(size_t i) const noexcept {return end_meta_[i];}
  inline int total_offsets() const noexcept {return total_offset_;}
  std::string name() const noexcept {return name_;}

  // Setters
  inline void setIsCopied(bool is_copied) {is_copied_ = is_copied;}
  inline void setSize(size_t size) {size_ = size;}
  inline void setDims(size_t dims) {dims_ = dims;}

  inline void setData(ScalarType *data) { 
    data_ = data; 
    #if defined( ENABLE_OPENACC )
      acc_attach( (void**) &data_ );
    #endif
  }
  inline void setStrides(int *strides) {
    strides_ = strides;
    #if defined( ENABLE_OPENACC )
      acc_attach( (void**) &strides_ );
    #endif
  }
  inline void setStridesMeta(const shape_nd<ND>& strides_meta) {
    strides_meta_ = strides_meta;
  }
  inline void setOffsetsMeta(const shape_nd<ND>& offsets_meta) {
    offsets_meta_ = offsets_meta;
  }
  inline void setEndMeta(const shape_nd<ND>& end_meta) {
    end_meta_ = end_meta;
  }
  inline void setTotalOffsets(const int total_offset) {
    total_offset_ = total_offset;
  }
  inline void setName(std::string name) {
    name_ = name;
  }

private:
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
    int idx = total_offset_ + i0 + i1 * strides_[0];
    return data_[idx];
  }
  
  template <typename I0, typename I1, typename I2, class L=LayoutPolicy>
  inline typename std::enable_if_t<std::is_same_v<L, stdex::layout_left>, ScalarType&>
  access(I0 i0, I1 i1, I2 i2) const noexcept {
    int idx = total_offset_ + i0 + i1 * strides_[0] + i2 * strides_[1];
    return data_[idx];
  }
  
  template <typename I0, typename I1, typename I2, typename I3, class L=LayoutPolicy>
  inline typename std::enable_if_t<std::is_same_v<L, stdex::layout_left>, ScalarType&>
  access(I0 i0, I1 i1, I2 i2, I3 i3) const noexcept {
    int idx = total_offset_ + i0 + i1 * strides_[0] + i2 * strides_[1] + i3 * strides_[2];
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
    int idx = total_offset_ + i1 + i0 * strides_[1];
    return data_[idx];
  }

  template <typename I0, typename I1, typename I2, class L=LayoutPolicy>
  inline typename std::enable_if_t<std::is_same_v<L, stdex::layout_right>, ScalarType&>
  access(I0 i0, I1 i1, I2 i2) const noexcept {
    int idx = total_offset_ + i2 + i1 * strides_[2] + i0 * strides_[1];
    return data_[idx];
  }

  template <typename I0, typename I1, typename I2, typename I3, class L=LayoutPolicy>
  inline typename std::enable_if_t<std::is_same_v<L, stdex::layout_right>, ScalarType&>
  access(I0 i0, I1 i1, I2 i2, I3 i3) const noexcept {
    int idx = total_offset_ + i3 + i2 * strides_[3] + i1 * strides_[2] + i0 * strides_[1];
    return data_[idx];
  }
};

#endif

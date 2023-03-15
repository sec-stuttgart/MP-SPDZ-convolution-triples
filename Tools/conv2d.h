#pragma once

#include <compare>
#include <span>
#include <vector>
#include <tuple>
#include <string>

#include "Tools/description.hpp"

template<typename T>
struct bounded_index
{
  T index;
  T bound;
};

template<typename T>
bounded_index<T> last_index(T bound)
{
  return { bound - 1, bound };
}

template<typename T>
bounded_index<T> first_index(T bound)
{
  return { 0, bound };
}

template<typename T>
bounded_index<T> reverse_index(T value, T bound)
{
  return { bound - value - 1, bound };
}

template<typename T>
bounded_index<T> reverse_index(T value, T max, T bound)
{
  return { max - value - 1, bound };
}

template<typename... Ts>
auto nDaccess_impl(bounded_index<Ts>... args)
{
  constexpr int N = sizeof...(Ts);

  static_assert(N > 0);

  using T = std::common_type_t<Ts...>;

  std::array<bounded_index<T>, N> arguments = { args... };

  T result = arguments[N-1].index;
#ifdef ASSERTIVE_CONV2D
  for (int i = 0; i < N; ++i)
  {
    CONV2D_ASSERT(0 <= arguments[i].index);
    CONV2D_ASSERT(arguments[i].index < arguments[i].bound);
  }
#endif

  if constexpr (N > 1)
  {
    T covered_elements = arguments[N-1].bound;

    for (int i = N - 2; i >= 0; --i)
    {
      result += arguments[i].index * covered_elements;
      covered_elements *= arguments[i].bound;
    }
  }
  return result;
}

template<typename T0 = int>
auto nDaccess(bounded_index<T0> arg0)
{
  return nDaccess_impl(arg0);
}
template<typename T0 = int, typename T1 = int>
auto nDaccess(bounded_index<T0> arg0, bounded_index<T1> arg1)
{
  return nDaccess_impl(arg0, arg1);
}
template<typename T0 = int, typename T1 = int, typename T2 = int>
auto nDaccess(bounded_index<T0> arg0, bounded_index<T1> arg1, bounded_index<T2> arg2)
{
  return nDaccess_impl(arg0, arg1, arg2);
}
template<typename T0 = int, typename T1 = int, typename T2 = int, typename T3 = int>
auto nDaccess(bounded_index<T0> arg0, bounded_index<T1> arg1, bounded_index<T2> arg2, bounded_index<T3> arg3)
{
  return nDaccess_impl(arg0, arg1, arg2, arg3);
}
template<typename T0 = int, typename T1 = int, typename T2 = int, typename T3 = int, typename T4 = int>
auto nDaccess(bounded_index<T0> arg0, bounded_index<T1> arg1, bounded_index<T2> arg2, bounded_index<T3> arg3, bounded_index<T4> arg4)
{
  return nDaccess_impl(arg0, arg1, arg2, arg3, arg4);
}
template<typename T0 = int, typename T1 = int, typename T2 = int, typename T3 = int, typename T4 = int, typename T5 = int>
auto nDaccess(bounded_index<T0> arg0, bounded_index<T1> arg1, bounded_index<T2> arg2, bounded_index<T3> arg3, bounded_index<T4> arg4, bounded_index<T5> arg5)
{
  return nDaccess_impl(arg0, arg1, arg2, arg3, arg4, arg5);
}

constexpr int CONVOLUTION_NO_IMAGE_DEPTH = -1;
constexpr int CONVOLUTION_NO_OUTPUT_DEPTH = -1;

struct depthwise_convolution_triple_dimensions
{
  static constexpr bool is_always_depthwise = true;

  int image_batch;
  int image_height;
  int image_width;
  int filter_height;
  int filter_width;

  auto full_output_height() const { return image_height + filter_height - 1; }
  auto full_output_width() const { return image_width + filter_width - 1; }
  auto full_output_area() const { return full_output_height() * full_output_width(); }

  auto image_size() const { return image_batch * image_height * image_width; }
  auto filter_size() const { return filter_height * filter_width; }
  auto full_output_size() const { return image_batch * full_output_area(); }

  std::vector<int> image_sparcity(int N) const;
  std::vector<int> filter_sparcity(int N) const;

  depthwise_convolution_triple_dimensions split(int split_y, int split_x) const
  {
    CONV2D_ASSERT(image_height % split_y == 0);
    CONV2D_ASSERT(image_width % split_x == 0);
    return depthwise_convolution_triple_dimensions{image_batch * split_x * split_y, image_height / split_y, image_width / split_x, filter_height, filter_width};
  }

  std::string as_string(std::string const& separator = " * ") const;

  std::tuple<int /*batches_per_convolution*/, int /*batches_required*/, int /*outputs_per_convolution*/> depthwise_split(int N) const;

  std::strong_ordering operator<=>(depthwise_convolution_triple_dimensions const&) const = default;
};

struct depthwise_convolution_dimensions
{
  int image_batch;
  int image_height;
  int image_width;
  int image_depth;
  int filter_height;
  int filter_width;

  auto full_output_height() const { return image_height + filter_height - 1; }
  auto full_output_width() const { return image_width + filter_width - 1; }
  auto full_output_area() const { return full_output_height() * full_output_width(); }
  
  auto image_size() const { return image_batch * image_height * image_width * image_depth; }
  auto filter_size() const { return filter_height * filter_width * image_depth; }
  auto output_size(int height, int width) const { return image_batch * height * width * image_depth; }
  auto full_output_size() const { return output_size(full_output_width(), full_output_width()); }

  explicit operator depthwise_convolution_triple_dimensions() const
  {
    return depthwise_convolution_triple_dimensions{image_batch, image_height, image_width, filter_height, filter_width};
  }

  std::strong_ordering operator<=>(depthwise_convolution_dimensions const&) const = default;
};

struct convolution_dimensions : private depthwise_convolution_dimensions
{
  static constexpr bool is_always_depthwise = false;

  using depthwise_convolution_dimensions::image_batch;
  using depthwise_convolution_dimensions::image_height;
  using depthwise_convolution_dimensions::image_width;
  using depthwise_convolution_dimensions::image_depth;
  using depthwise_convolution_dimensions::filter_height;
  using depthwise_convolution_dimensions::filter_width;
  int output_depth;

  convolution_dimensions() = default;
  explicit convolution_dimensions(depthwise_convolution_triple_dimensions dimensions)
    : depthwise_convolution_dimensions{dimensions.image_batch, dimensions.image_height, dimensions.image_width, CONVOLUTION_NO_IMAGE_DEPTH, dimensions.filter_height, dimensions.filter_width}
    , output_depth(CONVOLUTION_NO_OUTPUT_DEPTH)
  {
  }
  convolution_dimensions(int image_batch, int image_height, int image_width, int image_depth, int filter_height, int filter_width, int output_depth)

    : depthwise_convolution_dimensions{image_batch, image_height, image_width, image_depth, filter_height, filter_width}
    , output_depth(output_depth)
  {
  }

  using depthwise_convolution_dimensions::full_output_height;
  using depthwise_convolution_dimensions::full_output_width;
  using depthwise_convolution_dimensions::full_output_area;

  using depthwise_convolution_dimensions::image_size;
  auto filter_size() const { return output_depth * filter_height * filter_width * image_depth; }
  auto output_size(int height, int width) const { return image_batch * height * width * output_depth; }
  auto full_output_size() const { return output_size(full_output_width(), full_output_width()); }

  std::vector<int> image_sparcity(int N) const;
  std::vector<int> filter_sparcity(int N) const;
  std::vector<int> output_sparcity(int N) const;

  std::string as_string(std::string const& separator = " * ") const;

#ifdef CONV2D_DIRECT_SUM
  std::tuple<int /*batches_per_convolution*/, int /*batches_required*/, int /*outputs_per_convolution*/, int /*outputs_required*/, int /*inputs_per_convolution*/, int /*inputs_required*/> direct_sum_split(int N) const;
#endif

  bool is_depthwise() const { return output_depth == CONVOLUTION_NO_OUTPUT_DEPTH; }

  depthwise_convolution_triple_dimensions as_depthwise() const
  {
    CONV2D_ASSERT(is_depthwise());
    return static_cast<depthwise_convolution_triple_dimensions>(*this);
  }

  convolution_dimensions split(int split_y, int split_x) const
  {
    CONV2D_ASSERT(image_height % split_y == 0);
    CONV2D_ASSERT(image_width % split_x == 0);
    return convolution_dimensions{image_batch * split_x * split_y, image_height / split_y, image_width / split_x, image_depth, filter_height, filter_width, output_depth};
  }

  std::strong_ordering operator<=>(convolution_dimensions const&) const = default;
};

struct depthwise_convolution_desc
{
  using dimension_type = depthwise_convolution_dimensions;

  int output_address;
  int image_address;
  int filter_address;
  int padding_y;
  int padding_x;
  int stride_y;
  int stride_x;
  int image_batch;
  int image_height;
  int image_width;
  int image_depth;
  int filter_height;
  int filter_width;
  int output_height;
  int output_width;

  auto full_output_height() const { return image_height + filter_height - 1; }
  auto full_output_width() const { return image_width + filter_width - 1; }
  auto image_size() const { return image_batch * image_height * image_width * image_depth; }
  auto filter_size() const { return filter_height * filter_width * image_depth; }
  auto output_size() const { return image_batch * output_height * output_width * image_depth; }

  explicit operator depthwise_convolution_dimensions() const
  {
    return depthwise_convolution_dimensions{image_batch, image_height, image_width, image_depth, filter_height, filter_width};
  }
};

struct convolution_desc : private depthwise_convolution_desc
{
  using dimension_type = convolution_dimensions;

  using depthwise_convolution_desc::output_address;
  using depthwise_convolution_desc::image_address;
  using depthwise_convolution_desc::filter_address;
  using depthwise_convolution_desc::padding_y;
  using depthwise_convolution_desc::padding_x;
  using depthwise_convolution_desc::stride_y;
  using depthwise_convolution_desc::stride_x;
  using depthwise_convolution_desc::image_batch;
  using depthwise_convolution_desc::image_height;
  using depthwise_convolution_desc::image_width;
  using depthwise_convolution_desc::image_depth;
  using depthwise_convolution_desc::filter_height;
  using depthwise_convolution_desc::filter_width;
  using depthwise_convolution_desc::output_height;
  using depthwise_convolution_desc::output_width;
  int output_depth;

  using depthwise_convolution_desc::full_output_height;
  using depthwise_convolution_desc::full_output_width;
  using depthwise_convolution_desc::image_size;
  auto filter_size() const { return output_depth * filter_height * filter_width * image_depth; }
  auto output_size() const { return image_batch * output_height * output_width * output_depth; }

  explicit operator convolution_dimensions() const
  {
    return convolution_dimensions{image_batch, image_height, image_width, image_depth, filter_height, filter_width, output_depth};
  }

  bool is_depthwise() const { return output_depth == CONVOLUTION_NO_OUTPUT_DEPTH; }
  
  depthwise_convolution_desc as_depthwise() const
  {
    CONV2D_ASSERT(is_depthwise());
    return *this;
  }
};

constexpr int CONVOLUTION_DESC_FIELDS = 16;
constexpr int DEPTHWISE_CONVOLUTION_DESC_FIELDS = 15;
constexpr int CONVOLUTION_DIMENSIONS_FIELDS = 7;
constexpr int DEPTHWISE_CONVOLUTION_DIMENSIONS_FIELDS = 6;

static_assert(sizeof(convolution_desc) == CONVOLUTION_DESC_FIELDS * sizeof(int));
static_assert(sizeof(depthwise_convolution_desc) == DEPTHWISE_CONVOLUTION_DESC_FIELDS * sizeof(int));
static_assert(sizeof(convolution_dimensions) == CONVOLUTION_DIMENSIONS_FIELDS * sizeof(int));
static_assert(sizeof(depthwise_convolution_dimensions) == DEPTHWISE_CONVOLUTION_DIMENSIONS_FIELDS * sizeof(int));

using convolution_desc_range = description_range<convolution_desc, CONVOLUTION_DESC_FIELDS>;

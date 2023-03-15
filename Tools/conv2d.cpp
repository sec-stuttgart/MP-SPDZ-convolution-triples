#include "Tools/conv2d.h"

#include <Tools/int.h>

std::vector<int> depthwise_convolution_triple_dimensions::image_sparcity(int N) const
{
    std::vector<int> sparcity(N, 1);
    auto H = full_output_height();
    auto W = full_output_width();
    auto [batches_per_convolution, batches_required, outputs_per_convolution] = depthwise_split(N);
    for (int b = 0; b < batches_per_convolution; ++b)
    {
      for (int c = 0; c < outputs_per_convolution; ++c)
      {
        for (int y = 0; y < image_height; ++y)
        {
          for (int x = 0; x < image_width; ++x)
          {
#ifdef CONV2D_LOWGEAR_EXPANDED_BGV
            auto index = nDaccess({c, outputs_per_convolution}, {b, batches_per_convolution}, {y, H}, {x, W});
#else
            auto index = nDaccess({b, batches_per_convolution}, first_index(outputs_per_convolution), {c, outputs_per_convolution}, {y, H}, {x, W});
#endif
            CONV2D_ASSERT(index >= 0);
            CONV2D_ASSERT(index < N);
            sparcity[index] = 0;
          }
        }
      }
    }
    return sparcity;
}

std::vector<int> depthwise_convolution_triple_dimensions::filter_sparcity(int N) const
{
    std::vector<int> sparcity(N, 1);
    auto H = full_output_height();
    auto W = full_output_width();
    auto [batches_per_convolution, batches_required, outputs_per_convolution] = depthwise_split(N);
    for (int c = 0; c < outputs_per_convolution; ++c)
    {
      for (int y = 0; y < filter_height; ++y)
      {
        for (int x = 0; x < filter_width; ++x)
        {
          auto index = nDaccess(/*first_index(batches_per_convolution),*/ {c, outputs_per_convolution}, first_index(outputs_per_convolution), {y, H}, {x, W});
          CONV2D_ASSERT(index >= 0);
          CONV2D_ASSERT(index < N);
          sparcity[index] = 0;
        }
      }
    }
    return sparcity;
}

std::vector<int> convolution_dimensions::image_sparcity(int N) const 
{
    std::vector<int> sparcity(N, 1);
    auto H = full_output_height();
    auto W = full_output_width();
#ifdef CONV2D_DIRECT_SUM
    auto [batches_per_convolution, batches_required, outputs_per_convolution, outputs_required, inputs_per_convolution, inputs_required] = direct_sum_split(N);
    for (int b = 0; b < batches_per_convolution; ++b)
    {
      for (int c = 0; c < inputs_per_convolution; ++c)
      {
        for (int y = 0; y < image_height; ++y)
        {
          for (int x = 0; x < image_width; ++x)
          {
            auto index = nDaccess({b, batches_per_convolution}, first_index(outputs_per_convolution), {c, inputs_per_convolution}, {y, H}, {x, W});
            CONV2D_ASSERT(index >= 0);
            CONV2D_ASSERT(index < N);
            sparcity[index] = 0;
          }
        }
      }
    }
#else
    auto filter_count = N / (W * H);
    assert(filter_count >= 1);
    for (int i = 0; i < image_height; ++i)
    {
      for (int j = 0; j < image_width; ++j)
      {
        sparcity[i * W + j] = 0;
      }
    }
#endif
    return sparcity;
}

std::vector<int> convolution_dimensions::filter_sparcity(int N) const
{
    std::vector<int> sparcity(N, 1);
    auto H = full_output_height();
    auto W = full_output_width();
#ifdef CONV2D_DIRECT_SUM
    auto [batches_per_convolution, batches_required, outputs_per_convolution, outputs_required, inputs_per_convolution, inputs_required] = direct_sum_split(N);
    for (int d = 0; d < outputs_per_convolution; ++d)
    {
      for (int c = 0; c < inputs_per_convolution; ++c)
      {
        for (int y = 0; y < filter_height; ++y)
        {
          for (int x = 0; x < filter_width; ++x)
          {
            auto index = nDaccess({d, outputs_per_convolution}, {c, inputs_per_convolution}, {y, H}, {x, W});
            CONV2D_ASSERT(index >= 0);
            CONV2D_ASSERT(index < N);
            sparcity[index] = 0;
          }
        }
      }
    }
#else
    auto filter_count = N / (W * H);
    assert(filter_count >= 1);
    for (int k = 0; k < filter_count; ++k)
    {
      for (int i = 0; i < filter_height; ++i)
      {
        for (int j = 0; j < filter_width; ++j)
        {
          sparcity[k * H * W + i * W + j] = 0;
        }
      }
    }
#endif
    return sparcity;
}

std::vector<int> convolution_dimensions::output_sparcity(int N) const
{
    std::vector<int> sparcity(N, 1);
    auto H = full_output_height();
    auto W = full_output_width();
#ifdef CONV2D_DIRECT_SUM
    auto [batches_per_convolution, batches_required, outputs_per_convolution, outputs_required, inputs_per_convolution, inputs_required] = direct_sum_split(N);
    for (int i = 0; i < batches_per_convolution * outputs_per_convolution * inputs_per_convolution * H * W; ++i)
    {
      CONV2D_ASSERT(i < N);
      sparcity[i] = 0;
    }
#else
    auto filter_count = N / (W * H);
    assert(filter_count >= 1);
    for (int i = 0; i < filter_count * W * H; ++i)
    {
      sparcity[i] = 0;
    }
#endif
    return sparcity;
}

std::tuple<int, int, int> depthwise_convolution_triple_dimensions::depthwise_split(int N) const
{
    auto convolution_area = full_output_area();

    int batches = 2 * image_batch;

    int batches_per_convolution;
    int batches_required;
    int outputs_per_convolution;

    auto parallel_convolution_count = [N, convolution_area](int batches)
    {
#ifdef CONV2D_LOWGEAR_EXPANDED_BGV
      return N / (batches * convolution_area);
#else
      int count = 1;
      while (batches * convolution_area * (count + 1) * (count + 1) <= N)
      {
        ++count;
      }
      return count;
#endif
    };

    if (batches * convolution_area <= N)
    {
      batches_per_convolution = batches;
      batches_required = 1;
      outputs_per_convolution = parallel_convolution_count(batches);
    }
    else
    {
      CONV2D_ASSERT(convolution_area <= N);

      for (batches_per_convolution = N / convolution_area; batches % batches_per_convolution != 0; --batches_per_convolution)
      {
        /*empty*/;
      }

      CONV2D_ASSERT(batches % batches_per_convolution == 0);
      CONV2D_ASSERT(batches_per_convolution >= 1);
      batches_required = batches / batches_per_convolution;
      outputs_per_convolution = parallel_convolution_count(batches_per_convolution);
    }

    return std::make_tuple(batches_per_convolution, batches_required, outputs_per_convolution);
}

#ifdef CONV2D_DIRECT_SUM
std::tuple<int, int, int, int, int, int> convolution_dimensions::direct_sum_split(int N) const
{
    auto convolution_area = full_output_area();

    auto output_dimensions = 2 * output_depth;

    auto input_dimensions = image_depth;

    auto batches = image_batch;

    int batches_per_convolution;
    int batches_required;
    int outputs_per_convolution;
    int outputs_required;
    int inputs_per_convolution;
    int inputs_required;

    if (batches * input_dimensions * output_dimensions * convolution_area <= N)
    {
      batches_per_convolution = batches;
      batches_required = 1;
      outputs_per_convolution = output_dimensions;
      outputs_required = 1;
      inputs_per_convolution = input_dimensions;
      inputs_required = 1;
    }
    else if (input_dimensions * output_dimensions * convolution_area <= N)
    {
      batches_per_convolution = N / (input_dimensions * output_dimensions * convolution_area);
      batches_required = DIV_CEIL(batches, batches_per_convolution);
      outputs_per_convolution = output_dimensions;
      outputs_required = 1;
      inputs_per_convolution = input_dimensions;
      inputs_required = 1;
    }
    else if (input_dimensions * convolution_area <= N)
    {
      batches_per_convolution = 1;
      batches_required = batches;
      outputs_per_convolution = N / (input_dimensions * convolution_area);
      outputs_required = DIV_CEIL(output_dimensions, outputs_per_convolution);
      inputs_per_convolution = input_dimensions;
      inputs_required = 1;
    }
    else
    {
      CONV2D_ASSERT(convolution_area <= N);

      for (inputs_per_convolution = N / convolution_area; input_dimensions % inputs_per_convolution != 0; --inputs_per_convolution)
      {
        /*empty*/;
      }
      CONV2D_ASSERT(input_dimensions % inputs_per_convolution == 0);
      CONV2D_ASSERT(inputs_per_convolution >= 1);

      batches_per_convolution = 1;
      batches_required = batches;
      outputs_per_convolution = 1;
      outputs_required = output_dimensions;
      inputs_required = input_dimensions / inputs_per_convolution;
    }

    CONV2D_ASSERT(batches_per_convolution * inputs_per_convolution * outputs_per_convolution * convolution_area <= N);
    CONV2D_ASSERT(batches_per_convolution * batches_required >= batches);
    CONV2D_ASSERT(outputs_per_convolution * outputs_required >= output_dimensions);
    CONV2D_ASSERT(inputs_per_convolution * inputs_required == input_dimensions);

    return std::make_tuple(batches_per_convolution, batches_required, outputs_per_convolution, outputs_required, inputs_per_convolution, inputs_required);
}
#endif

std::string depthwise_convolution_triple_dimensions::as_string(std::string const& separator) const
{
  std::string result = std::to_string(image_batch);
  result += "x";
  result += std::to_string(image_height);
  result += "x";
  result += std::to_string(image_width);
  result += separator;
  result += std::to_string(filter_height);
  result += "x";
  result += std::to_string(filter_width);
  return result;
}

std::string convolution_dimensions::as_string(std::string const& separator) const
{
  std::string result = std::to_string(image_batch);
  result += "x";
  result += std::to_string(image_height);
  result += "x";
  result += std::to_string(image_width);
  result += "x";
  result += std::to_string(image_depth);
  result += separator;
  result += std::to_string(output_depth);
  result += "x";
  result += std::to_string(filter_height);
  result += "x";
  result += std::to_string(filter_width);
  result += "x";
  result += std::to_string(image_depth);
  return result;
}
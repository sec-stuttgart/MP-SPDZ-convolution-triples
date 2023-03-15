#include "Tools/matmul.h"

#include "Tools/config.h"

std::vector<int> matmul_dimensions::left_sparcity(int N) const
{
    std::vector<int> sparcity(N, 1);
#ifdef CONV2D_DIRECT_SUM
    CONV2D_ASSERT(inner_dimension <= N);
    for (int i = 0; i < inner_dimension; ++i)
#else
    CONV2D_ASSERT(left_outer_dimension <= N);
    for (int i = 0; i < left_outer_dimension; ++i)
#endif
    {
        sparcity[i] = 0;
    }
    return sparcity;
}

std::vector<int> matmul_dimensions::right_sparcity(int N) const
{
#ifdef CONV2D_DIRECT_SUM
    return std::vector<int>(N, 0);
#else
    CONV2D_ASSERT(left_outer_dimension <= N);
    auto count = N / left_outer_dimension;
    CONV2D_ASSERT(count > 0);
    std::vector<int> sparcity(N, 1);
    for (int i = 0; i < count; ++i)
    {
        sparcity[i * left_outer_dimension] = 0;
    }
    return sparcity;
#endif
}

std::vector<int> matmul_dimensions::result_sparcity(int N) const
{
#ifdef CONV2D_DIRECT_SUM
    return std::vector<int>(N, 0);
#else
    CONV2D_ASSERT(left_outer_dimension <= N);
    auto count = N / left_outer_dimension;
    CONV2D_ASSERT(count > 0);
    std::vector<int> sparcity(N, 1);
    for (int i = 0; i < count * left_outer_dimension; ++i)
    {
        sparcity[i] = 0;
    }
    return sparcity;
#endif
}

std::string matmul_dimensions::as_string(std::string const& separator) const
{
    std::string result = std::to_string(left_outer_dimension);
    result += "x";
    result += std::to_string(inner_dimension);
    result += separator;
    result += std::to_string(inner_dimension);
    result += "x";
    result += std::to_string(right_outer_dimension);
    return result;
}

#pragma once

#include <compare>
#include <span>
#include <vector>
#include <string>

#include "Tools/description.hpp"

struct matmul_desc
{
    int left_address;
    int right_address;
    int result_address;
    int left_outer_dimension;
    int inner_dimension;
    int right_outer_dimension;

    auto left_size() const { return left_outer_dimension * inner_dimension; }
    auto right_size() const { return inner_dimension * right_outer_dimension; }
    auto result_size() const { return left_outer_dimension * right_outer_dimension; }
};

struct matmul_dimensions
{
    int left_outer_dimension;
    int inner_dimension;
    int right_outer_dimension;

    matmul_dimensions() = default;

    matmul_dimensions(int left_outer_dimension, int inner_dimension, int right_outer_dimension)
        : left_outer_dimension(left_outer_dimension)
        , inner_dimension(inner_dimension)
        , right_outer_dimension(right_outer_dimension)
    {
    }

    matmul_dimensions(matmul_desc desc)
        : left_outer_dimension(desc.left_outer_dimension)
        , inner_dimension(desc.inner_dimension)
        , right_outer_dimension(desc.right_outer_dimension)
    {
    }

    auto left_size() const { return left_outer_dimension * inner_dimension; }
    auto right_size() const { return inner_dimension * right_outer_dimension; }
    auto result_size() const { return left_outer_dimension * right_outer_dimension; }

    std::vector<int> left_sparcity(int N) const;
    std::vector<int> right_sparcity(int N) const;
    std::vector<int> result_sparcity(int N) const;

    std::string as_string(std::string const& separator = " * ") const;

    std::strong_ordering operator<=>(matmul_dimensions const&) const = default;
};

constexpr int MATMUL_DESC_FIELDS = 6;
constexpr int MATMUL_DIMENSIONS_FIELDS = 3;

static_assert(sizeof(matmul_desc) == MATMUL_DESC_FIELDS * sizeof(int));
static_assert(sizeof(matmul_dimensions) == MATMUL_DIMENSIONS_FIELDS * sizeof(int));

using matmul_desc_range = description_range<matmul_desc, MATMUL_DESC_FIELDS>;
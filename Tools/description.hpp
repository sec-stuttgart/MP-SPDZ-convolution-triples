#pragma once

#include <span>
#include <cstring>

#include "Tools/config.h"

template<typename Description, std::size_t FieldCount = sizeof(Description) / sizeof(int)>
struct description_iterator
{
    static_assert(sizeof(Description) == FieldCount * sizeof(int));

    struct sentinel {};

    std::span<int const> data;

    description_iterator& operator++()
    {
        data = data.subspan(FieldCount);
        return *this;
    }

    Description operator*() const
    {
        CONV2D_ASSERT(data.size() >= FieldCount);
        Description result;
        std::memcpy(static_cast<void*>(std::addressof(result)), data.data(), sizeof(Description));
        return result;
    }

    bool operator!=(sentinel) const
    {
        return not data.empty();
    }
};

template<typename Description, std::size_t FieldCount = sizeof(Description) / sizeof(int)>
struct description_range
{
    std::span<int const> data;

    description_iterator<Description, FieldCount> begin() const { return { data }; }
    description_iterator<Description, FieldCount>::sentinel end() const { return {}; }
    std::size_t size() const { return data.size() / FieldCount; }
};

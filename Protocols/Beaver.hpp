/*
 * Beaver.cpp
 *
 */

#ifndef PROTOCOLS_BEAVER_HPP_
#define PROTOCOLS_BEAVER_HPP_

#include "Beaver.h"

#include "Replicated.hpp"

#include <array>

template<class T>
typename T::Protocol Beaver<T>::branch()
{
    typename T::Protocol res(P);
    res.prep = prep;
    res.MC = MC;
    res.init_mul();
    return res;
}

template<class T>
void Beaver<T>::init(Preprocessing<T>& prep, typename T::MAC_Check& MC)
{
    this->prep = &prep;
    this->MC = &MC;
}

template<class T>
void Beaver<T>::init_mul()
{
    assert(this->prep);
    assert(this->MC);
    shares.clear();
    opened.clear();
    triples.clear();
    vtriples.clear();
}

template<class T>
void Beaver<T>::prepare_mul(const T& x, const T& y, int n)
{
    (void) n;
    triples.push_back({{}});
    auto& triple = triples.back();
    triple = prep->get_triple(n);
    shares.push_back(x - triple[0]);
    shares.push_back(y - triple[1]);
}

template<class T>
void Beaver<T>::exchange()
{
    MC->POpen(opened, shares, P);
    it = opened.begin();
    triple = triples.begin();
    vtriple = vtriples.begin();
}

template<class T>
void Beaver<T>::start_exchange()
{
    MC->POpen_Begin(opened, shares, P);
}

template<class T>
void Beaver<T>::stop_exchange()
{
    MC->POpen_End(opened, shares, P);
    it = opened.begin();
    triple = triples.begin();
    vtriple = vtriples.begin();
}

template<class T>
T Beaver<T>::finalize_mul(int n)
{
    (void) n;
    typename T::open_type masked[2];
    T& tmp = (*triple)[2];
    for (int k = 0; k < 2; k++)
    {
        masked[k] = *it++;
    }
    tmp += (masked[0] * (*triple)[1]);
    tmp += ((*triple)[0] * masked[1]);
    tmp += T::constant(masked[0] * masked[1], P.my_num(), MC->get_alphai());
    triple++;
    return tmp;
}

template<class T>
void Beaver<T>::prepare_matmul(std::span<T const> registers, matmul_desc matmul)
{
    auto left_size = matmul.left_size();
    auto right_size = matmul.right_size();

    vtriples.push_back(prep->get_matmul_triple(matmul));
    auto const& triple = vtriples.back();
    auto& [left_mask, right_mask, correlated] = triple;

    for (int i = 0; i < left_size; ++i)
    {
        shares.push_back(registers[matmul.left_address + i] - left_mask[i]);
    }

    for (int i = 0; i < right_size; ++i)
    {
        shares.push_back(registers[matmul.right_address + i] - right_mask[i]);
    }
}

template<class T>
void Beaver<T>::finalize_matmul(std::span<T> registers, matmul_desc matmul)
{
    auto left_size = matmul.left_size();
    auto right_size = matmul.right_size();

    auto masked_left = it;
    auto masked_right = it + left_size;

    auto& [left_mask, right_mask, correlated] = *vtriple;

    for (int i = 0; i < matmul.left_outer_dimension; ++i)
    {
        for (int j = 0; j < matmul.right_outer_dimension; ++j)
        {
            T result = correlated[nDaccess({i, matmul.left_outer_dimension}, {j, matmul.right_outer_dimension})];
            for (int k = 0; k < matmul.inner_dimension; ++k)
            {
                auto left_index = nDaccess({i, matmul.left_outer_dimension}, {k, matmul.inner_dimension});
                auto right_index = nDaccess({k, matmul.inner_dimension}, {j, matmul.right_outer_dimension});
                
                result += masked_right[right_index] * left_mask[left_index];
                result += masked_left[left_index] * right_mask[right_index];
                result += T::constant(masked_left[left_index] * masked_right[right_index], P.my_num(), MC->get_alphai());
            }
            registers[matmul.result_address + nDaccess({i, matmul.left_outer_dimension}, {j, matmul.right_outer_dimension})] = result;
        }
    }

    it += left_size + right_size;
    ++vtriple;
}

template<class T>
template<typename ConvolutionDesc>
void Beaver<T>::prepare_conv2d(std::span<T const> registers, ConvolutionDesc conv)
{
    auto image_size = conv.image_size();
    auto filter_size = conv.filter_size();

    vtriples.push_back(prep->get_conv2d_triple(static_cast<typename ConvolutionDesc::dimension_type>(conv)));
    auto const& triple = vtriples.back();
    auto& [image_mask, filter_mask, correlated] = triple;

    for (int i = 0; i < image_size; ++i)
    {
        shares.push_back(registers[conv.image_address + i] - image_mask[i]);
    }

    for (int i = 0; i < filter_size; ++i)
    {
        shares.push_back(registers[conv.filter_address + i] - filter_mask[i]);
    }
}

template<class T>
void Beaver<T>::finalize_conv2d(std::span<T> registers, convolution_desc conv)
{
    auto conv_height = conv.full_output_height();
    auto conv_width = conv.full_output_width();
    auto offset_y = conv.filter_height - 1;
    auto offset_x = conv.filter_width - 1;

    auto image_size = conv.image_size();
    auto filter_size = conv.filter_size();

    auto masked_image = it;
    auto masked_filter = it + image_size;

    auto& [triple_image_mask, triple_filter_mask, triple_convolved] = *vtriple;

    for (int b = 0; b < conv.image_batch; ++b)
    {
        for (int y = 0; y < conv.output_height; ++y)
        {
            for (int x = 0; x < conv.output_width; ++x)
            {
                for (int c = 0; c < conv.output_depth; ++c)
                {
                    int image_y = y * conv.stride_y - conv.padding_y;
                    int image_x = x * conv.stride_x - conv.padding_x;

                    T value = triple_convolved[nDaccess({b, conv.image_batch}, {offset_y + image_y, conv_height}, {offset_x + image_x, conv_width}, {c, conv.output_depth})];

                    for (int sample_y = std::max(0, image_y); sample_y < std::min(image_y + conv.filter_height, conv.image_height); ++sample_y)
                    {
                        int filter_y = sample_y - image_y;
                        
                        for (int sample_x = std::max(0, image_x); sample_x < std::min(image_x + conv.filter_width, conv.image_width); ++sample_x)
                        {
                            int filter_x = sample_x - image_x;

                            for (int dc = 0; dc < conv.image_depth; ++dc)
                            {
                                auto filter_index = nDaccess({c, conv.output_depth}, {filter_y, conv.filter_height}, {filter_x, conv.filter_width}, {dc, conv.image_depth});
                                auto image_index = nDaccess({b, conv.image_batch}, {sample_y, conv.image_height}, {sample_x, conv.image_width}, {dc, conv.image_depth});
                                
                                value += triple_image_mask[image_index] * masked_filter[filter_index];
                                value += triple_filter_mask[filter_index] * masked_image[image_index];
                                value += T::constant(masked_image[image_index] * masked_filter[filter_index], P.my_num(), MC->get_alphai());
                            }
                        }
                    }
                    registers[conv.output_address + nDaccess({b, conv.image_batch}, {y, conv.output_height}, {x, conv.output_width}, {c, conv.output_depth})] = value;
                }
            }
        }
    }
    
    it += image_size + filter_size;
    ++vtriple;
}

template<class T>
void Beaver<T>::finalize_conv2d(std::span<T> registers, depthwise_convolution_desc conv)
{
    auto conv_height = conv.full_output_height();
    auto conv_width = conv.full_output_width();
    auto offset_y = conv.filter_height - 1;
    auto offset_x = conv.filter_width - 1;

    auto image_size = conv.image_size();
    auto filter_size = conv.filter_size();

    auto masked_image = it;
    auto masked_filter = it + image_size;

    auto& [triple_image_mask, triple_filter_mask, triple_convolved] = *vtriple;

    for (int b = 0; b < conv.image_batch; ++b)
    {
        for (int y = 0; y < conv.output_height; ++y)
        {
            for (int x = 0; x < conv.output_width; ++x)
            {
                for (int c = 0; c < conv.image_depth; ++c)
                {
                    int image_y = y * conv.stride_y - conv.padding_y;
                    int image_x = x * conv.stride_x - conv.padding_x;

                    T value = triple_convolved[nDaccess({b, conv.image_batch}, {offset_y + image_y, conv_height}, {offset_x + image_x, conv_width}, {c, conv.image_depth})];

                    for (int sample_y = std::max(0, image_y); sample_y < std::min(image_y + conv.filter_height, conv.image_height); ++sample_y)
                    {
                        int filter_y = sample_y - image_y;
                        
                        for (int sample_x = std::max(0, image_x); sample_x < std::min(image_x + conv.filter_width, conv.image_width); ++sample_x)
                        {
                            int filter_x = sample_x - image_x;

                            auto filter_index = nDaccess({filter_y, conv.filter_height}, {filter_x, conv.filter_width}, {c, conv.image_depth});
                            auto image_index = nDaccess({b, conv.image_batch}, {sample_y, conv.image_height}, {sample_x, conv.image_width}, {c, conv.image_depth});

                            value += triple_image_mask[image_index] * masked_filter[filter_index];
                            value += triple_filter_mask[filter_index] * masked_image[image_index];
                            value += T::constant(masked_image[image_index] * masked_filter[filter_index], P.my_num(), MC->get_alphai());
                        }
                    }
                    registers[conv.output_address + nDaccess({b, conv.image_batch}, {y, conv.output_height}, {x, conv.output_width}, {c, conv.image_depth})] = value;
                }
            }
        }
    }

    it += image_size + filter_size;
    ++vtriple;
}

template<class T>
void Beaver<T>::check()
{
    assert(MC);
    MC->Check(P);
}

#endif

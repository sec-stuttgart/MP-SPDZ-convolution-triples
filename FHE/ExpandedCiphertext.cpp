#include "FHE/ExpandedCiphertext.h"
#include "FHE/Ciphertext.h"
#include "Tools/conv2d.h"
#include "Math/modp.hpp"
#include "FHEOffline/SimpleMachine.h"
#include "FHEOffline/Proof.h"

modp Circulant_Matrix::get_element(int i, int j) const
{
    CONV2D_ASSERT(element.get_rep() == polynomial);
    int N = element.get_FFTD().num_slots();
    int k = i - j;
    if (k < 0)
    {
        auto x = element.get_element(k + N);
        Negate(x, x, element.get_prD());
        return x;
    }
    else
    {
        return element.get_element(k);
    }
}

Ring_Element Circulant_Matrix::get_row(int i) const
{
    CONV2D_ASSERT(element.get_rep() == polynomial);
    int N = element.get_FFTD().num_slots();
    auto const& p = element.get_prD();
    auto row = Ring_Element(element.get_FFTD(), polynomial);
    for (int j = 0; j < N; ++j)
    {
        int k = i - j;
        if (k < 0)
        {
            auto x = element.get_element(k + N);
            Negate(x, x, p);
            row.set_element(j, x);
        }
        else
        {
            row.set_element(j, element.get_element(k));
        }
    }
    return row;
}

Ring_Element Circulant_Matrix::get_column(int j) const
{
    CONV2D_ASSERT(element.get_rep() == polynomial);
    int N = element.get_FFTD().num_slots();
    auto const& p = element.get_prD();
    auto column = Ring_Element(element.get_FFTD(), polynomial);
    for (int i = 0; i < N; ++i)
    {
        int k = i - j;
        if (k < 0)
        {
            auto x = element.get_element(k + N);
            Negate(x, x, p);
            column.set_element(i, x);
        }
        else
        {
            column.set_element(i, element.get_element(k));
        }
    }
    return column;
}

Ring_Element Circulant_Matrix::operator*(Ring_Element const& v) const
{
    if (v.get_rep() != evaluation)
    {
        throw rep_mismatch();
    }
    
    Ring_Element result = element;
    result.change_rep(evaluation);
    result *= v;
    return result;
}

void Circulant_Matrix::transpose()
{
    element.change_rep(polynomial);
    Ring_Element transposed = element;
    int N = element.get_FFTD().num_slots();
    element.negate();
    for (int i = 1; i < N; ++i) // starts from 1; first element is unchanged
    {
        transposed.set_element(i, element.get_element(N - i));
    }
    element = std::move(transposed);
}

Circulant_Matrix::operator Ring_Matrix() const
{
    CONV2D_ASSERT(element.get_rep() == polynomial);
    int N = element.get_FFTD().num_slots();
    auto const& p = element.get_prD();
    auto result = Ring_Matrix(element.get_FFTD());
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            int k = i - j;
            if (k < 0)
            {
                auto x = element.get_element(k + N);
                Negate(x, x, p);
                result.set_element(i, j, x);
            }
            else
            {
                result.set_element(i, j, element.get_element(k));
            }
        }
    }
    return result;
}

Ring_Matrix::Ring_Matrix(FFT_Data const& fftd)
    : FFTD(&fftd), matrix(fftd.num_slots() * fftd.num_slots())
{
}

void Ring_Matrix::set_data(FFT_Data const& fftd)
{
    FFTD = &fftd;
    allocate();
}

FFT_Data const& Ring_Matrix::get_FFTD() const
{
    return *FFTD;
}

void Ring_Matrix::assign_zero()
{
    allocate();
    std::generate(begin(matrix), end(matrix), [](){ return modp{}; });
}

void Ring_Matrix::allocate()
{
    int N = FFTD->num_slots();
    assert(matrix.size() == static_cast<std::size_t>(N * N));
}

modp Ring_Matrix::get_element(int i, int j) const
{
    int N = FFTD->num_slots();
    auto index = nDaccess({i, N}, {j, N});
    return matrix[index];
}

void Ring_Matrix::set_element(int i, int j, modp const& a)
{
    int N = FFTD->num_slots();
    auto index = nDaccess({i, N}, {j, N});
    matrix[index] = a;
}

Ring_Element Ring_Matrix::get_row(int i) const
{
    int N = FFTD->num_slots();
    auto row = Ring_Element(*FFTD, rep);
    for (int j = 0; j < N; ++j)
    {
        auto index = nDaccess({i, N}, {j, N});
        row.set_element(j, matrix[index]);
    }
    return row;
}
    
void Ring_Matrix::set_row(int i, Ring_Element const& row)
{
    if (row.get_FFTD() != *FFTD)
    {
        throw pr_mismatch();
    }
    if (row.get_rep() != rep)
    {
        throw rep_mismatch();
    }
    
    int N = FFTD->num_slots();
    for (int j = 0; j < N; ++j)
    {
        auto index = nDaccess({i, N}, {j, N});
        matrix[index] = row.get_element(j);
    }
}

Ring_Element Ring_Matrix::get_column(int j) const
{
    int N = FFTD->num_slots();
    auto column = Ring_Element(*FFTD, rep);
    for (int i = 0; i < N; ++i)
    {
        auto index = nDaccess({i, N}, {j, N});
        column.set_element(i, matrix[index]);
    }
    return column;
}

void Ring_Matrix::set_column(int j, Ring_Element const& column)
{
    if (column.get_FFTD() != *FFTD)
    {
        throw pr_mismatch();
    }
    if (column.get_rep() != rep)
    {
        throw rep_mismatch();
    }

    int N = FFTD->num_slots();
    for (int i = 0; i < N; ++i)
    {
        auto index = nDaccess({i, N}, {j, N});
        matrix[index] = column.get_element(i);
    }
}

Ring_Matrix& Ring_Matrix::operator+=(Ring_Matrix const& other)
{
    if (*FFTD != *other.FFTD)
    {
        throw pr_mismatch();
    }

    auto const& p = FFTD->get_prD();
    CONV2D_ASSERT(matrix.size() == other.matrix.size());
    for (std::size_t i = 0; i < matrix.size(); ++i)
    {
        Add(matrix[i], matrix[i], other.matrix[i], p);
    }
    return *this;
}

Ring_Element Ring_Matrix::operator*(Ring_Element const& v) const
{
    if (v.get_FFTD() != *FFTD)
    {
        throw pr_mismatch();
    }
    if (v.get_rep() != rep)
    {
        throw rep_mismatch();
    }
    
    int N = FFTD->num_slots();
    auto const& p = FFTD->get_prD();
    auto result = Ring_Element(*FFTD, rep);
    for (int i = 0; i < N; ++i)
    {
        auto y = modp{};
        for (int j = 0; j < N; ++j)
        {
            auto index = nDaccess({i, N}, {j, N});
            y = y.add(matrix[index].mul(v.get_element(j), p), p);
        }
        result.set_element(i, y);
    }
    return result;
}

Ring_Matrix Ring_Matrix::operator*(Circulant_Matrix const& circ) const
{
    auto transposed_circ = circ;
    transposed_circ.transpose();
    transposed_circ.change_rep(evaluation);

    auto result = Ring_Matrix(*FFTD);
    int N = FFTD->num_slots();

    for (int i = 0; i < N; ++i)
    {
        auto row = get_row(i);
        row.change_rep(evaluation);

        row = transposed_circ * row;
        row.change_rep(polynomial);

        result.set_row(i, row);
    }
    return result;
}

Ring_Matrix Ring_Matrix::operator*(modp const& x) const
{
    int N = FFTD->num_slots();
    auto const& p = FFTD->get_prD();
    auto result = Ring_Matrix(*FFTD);
    for (int i = 0; i < N; ++i)
    {
        Mul(result.matrix[i], matrix[i], x, p);
    }
    return result;
}

Ring_Matrix Ring_Matrix::operator*(bigint const& x) const
{
    auto const& p = FFTD->get_prD();
    modp y;
    to_modp(y, x, p);
    return (*this) * y;
}

void Ring_Matrix::randomize(PRNG& G)
{
    auto const& p = FFTD->get_prD();

    modp options[4];
    assignOne(options[3], p);
    assignZero(options[2], p);
    options[1] = options[2];
    Negate(options[0], options[3], p);

    for (auto& element : matrix)
    {
        auto index = G.get_uint(4);
        CONV2D_ASSERT(index < 4);
        element = options[index];
    }
}

void Ring_Matrix::pack(octetStream& o) const
{
    store(o, matrix, FFTD->get_prD());
}

void Ring_Matrix::unpack(octetStream& o)
{
    get(o, matrix, FFTD->get_prD());
    
    int N = FFTD->num_slots();
    assert(matrix.size() == static_cast<std::size_t>(N * N));
}

void Ring_Matrix::output(ostream& s) const
{
    auto size = matrix.size();
    s.write((char*)&size, sizeof(size));
    for (auto& x : matrix)
        x.output(s, FFTD->get_prD(), false);
}

void Ring_Matrix::input(istream& s)
{
    auto size = matrix.size();
    s.read((char*)&size, sizeof(size));
    matrix.resize(size);
    for (auto& x : matrix)
        x.input(s, FFTD->get_prD(), false);

    int N = FFTD->num_slots();
    assert(matrix.size() == static_cast<std::size_t>(N * N));
}


size_t Ring_Matrix::report_size(ReportType type) const
{
    if (type == CAPACITY)
        return sizeof(modp) * matrix.capacity();
    else
        return sizeof(mp_limb_t) * FFTD->get_prD().get_t() * matrix.size();
}

void MultiConvolution_Matrix::mul(ExpandedCiphertext& result, Ciphertext const& a) const
{
    CONV2D_ASSERT(a.c0().n_mults() == 0);
    CONV2D_ASSERT(a.c1().n_mults() == 0);
    auto c0 = a.c0().get(0);
    c0.change_rep(polynomial);
    auto c1 = a.c1().get(0);
    c1.change_rep(polynomial);
    this->mul(result.get_reference_to_c1(), Circulant_Matrix(c1)); 
    result.get_reference_to_c0() = (*this) * c0;
}

MultiConvolution_Matrix::MultiConvolution_Matrix(FFT_Data const& fftd, depthwise_convolution_triple_dimensions dimensions)
    : FFTD(&fftd), dimensions(dimensions)
{
    int N = FFTD->num_slots();
    auto [batches_per_convolution, batches_required, convolutions] = dimensions.depthwise_split(N);
    elements.resize(convolutions * dimensions.filter_size());
}

Ring_Element MultiConvolution_Matrix::operator*(Ring_Element const& v) const
{
    if (v.get_FFTD() != *FFTD)
    {
        throw pr_mismatch();
    }
    if (v.get_rep() != polynomial)
    {
        throw rep_mismatch();
    }

    int N = FFTD->num_slots();
    auto const& p = FFTD->get_prD();
    auto [batches_per_convolution, batches_required, convolutions] = dimensions.depthwise_split(N);
    int H = dimensions.full_output_height();
    int W = dimensions.full_output_width();

    auto result = Ring_Element(*FFTD, polynomial);

    for (int c = 0; c < convolutions; ++c)
    {
        for (int b = 0; b < batches_per_convolution; ++b)
        {
            for (int y = 0; y < H; ++y)
            {
                for (int x = 0; x < W; ++x)
                {
                    int image_y = y - dimensions.filter_height + 1;
                    int image_x = x - dimensions.filter_width + 1;

                    int output_index = nDaccess({c, convolutions}, {b, batches_per_convolution}, {y, H}, {x, W});
                    CONV2D_ASSERT(output_index >= 0);
                    CONV2D_ASSERT(output_index < N);

                    auto sum = modp{};

                    for (int sample_y = std::max(0, image_y); sample_y < std::min(image_y + dimensions.filter_height, dimensions.image_height); ++sample_y)
                    {
                        int filter_y = sample_y - image_y;
                        
                        for (int sample_x = std::max(0, image_x); sample_x < std::min(image_x + dimensions.filter_width, dimensions.image_width); ++sample_x)
                        {
                            int filter_x = sample_x - image_x;
                        
                            int filter_index = nDaccess({c, convolutions}, {filter_y, dimensions.filter_height}, {filter_x, dimensions.filter_width});
                            int image_index = nDaccess({c, convolutions}, {b, batches_per_convolution}, {sample_y, H}, {sample_x, W});
                            
                            CONV2D_ASSERT(filter_index >= 0);
                            CONV2D_ASSERT(static_cast<std::size_t>(filter_index) < elements.size());
                            CONV2D_ASSERT(image_index >= 0);
                            CONV2D_ASSERT(image_index < N);

                            Add(sum, sum, 
                                elements[filter_index].mul(v.get_element(image_index), p),
                                p);
                        }
                    }
                    result.set_element(output_index, sum);
                }
            }
        }
    }
    return result;
}

void MultiConvolution_Matrix::mul(Ring_Matrix& result, Circulant_Matrix const& circ) const
{
    int N = FFTD->num_slots();
    for (int j = 0; j < N; ++j)
    {
        result.set_column(j, (*this) * circ.get_column(j));
    }
}

MultiConvolution_Matrix::operator Ring_Matrix() const
{
    int N = FFTD->num_slots();
    auto [batches_per_convolution, batches_required, convolutions] = dimensions.depthwise_split(N);
    int H = dimensions.full_output_height();
    int W = dimensions.full_output_width();

    auto result = Ring_Matrix(*FFTD);

    for (int c = 0; c < convolutions; ++c)
    {
        for (int b = 0; b < batches_per_convolution; ++b)
        {
            for (int y = 0; y < H; ++y)
            {
                for (int x = 0; x < W; ++x)
                {
                    int image_y = y - dimensions.filter_height + 1;
                    int image_x = x - dimensions.filter_width + 1;

                    int output_index = nDaccess({c, convolutions}, {b, batches_per_convolution}, {y, H}, {x, W});
                    CONV2D_ASSERT(output_index >= 0);
                    CONV2D_ASSERT(output_index < N);

                    for (int sample_y = std::max(0, image_y); sample_y < std::min(image_y + dimensions.filter_height, dimensions.image_height); ++sample_y)
                    {
                        int filter_y = sample_y - image_y;
                        
                        for (int sample_x = std::max(0, image_x); sample_x < std::min(image_x + dimensions.filter_width, dimensions.image_width); ++sample_x)
                        {
                            int filter_x = sample_x - image_x;
                        
                            int filter_index = nDaccess({c, convolutions}, {filter_y, dimensions.filter_height}, {filter_x, dimensions.filter_width});
                            int image_index = nDaccess({c, convolutions}, {b, batches_per_convolution}, {sample_y, H}, {sample_x, W});
                            
                            CONV2D_ASSERT(filter_index >= 0);
                            CONV2D_ASSERT(static_cast<std::size_t>(filter_index) < elements.size());
                            CONV2D_ASSERT(image_index >= 0);
                            CONV2D_ASSERT(image_index < N);

                            result.set_element(image_index, output_index, elements[filter_index]);
                        }
                    }
                }
            }
        }
    }
    return result;
}

void ExpandedCiphertext::rerandomize(FHE_PK const& pk)
{
#ifdef CONV2D_LOWGEAR_NO_EXPANDED_MASK
    Ciphertext mask(pk);

    mask.rerandomize(pk);

    *this += mask;
#else
    CONV2D_ASSERT(cc0.get_rep() == polynomial);

    auto N = pk.get_params().phi_m();
    auto const& p = cc0.get_FFTD().get_prD();
    
    for (int i = 0; i < N; ++i)
    {
        Ciphertext mask(pk);
        mask.rerandomize(pk);

        CONV2D_ASSERT(mask.c0().n_mults() == 0);
        CONV2D_ASSERT(mask.c1().n_mults() == 0);

        auto mask_c0 = mask.c0().get(0);
        mask_c0.change_rep(polynomial);

        auto mask_c1 = mask.c1().get(0);
        mask_c1.change_rep(polynomial);

        cc0.set_element(i, cc0.get_element(i).add(mask_c0.get_element(i), p));

        cc1.set_row(i, cc1.get_row(i) += static_cast<Circulant_Matrix>(mask_c1).get_row(i));
    }
#endif
}

ExpandedCiphertext expand(Ciphertext const& ciphertext)
{
    ExpandedCiphertext result(ciphertext.get_params());

    CONV2D_ASSERT(ciphertext.c0().n_mults() == 0);
    CONV2D_ASSERT(ciphertext.c1().n_mults() == 0);

    result.set(ciphertext.c0().get(0), static_cast<Ring_Matrix>(static_cast<Circulant_Matrix>(ciphertext.c1().get(1))));

    return result;
}

ExpandedCiphertext& ExpandedCiphertext::operator+=(Ciphertext const& other)
{
    CONV2D_ASSERT(other.c0().n_mults() == 0);
    CONV2D_ASSERT(other.c1().n_mults() == 0);

    auto other_c0 = other.c0().get(0);
    other_c0.change_rep(polynomial);

    auto other_c1 = other.c1().get(0);
    other_c1.change_rep(polynomial);

    cc0 += other_c0;
    auto other_cc1 = static_cast<Circulant_Matrix>(other_c1);
    
    auto N = params->phi_m();
    auto const& p = cc1.get_FFTD().get_prD();

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            cc1.set_element(i, j, cc1.get_element(i, j).add(other_cc1.get_element(i, j), p));
        }
    }

    return *this;
}

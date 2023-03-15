#pragma once

#include "FHE/FHE_Params.h"
#include "FHE/Ring_Element.h"
#include "Tools/conv2d.h"
#include "FHE/Plaintext.h"
#include "FHE/FFT_Data.h"
#include "FHE/P2Data.h"

class FHE_PK;
class Ciphertext;
class ExpandedCiphertext;
class MachineBase;
class Ring_Matrix;

class Circulant_Matrix
{
    Ring_Element element;

public:
    Circulant_Matrix(Ring_Element const& element)
        : element(element)
    {
    }

    modp get_element(int i, int j) const;
    Ring_Element get_row(int i) const;
    Ring_Element get_column(int j) const;

    Ring_Element operator*(Ring_Element const& v) const;

    void transpose();

    void change_rep(RepType rep)
    {
        element.change_rep(rep);
    }

    explicit operator Ring_Matrix() const;
};

class Ring_Matrix
{
    static RepType const rep = polynomial;

    FFT_Data const* FFTD;

    std::vector<modp> matrix;

public:
    Ring_Matrix(FFT_Data const& fftd);

    void set_data(FFT_Data const& fftd);

    FFT_Data const& get_FFTD() const;

    void assign_zero();
    void allocate();

    modp get_element(int i, int j) const;
    void set_element(int i, int j, modp const& a);

    Ring_Element get_row(int i) const;
    void set_row(int i, Ring_Element const& row);

    Ring_Element get_column(int j) const;
    void set_column(int j, Ring_Element const& column);


    Ring_Matrix& operator+=(Ring_Matrix const& other);
    Ring_Element operator*(Ring_Element const& v) const;
    Ring_Matrix operator*(Circulant_Matrix const& circ) const;
    Ring_Matrix operator*(modp const& x) const;
    Ring_Matrix operator*(bigint const& x) const;

    void randomize(PRNG& G);

    void pack(octetStream& o) const;
    void unpack(octetStream& o);
    void unpack(octetStream& o, FFT_Data);

    void output(ostream& s) const;
    void input(istream& s);

    size_t report_size(ReportType type) const;
};

class MultiConvolution_Matrix
{
    FFT_Data const* FFTD;
    std::vector<modp> elements;
    depthwise_convolution_triple_dimensions dimensions;

public:
    template<typename T, typename FD, typename S> friend class MultiConvolution_PlaintextMatrix;


    MultiConvolution_Matrix(FFT_Data const& fftd, depthwise_convolution_triple_dimensions dimensions);

    std::size_t size() const { return elements.size(); }

    template<typename T>
    void from(Generator<T> const& generator);

    Ring_Element operator*(Ring_Element const& v) const;
    void mul(Ring_Matrix& result, Circulant_Matrix const& circ) const;
    void mul(ExpandedCiphertext& result, Ciphertext const& a) const;

    explicit operator Ring_Matrix() const;
};

template<typename T, typename FD, typename S>
class MultiConvolution_PlaintextMatrix
{
    using poly_type = typename Plaintext<T, FD, S>::poly_type;

    FD const* FieldD;
    std::vector<poly_type> elements;
    depthwise_convolution_triple_dimensions dimensions;

    using sum_type = std::conditional_t<std::is_same_v<FD, FFT_Data>, bigint, poly_type>;

    auto fma(sum_type&, poly_type const&, poly_type const&) const -> void;
    auto from_sum(sum_type const&) const -> poly_type;

public:
    MultiConvolution_PlaintextMatrix(FD const& fieldD, depthwise_convolution_triple_dimensions dimensions)
        : FieldD(&fieldD), dimensions(dimensions)
    {
        int N = FieldD->num_slots();
        auto [batches_per_convolution, batches_required, convolutions] = dimensions.depthwise_split(N);
        elements.resize(convolutions * dimensions.filter_size());
    }

    std::size_t size() const { return elements.size(); }

    Iterator<poly_type> get_iterator() const { return elements; }

    void randomize(PRNG& G)
    {
        for (auto& element : elements)
        {
            element.randomBnd(G, FieldD->get_prime(), true);
        }
    }
    
    void randomize(MultiConvolution_Matrix& matrix, PRNG& G)
    {
        CONV2D_ASSERT(matrix.size() == elements.size());

        for (std::size_t i = 0; i < elements.size(); ++i)
        {
            elements[i].randomBnd(G, FieldD->get_prime(), true);

            auto copy = elements[i];
            
            modp tmp;
            tmp.convert_destroy(copy, matrix.FFTD->get_prD());
            matrix.elements[i] = tmp;
        }
    }

    explicit operator Plaintext<T, FD, S>() const
    {
        auto result = Plaintext<T, FD, S>(*FieldD);
        auto coefficients = elements;
        coefficients.resize(FieldD->num_slots());
        result.set_poly(coefficients);
        return result;
    }

    poly_type const& coeff(int i) const
    {
        CONV2D_ASSERT(i >= 0);
        CONV2D_ASSERT(static_cast<std::size_t>(i) < elements.size());
        return elements[i];
    }
    void set_coeff(int i, poly_type const& element)
    {
        CONV2D_ASSERT(i >= 0);
        CONV2D_ASSERT(static_cast<std::size_t>(i) < elements.size());
        elements[i] = element;
    }

    poly_type const& coeff(int c, int y, int x) const
    {
        int N = FieldD->num_slots();
        auto [batches_per_convolution, batches_required, convolutions] = dimensions.depthwise_split(N);
        int filter_index = nDaccess({c, convolutions}, {y, dimensions.filter_height}, {x, dimensions.filter_width});
        CONV2D_ASSERT(filter_index >= 0);
        CONV2D_ASSERT(static_cast<std::size_t>(filter_index) < elements.size());
        return elements[filter_index];
    }

    Plaintext<T, FD, S> operator*(Plaintext<T, FD, S> const& m) const
    {
        if (m.get_field() != *FieldD)
        {
            throw field_mismatch();
        }

        int N = FieldD->num_slots();
        auto [batches_per_convolution, batches_required, convolutions] = dimensions.depthwise_split(N);
        int H = dimensions.full_output_height();
        int W = dimensions.full_output_width();

        auto result = Plaintext<T, FD, S>(*FieldD);

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

                        auto sum = sum_type{};

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

                                fma(sum, m.coeff(image_index), elements[filter_index]);
                            }
                        }
                        result.set_coeff(output_index, from_sum(sum));
                    }
                }
            }
        }
        return result;
    }
};

template<>
inline auto MultiConvolution_PlaintextMatrix<gfp, FFT_Data, bigint>::fma(sum_type& sum, poly_type const& image, poly_type const& filter) const -> void
{
    auto const& p = FieldD->get_prime();
    sum_type i = image;
    sum_type f = filter;
    sum += i * f;
    sum %= p;
}

template<>
inline auto MultiConvolution_PlaintextMatrix<gfp, FFT_Data, bigint>::from_sum(sum_type const& sum) const -> poly_type
{
    return sum;
}

template<>
inline auto MultiConvolution_PlaintextMatrix<gf2n_short, P2Data, int>::fma(sum_type& sum, poly_type const& image, poly_type const& filter) const -> void
{
    sum_type product = image * filter;
    sum += product;
}

template<>
inline auto MultiConvolution_PlaintextMatrix<gf2n_short, P2Data, int>::from_sum(sum_type const& sum) const -> poly_type
{
    return sum;
}

template<typename FD>
using MultiConvolution_PlaintextMatrix_ = MultiConvolution_PlaintextMatrix<typename FD::T, FD, typename FD::S>;

template<typename T>
void MultiConvolution_Matrix::from(Generator<T> const& generator)
{
    T tmp;
    modp tmp2;
    
    for (std::size_t i = 0; i < elements.size(); ++i)
    {
        generator.get(tmp);
        tmp2.convert_destroy(tmp, FFTD->get_prD());
        elements[i] = tmp2;
    }
}

class ExpandedCiphertext
{
    Ring_Element cc0;
    Ring_Matrix cc1;
    FHE_Params const* params;
    // identifier for debugging
    word pk_id;

public:
    static int size() { return 0; }

    FHE_Params const& get_params() const { return *params; }
    FFT_Data const& get_FFTD() const { return params->FFTD()[0]; }

    ExpandedCiphertext(const FHE_Params& p)
        : cc0(p.FFTD()[0])
        , cc1(p.FFTD()[0])
        , params(&p)
        , pk_id(0)
    {
        assert(p.FFTD().size() == 1);
    }

    void set(Ring_Element const& c0, Ring_Matrix const& c1)
    {
        CONV2D_ASSERT(params);
        if (c0.get_FFTD() != get_FFTD())
        {
            throw field_mismatch();
        }
        if (c1.get_FFTD() != get_FFTD())
        {
            throw field_mismatch();
        }
        cc0 = c0;
        cc1 = c1;
    }
    
    word get_pk_id() const { return pk_id; }

    const Ring_Element& c0() const { return cc0; }
    const Ring_Matrix& c1() const { return cc1; }

    Ring_Element& get_reference_to_c0() { return cc0; }
    Ring_Matrix& get_reference_to_c1() { return cc1; }
    
    void assign_zero() { cc0.assign_zero(); cc1.assign_zero(); pk_id = 0; }
    void allocate() { cc0.allocate(); cc1.allocate(); }

    void rerandomize(const FHE_PK& pk);

    template<typename T, typename FD, typename S>
    ExpandedCiphertext& operator+=(Plaintext<T, FD, S> const& m)
    {
        auto r = Ring_Element(cc0.get_FFTD());
        r.from(m.get_iterator());
        return (*this) += r;
    }

    ExpandedCiphertext& operator+=(Ring_Element const& m)
    {
        cc0 += m;
        return *this;
    }

    ExpandedCiphertext& operator+=(Ciphertext const& other);

    ExpandedCiphertext& operator+=(ExpandedCiphertext const& other)
    {
        cc0 += other.cc0;
        cc1 += other.cc1;
        return *this;
    }

    void pack(octetStream& o) const
    { 
        cc0.pack(o); 
        cc1.pack(o);
        o.store(pk_id); 
    }

    void unpack(octetStream& o)
    {
        cc0.set_data(params->FFTD()[0]); 
        cc0.unpack(o);
        cc1.set_data(params->FFTD()[0]);
        cc1.unpack(o); 
        o.get(pk_id);
    }

    void output(ostream& s) const
    { 
        cc0.output(s); 
        cc1.output(s); 
        s.write((char*)&pk_id, sizeof(pk_id)); 
    }
    void input(istream& s)
    { 
        cc0.input(s); 
        cc1.input(s); 
        s.read((char*)&pk_id, sizeof(pk_id)); 
    }

    size_t report_size(ReportType type) const { return cc0.report_size(type) + cc1.report_size(type); }
};

ExpandedCiphertext expand(Ciphertext const& ciphertext);

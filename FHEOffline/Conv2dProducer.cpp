#include "FHEOffline/Conv2dProducer.h"
#include "Tools/Subroutines.h"
#include "FHEOffline/PairwiseGenerator.h"
#include "FHEOffline/PairwiseMachine.h"

template<typename ConvolutionDimensions>
std::pair<int, int> get_fitting_split(ConvolutionDimensions dimensions, FHE_Params const& params)
{
  auto N = params.phi_m();

  int log_split_y = 0;
  int log_split_x = 0;

  CONV2D_ASSERT(dimensions.image_height > 0);
  CONV2D_ASSERT(dimensions.image_width > 0);
  
  int max_log_split_y = std::countr_zero(static_cast<unsigned int>(dimensions.image_height));
  int max_log_split_x = std::countr_zero(static_cast<unsigned int>(dimensions.image_width));

  int max_log_split_total = max_log_split_x + max_log_split_y;

  for (int i = 0; dimensions.full_output_area() > N and i < max_log_split_total; ++i)
  {
    if (((i & 1) == 0) and (log_split_y < max_log_split_y))
    {
      ++log_split_y;
      dimensions.image_height >>= 1;
    }
    else
    {
      CONV2D_ASSERT(log_split_x < max_log_split_x);
      ++log_split_x;
      dimensions.image_width >>= 1;
    }
  }

  if (dimensions.full_output_area() <= N)
  {
    return std::make_pair(1 << log_split_y, 1 << log_split_x);
  }
  else
  {
    // TODO: We could factor image_width and image_height in remainding (non-2) prime factors and split by that.
    // TODO: After no prime factors are left, we could still split but this is then not a perfect split of the input
    //       and needs special treatment when combining the smaller convolutions.
    throw std::runtime_error("Unable to make this convolution dimensions fit into a plaintext/ciphertext");
  }
}

template<typename T>
void adapt_after_sacrificing(depthwise_convolution_triple_dimensions dimensions, depthwise_convolution_triple_dimensions split_dimensions, int split_y, int split_x, std::vector<T>& image, std::vector<T>& output)
{
  auto split_conv_height = split_dimensions.full_output_height();
  auto split_conv_width = split_dimensions.full_output_width();

  auto converted_image = std::vector<T>(dimensions.image_size());
  auto converted_output = std::vector<T>(dimensions.full_output_size());

  auto split_height = split_dimensions.image_height;
  auto split_width = split_dimensions.image_width;

  CONV2D_ASSERT(dimensions.image_height % split_y == 0);
  CONV2D_ASSERT(dimensions.image_width % split_x == 0);
  CONV2D_ASSERT(split_height == dimensions.image_height / split_y);
  CONV2D_ASSERT(split_width == dimensions.image_width / split_x);
  CONV2D_ASSERT(split_dimensions.image_batch == dimensions.image_batch * split_x * split_y);
  CONV2D_ASSERT(dimensions.filter_height == split_dimensions.filter_height);
  CONV2D_ASSERT(dimensions.filter_width == split_dimensions.filter_width);

  for (int i = 0; i < split_y; ++i)
  {
    for (int j = 0; j < split_x; ++j)
    {
      for (int b = 0; b < dimensions.image_batch; ++b)
      {
        for (int y = 0; y < split_height; ++y)
        {
          for (int x = 0; x < split_width; ++x)
          {
            auto source_index = nDaccess({i, split_y}, {j, split_x}, {b, dimensions.image_batch}, {y, split_height}, {x, split_width});
            auto target_index = nDaccess({b, dimensions.image_batch}, {i * split_height + y, dimensions.image_height}, {j * split_width + x, dimensions.image_width});
            converted_image[target_index] = image[source_index];
          }
        }
      }
    }
  }

  // filter is the same

  auto conv_height = dimensions.full_output_height();
  auto conv_width = dimensions.full_output_width();
  auto offset_y = dimensions.filter_height - 1;
  auto offset_x = dimensions.filter_width - 1;

  for (int b = 0; b < dimensions.image_batch; ++b)
  {
    for (int y = 0; y < conv_height; ++y)
    {
      for (int x = 0; x < conv_width; ++x)
      {
        for (int i = std::max(0, (y - offset_y) / split_height); i < std::min(y / split_height + 1, split_y); ++i)
        {
          for (int j = std::max(0, (x - offset_x) / split_width); j < std::min(x / split_width + 1, split_x); ++j)
          {
            int split_output_y = y - i * split_height;
            int split_output_x = x - j * split_width;
            auto source_index = nDaccess({i, split_y}, {j, split_x}, {b, dimensions.image_batch}, {split_output_y, split_conv_height}, {split_output_x, split_conv_width});
            auto target_index = nDaccess({b, dimensions.image_batch}, {y, conv_width}, {x, conv_height});
            converted_output[target_index] += output[source_index];
          }
        }
      }
    }
  }

  image = std::move(converted_image);
  output = std::move(converted_output);
}

template<typename T>
void adapt_after_sacrificing(convolution_dimensions dimensions, convolution_dimensions split_dimensions, int split_y, int split_x, std::vector<T>& image, std::vector<T>& output)
{
  auto split_conv_height = split_dimensions.full_output_height();
  auto split_conv_width = split_dimensions.full_output_width();

  auto converted_image = std::vector<T>(dimensions.image_size());
  auto converted_output = std::vector<T>(dimensions.full_output_size());

  auto split_height = split_dimensions.image_height;
  auto split_width = split_dimensions.image_width;

  CONV2D_ASSERT(dimensions.image_height % split_y == 0);
  CONV2D_ASSERT(dimensions.image_width % split_x == 0);
  CONV2D_ASSERT(split_height == dimensions.image_height / split_y);
  CONV2D_ASSERT(split_width == dimensions.image_width / split_x);
  CONV2D_ASSERT(split_dimensions.image_batch == dimensions.image_batch * split_x * split_y);
  CONV2D_ASSERT(dimensions.filter_height == split_dimensions.filter_height);
  CONV2D_ASSERT(dimensions.filter_width == split_dimensions.filter_width);
  CONV2D_ASSERT(dimensions.image_depth == split_dimensions.image_depth);
  CONV2D_ASSERT(dimensions.output_depth == split_dimensions.output_depth);

  for (int i = 0; i < split_y; ++i)
  {
    for (int j = 0; j < split_x; ++j)
    {
      for (int b = 0; b < dimensions.image_batch; ++b)
      {
        for (int y = 0; y < split_height; ++y)
        {
          for (int x = 0; x < split_width; ++x)
          {
            for (int c = 0; c < dimensions.image_depth; ++c)
            {
              auto source_index = nDaccess({i, split_y}, {j, split_x}, {b, dimensions.image_batch}, {y, split_height}, {x, split_width}, {c, dimensions.image_depth});
              auto target_index = nDaccess({b, dimensions.image_batch}, {i * split_height + y, dimensions.image_height}, {j * split_width + x, dimensions.image_width}, {c, dimensions.image_depth});
              converted_image[target_index] = image[source_index];
            }
          }
        }
      }
    }
  }

  // filter is the same

  auto conv_height = dimensions.full_output_height();
  auto conv_width = dimensions.full_output_width();
  auto offset_y = dimensions.filter_height - 1;
  auto offset_x = dimensions.filter_width - 1;

  for (int b = 0; b < dimensions.image_batch; ++b)
  {
    for (int y = 0; y < conv_height; ++y)
    {
      for (int x = 0; x < conv_width; ++x)
      {
        for (int c = 0; c < dimensions.output_depth; ++c)
        {
          for (int i = std::max(0, (y - offset_y) / split_height); i < std::min(y / split_height + 1, split_y); ++i)
          {
            for (int j = std::max(0, (x - offset_x) / split_width); j < std::min(x / split_width + 1, split_x); ++j)
            {
              int split_output_y = y - i * split_height;
              int split_output_x = x - j * split_width;
              auto source_index = nDaccess({i, split_y}, {j, split_x}, {b, dimensions.image_batch}, {split_output_y, split_conv_height}, {split_output_x, split_conv_width}, {c, dimensions.output_depth});
              auto target_index = nDaccess({b, dimensions.image_batch}, {y, conv_width}, {x, conv_height}, {c, dimensions.output_depth});
              converted_output[target_index] += output[source_index];
            }
          }
        }
      }
    }
  }

  image = std::move(converted_image);
  output = std::move(converted_output);
}

template<class FD>
template<typename ConvolutionDimensions>
void BaseConv2dTripleProducer<FD>::clear_and_set_dimensions(ConvolutionDimensions dimensions)
{
  this->dimensions = static_cast<convolution_dimensions>(dimensions);
  auto image_size = dimensions.image_size();
  auto filter_size = dimensions.filter_size();
  auto output_size = dimensions.full_output_size();
  int triple_count = this->adapt_for_dimensions(dimensions);
  this->clear();

  if constexpr (ConvolutionDimensions::is_always_depthwise)
  {
    this->presacrificing_shares = std::vector(triple_count, std::array{ std::vector<T>(image_size), std::vector<T>(image_size), std::vector<T>(filter_size), std::vector<T>(output_size), std::vector<T>(output_size) });
    this->presacrificing_macs = std::vector(triple_count, std::array{ std::vector<T>(image_size), std::vector<T>(image_size), std::vector<T>(filter_size), std::vector<T>(output_size), std::vector<T>(output_size) });
  }
  else
  {
    this->presacrificing_shares = std::vector(triple_count, std::array{ std::vector<T>(image_size), std::vector<T>(filter_size), std::vector<T>(filter_size), std::vector<T>(output_size), std::vector<T>(output_size) });
    this->presacrificing_macs = std::vector(triple_count, std::array{ std::vector<T>(image_size), std::vector<T>(filter_size), std::vector<T>(filter_size), std::vector<T>(output_size), std::vector<T>(output_size) });
  }
}

template<class FD>
template<typename ConvolutionDimensions>
void BaseConv2dTripleProducer<FD>::produce(ConvolutionDimensions dimensions, const Player& P, MAC_Check<T>& MC, const FHE_PK& pk, const Ciphertext& calpha, EncCommitBase_<FD>& EC, DistDecrypt<FD>& dd, const T& alphai)
{
  auto [split_y, split_x] = get_fitting_split(dimensions, pk.get_params());
  if (split_y == 1 and split_x == 1)
  {
    this->clear_and_set_dimensions(dimensions);
    this->run(P, pk, calpha, EC, dd, alphai);
    sacrifice(dimensions, P, MC);
    MC.Check(P);
  }
  else
  {
    auto split_dimensions = dimensions.split(split_y, split_x);
#ifdef VERBOSE_CONV2D
    std::cerr << "Computing a " << dimensions.as_string() << " conv as " 
    << split_dimensions.as_string() << " conv (split = [" << split_y << ", " << split_x << "])\n";
#endif
    this->clear_and_set_dimensions(split_dimensions);
    this->run(P, pk, calpha, EC, dd, alphai);
    auto triple_count = sacrifice(split_dimensions, P, MC);
    MC.Check(P);

#ifdef VERBOSE_CONV2D
    std::cerr << "Computing conv triples from split conv triples\n";
#endif
    for (int j = 0; j < triple_count; ++j)
    {
      auto& [image, filter, output] = this->triples[j];
      adapt_after_sacrificing(dimensions, split_dimensions, split_y, split_x, image, output);
    }

    this->dimensions = static_cast<convolution_dimensions>(dimensions);
  }
}

template<class FD>
int BaseConv2dTripleProducer<FD>::sacrifice(const Player& P, MAC_Check<T>& MC)
{
  if (dimensions.is_depthwise())
  {
    return sacrifice(dimensions.as_depthwise(), P, MC);
  }
  else
  {
    return sacrifice(dimensions, P, MC);
  }
}

template<class FD>
int BaseConv2dTripleProducer<FD>::sacrifice(convolution_dimensions dimensions, const Player& P, MAC_Check<T>& MC)
{
    check_field_size<T>();

    auto combine = [](auto const& share, auto const& mac) { Share<T> result; result.set_share(share); result.set_mac(mac); return result; };

    int triple_count = this->presacrificing_shares.size();
    CONV2D_ASSERT(this->presacrificing_macs.size() == static_cast<std::size_t>(triple_count));
    std::vector<T> ss(triple_count);

    std::size_t image_size = dimensions.image_size();
    std::size_t filter_size = dimensions.filter_size();
    std::size_t output_size = dimensions.full_output_size();
    auto H = dimensions.full_output_height();
    auto W = dimensions.full_output_width();

    std::vector<Share<T>> shared_rho(filter_size * triple_count);
    std::vector<T> rho(filter_size * triple_count);

    for (int j = 0; j < triple_count; ++j)
    {
      auto& [image, filter, filter_prime, output, output_prime] = this->presacrificing_shares[j];
      auto& [image_mac, filter_mac, filter_prime_mac, output_mac, output_prime_mac] = this->presacrificing_macs[j];
      CONV2D_ASSERT(image.size() == image_size);
      CONV2D_ASSERT(image.size() == image_mac.size());
      CONV2D_ASSERT(filter.size() == filter_size);
      CONV2D_ASSERT(filter.size() == filter_mac.size());
      CONV2D_ASSERT(output.size() == output_size);
      CONV2D_ASSERT(output.size() == output_mac.size());
      CONV2D_ASSERT(filter.size() == filter_prime.size());
      CONV2D_ASSERT(filter_mac.size() == filter_prime_mac.size());
      CONV2D_ASSERT(output.size() == output_prime.size());
      CONV2D_ASSERT(output_mac.size() == output_prime_mac.size());

      // sacrificing
      auto& s = ss[j];
      Create_Random(s, P);
      for (std::size_t i = 0; i < filter_size; ++i)
      {
        auto b = combine(filter[i], filter_mac[i]);
        auto b_prime = combine(filter_prime[i], filter_prime_mac[i]);
        shared_rho[nDaccess({j, triple_count}, {static_cast<int>(i), static_cast<int>(filter_size)})] = s * b - b_prime;
      }
    }
    MC.POpen(rho, shared_rho, P);

    std::vector<Share<T>> shared_sigma(output_size * triple_count);
    std::vector<T> sigma(output_size * triple_count);

    for (int j = 0; j < triple_count; ++j)
    {
      auto& [image, filter, filter_prime, output, output_prime] = this->presacrificing_shares[j];
      auto& [image_mac, filter_mac, filter_prime_mac, output_mac, output_prime_mac] = this->presacrificing_macs[j];
      auto& s = ss[j];
      for (int b = 0; b < dimensions.image_batch; ++b)
      {
        for (int y = 0; y < H; ++y)
        {
          for (int x = 0; x < W; ++x)
          {
            for (int d = 0; d < dimensions.output_depth; ++d)
            {
              int image_y = y - dimensions.filter_height + 1;
              int image_x = x - dimensions.filter_width + 1;

              auto output_index = nDaccess({b, dimensions.image_batch}, {y, H}, {x, W}, {d, dimensions.output_depth});
              auto& value = shared_sigma[nDaccess({j, triple_count}, {output_index, static_cast<int>(output_size)})];
              auto c = combine(output[output_index], output_mac[output_index]);
              auto c_prime = combine(output_prime[output_index], output_prime_mac[output_index]);

              value = s * c - c_prime;
              for (int sample_y = std::max(0, image_y); sample_y < std::min(image_y + dimensions.filter_height, dimensions.image_height); ++sample_y)
              {
                int filter_y = sample_y - image_y;
                
                for (int sample_x = std::max(0, image_x); sample_x < std::min(image_x + dimensions.filter_width, dimensions.image_width); ++sample_x)
                {
                  int filter_x = sample_x - image_x;

                  for (int c = 0; c < dimensions.image_depth; ++c)
                  {
                    auto image_index = nDaccess({b, dimensions.image_batch}, {sample_y, dimensions.image_height}, {sample_x, dimensions.image_width}, {c, dimensions.image_depth});
                    auto filter_index = nDaccess({d, dimensions.output_depth}, {filter_y, dimensions.filter_height}, {filter_x, dimensions.filter_width}, {c, dimensions.image_depth});
                    
                    auto a = combine(image[image_index], image_mac[image_index]);
                    value -= rho[nDaccess({j, triple_count}, {filter_index, static_cast<int>(filter_size)})] * a;
                  }
                }
              }
            }
          }
        }
      }
    }

    MC.POpen(sigma, shared_sigma, P);
    if (auto non_zero = std::find_if(begin(sigma), end(sigma), [](auto const&s) { return not s.is_zero(); }); non_zero != end(sigma))
    {
      // could use non_zero to debug where an element is non-zero
      throw Offline_Check_Error("Conv2d sacrificing failed");
    }

    this->triples.reserve(this->triples.size() + triple_count);
    for (int j = 0; j < triple_count; ++j)
    {
      auto& [image, filter, filter_prime, output, output_prime] = this->presacrificing_shares[j];
      auto& [image_mac, filter_mac, filter_prime_mac, output_mac, output_prime_mac] = this->presacrificing_macs[j];
      auto& [a, b, c] = this->triples.emplace_back();

      // copy output
      a.resize(image_size);
      std::transform(begin(image), end(image), begin(image_mac), begin(a), combine);

      b.resize(filter_size);
      std::transform(begin(filter), end(filter), begin(filter_mac), begin(b), combine);

      c.resize(output_size);
      std::transform(begin(output), end(output), begin(output_mac), begin(c), combine);
    }

    // write output
    // if (this->write_output)
    // {
    //   throw std::runtime_error("Writing to file not yet implemented");
    // }

    return triple_count;
}

template<class FD>
int BaseConv2dTripleProducer<FD>::sacrifice(depthwise_convolution_triple_dimensions dimensions, const Player& P, MAC_Check<T>& MC)
{
    check_field_size<T>();

    auto combine = [](auto const& share, auto const& mac) { Share<T> result; result.set_share(share); result.set_mac(mac); return result; };

    int triple_count = this->presacrificing_shares.size();
    CONV2D_ASSERT(this->presacrificing_macs.size() == static_cast<std::size_t>(triple_count));
    std::vector<T> ss(triple_count);

    std::size_t image_size = dimensions.image_size();
    std::size_t filter_size = dimensions.filter_size();
    std::size_t output_size = dimensions.full_output_size();
    auto H = dimensions.full_output_height();
    auto W = dimensions.full_output_width();

    // here, we produce triples (a, b, c), (a', b, c') instead of (a, b, c), (a, b', c')

    std::vector<Share<T>> shared_rho(image_size * triple_count);
    std::vector<T> rho(image_size * triple_count);

    for (int j = 0; j < triple_count; ++j)
    {
      auto& [image, image_prime, filter, output, output_prime] = this->presacrificing_shares[j];
      auto& [image_mac, image_prime_mac, filter_mac, output_mac, output_prime_mac] = this->presacrificing_macs[j];
      CONV2D_ASSERT(image.size() == image_size);
      CONV2D_ASSERT(image.size() == image_mac.size());
      CONV2D_ASSERT(filter.size() == filter_size);
      CONV2D_ASSERT(filter.size() == filter_mac.size());
      CONV2D_ASSERT(output.size() == output_size);
      CONV2D_ASSERT(output.size() == output_mac.size());
      CONV2D_ASSERT(image.size() == image_prime.size());
      CONV2D_ASSERT(image_mac.size() == image_prime_mac.size());
      CONV2D_ASSERT(output.size() == output_prime.size());
      CONV2D_ASSERT(output_mac.size() == output_prime_mac.size());

      // sacrificing
      auto& s = ss[j];
      Create_Random(s, P);
      for (std::size_t i = 0; i < image_size; ++i)
      {
        auto a = combine(image[i], image_mac[i]);
        auto a_prime = combine(image_prime[i], image_prime_mac[i]);
        shared_rho[nDaccess({j, triple_count}, {static_cast<int>(i), static_cast<int>(image_size)})] = s * a - a_prime;
      }
    }
    MC.POpen(rho, shared_rho, P);

    std::vector<Share<T>> shared_sigma(output_size * triple_count);
    std::vector<T> sigma(output_size * triple_count);

    for (int j = 0; j < triple_count; ++j)
    {
      auto& [image, image_prime, filter, output, output_prime] = this->presacrificing_shares[j];
      auto& [image_mac, image_prime_mac, filter_mac, output_mac, output_prime_mac] = this->presacrificing_macs[j];
      auto& s = ss[j];
      for (int b = 0; b < dimensions.image_batch; ++b)
      {
        for (int y = 0; y < H; ++y)
        {
          for (int x = 0; x < W; ++x)
          {
            int image_y = y - dimensions.filter_height + 1;
            int image_x = x - dimensions.filter_width + 1;

            auto output_index = nDaccess({b, dimensions.image_batch}, {y, H}, {x, W});
            auto& value = shared_sigma[nDaccess({j, triple_count}, {output_index, static_cast<int>(output_size)})];
            auto c = combine(output[output_index], output_mac[output_index]);
            auto c_prime = combine(output_prime[output_index], output_prime_mac[output_index]);

            value = s * c - c_prime;
            for (int sample_y = std::max(0, image_y); sample_y < std::min(image_y + dimensions.filter_height, dimensions.image_height); ++sample_y)
            {
              int filter_y = sample_y - image_y;
              
              for (int sample_x = std::max(0, image_x); sample_x < std::min(image_x + dimensions.filter_width, dimensions.image_width); ++sample_x)
              {
                int filter_x = sample_x - image_x;

                auto image_index = nDaccess({b, dimensions.image_batch}, {sample_y, dimensions.image_height}, {sample_x, dimensions.image_width});
                auto filter_index = nDaccess({filter_y, dimensions.filter_height}, {filter_x, dimensions.filter_width});
                
                auto b = combine(filter[filter_index], filter_mac[filter_index]);
                value -= rho[nDaccess({j, triple_count}, {image_index, static_cast<int>(image_size)})] * b;
              }
            }
          }
        }
      }
    }

    MC.POpen(sigma, shared_sigma, P);
    if (auto non_zero = std::find_if(begin(sigma), end(sigma), [](auto const&s) { return not s.is_zero(); }); non_zero != end(sigma))
    {
      // could use non_zero to debug where an element is non-zero
      throw Offline_Check_Error("Depthwise conv2d sacrificing failed");
    }

    this->triples.reserve(this->triples.size() + triple_count);
    for (int j = 0; j < triple_count; ++j)
    {
      auto& [image, image_prime, filter, output, output_prime] = this->presacrificing_shares[j];
      auto& [image_mac, image_prime_mac, filter_mac, output_mac, output_prime_mac] = this->presacrificing_macs[j];
      auto& [a, b, c] = this->triples.emplace_back();

      // copy output
      a.resize(image_size);
      std::transform(begin(image), end(image), begin(image_mac), begin(a), combine);

      b.resize(filter_size);
      std::transform(begin(filter), end(filter), begin(filter_mac), begin(b), combine);

      c.resize(output_size);
      std::transform(begin(output), end(output), begin(output_mac), begin(c), combine);
    }

    // write output
    // if (this->write_output)
    // {
    //   throw std::runtime_error("Writing to file not yet implemented");
    // }

    return triple_count;
}

#ifdef CONV2D_DIRECT_SUM
template<class FD>
void highgear_conv2d(convolution_dimensions dimensions, Player const& P, const FHE_PK& pk, const Ciphertext& calpha, DistDecrypt<FD>& dd, FD const& FieldD, map<string, Timer>& timers, SparseSummingEncCommit<FD>& imageEC, SparseSummingEncCommit<FD>& filterEC, EncCommitBase_<FD>& summingEC, std::span<typename FD::T> presacrificing_image, std::span<typename FD::T> presacrificing_filter, std::span<typename FD::T> presacrificing_filter_prime, std::span<typename FD::T> presacrificing_output, std::span<typename FD::T> presacrificing_output_prime, std::span<typename FD::T> presacrificing_image_mac, std::span<typename FD::T> presacrificing_filter_mac, std::span<typename FD::T> presacrificing_filter_prime_mac, std::span<typename FD::T> presacrificing_output_mac, std::span<typename FD::T> presacrificing_output_prime_mac)
{
    using T = typename FD::T;

    const FHE_Params& params = pk.get_params();

    CONV2D_ASSERT(params.get_plaintext_modulus() == FieldD.get_prime());
    CONV2D_ASSERT(params.phi_m() == FieldD.phi_m());
    CONV2D_ASSERT(FieldD.num_slots() == FieldD.phi_m());

    auto& outputEC = dynamic_cast<SummingEncCommit<FD>&>(summingEC);
#ifdef CONV2D_HIGHGEAR_GENERIC_EC
    auto& imageReshareEC = outputEC;
    auto& filterReshareEC = outputEC;
#else
    auto& imageReshareEC = imageEC;
    auto& filterReshareEC = filterEC;
#endif

    auto H = dimensions.full_output_height();
    auto W = dimensions.full_output_width();
    auto N = FieldD.num_slots();

    auto [batches_per_convolution, batches_required, outputs_per_convolution, outputs_required, inputs_per_convolution, inputs_required] = dimensions.direct_sum_split(N);
#ifdef VERBOSE_CONV2D
    std::cerr << "Starting summing (direct sum) convolution "
#if CONV2D_SUMMING_CIPHERTEXTS
      "(summing over ciphertexts; at most " << CONV2D_MAX_IMAGE_DEPTH << " = " << CONV2D_EXTRA_SLACK << " ?= " << OnlineOptions::singleton.matrix_dimensions << " = " << params.get_matrix_dim() << " ciphertexts) "
#endif
#ifdef CONV2D_HIGHGEAR_GENERIC_EC
      "(using the generic EC for resharing) "
#endif
      << dimensions.as_string() << "\n";
    std::cerr << "H=" << H << ", W=" << W << ", N=" << N << ", batches_per_convolution=" << batches_per_convolution << ", batches_required=" << batches_required << ", outputs_per_convolution=" << outputs_per_convolution << ", outputs_required=" << outputs_required << ", inputs_per_convolution=" << inputs_per_convolution << ", inputs_required=" << inputs_required << "\n";
#endif

    Ciphertext coutput(params), ccouput(params);
    std::vector<std::vector<Ciphertext>> cfilters, cimages;
    Plaintext_<FD> image(FieldD), filter(FieldD), output(FieldD);
    Ciphertext cmacd_image(params), cmacd_filter(params), cmacd_output(params);
    Plaintext_<FD> macd_image(FieldD), macd_filter(FieldD), macd_output(FieldD);

    timers["Filters"].start();
    cfilters.resize(outputs_required);
    for (int c = 0; c < outputs_required; ++c)
    {
      cfilters[c].resize(inputs_required, Ciphertext(params));
      for (int d = 0; d < inputs_required; ++d)
      {
        filterEC.next(filter, cfilters[c][d]);
        mul(cmacd_filter, calpha, cfilters[c][d], pk);
        dd.reshare(macd_filter, cmacd_filter, filterReshareEC);

        for (int k = 0; k < outputs_per_convolution; ++k)
        {
          for (int l = 0; l < inputs_per_convolution; ++l)
          {
            auto depth = nDaccess({c, outputs_required}, {k, outputs_per_convolution});
            auto idepth = nDaccess({d, inputs_required}, {l, inputs_per_convolution});
            for (int i = 0; i < dimensions.filter_height; ++i)
            {
              for (int j = 0; j < dimensions.filter_width; ++j)
              {
                auto source_index = nDaccess({k, outputs_per_convolution}, reverse_index(l, inputs_per_convolution), reverse_index(i, dimensions.filter_height, H), reverse_index(j, dimensions.filter_width, W)); // reverse filter to get cross-correlation instead of a convolution; input channels have to be reversed for direct sum as well
                CONV2D_ASSERT(source_index >= 0);
                CONV2D_ASSERT(source_index < N);
                if (depth < dimensions.output_depth)
                {
                  auto dest_index = nDaccess({depth, dimensions.output_depth}, {i, dimensions.filter_height}, {j, dimensions.filter_width}, {idepth, dimensions.image_depth});
                  presacrificing_filter[dest_index] = filter.coeff(source_index);
                  presacrificing_filter_mac[dest_index] = macd_filter.coeff(source_index);
                }
                else if (depth < 2 * dimensions.output_depth)
                {
                  auto dest_index = nDaccess({depth - dimensions.output_depth, dimensions.output_depth}, {i, dimensions.filter_height}, {j, dimensions.filter_width}, {idepth, dimensions.image_depth});
                  presacrificing_filter_prime[dest_index] = filter.coeff(source_index);
                  presacrificing_filter_prime_mac[dest_index] = macd_filter.coeff(source_index);
                }
              }
            }
          }
        }
      }
    }
    timers["Filters"].stop();
    
    timers["Images"].start();
    cimages.resize(batches_required);
    for (int b = 0; b < batches_required; ++b)
    {
      cimages[b].resize(inputs_required, Ciphertext(params));
      for (int c = 0; c < inputs_required; ++c)
      {
        imageEC.next(image, cimages[b][c]);
        mul(cmacd_image, calpha, cimages[b][c], pk);
        dd.reshare(macd_image, cmacd_image, imageReshareEC);

        for (int k = 0; k < batches_per_convolution; ++k)
        {
          for (int l = 0; l < inputs_per_convolution; ++l)
          {
            for (int i = 0; i < dimensions.image_height; ++i)
            {
              for (int j = 0; j < dimensions.image_width; ++j)
              {
                auto batch = nDaccess({b, batches_required}, {k, batches_per_convolution});
                auto idepth = nDaccess({c, inputs_required}, {l, inputs_per_convolution});

                if (batch < dimensions.image_batch)
                {
                  auto dest_index = nDaccess({batch, dimensions.image_batch}, {i, dimensions.image_height}, {j, dimensions.image_width}, {idepth, dimensions.image_depth});
                  auto source_index = nDaccess({k, batches_per_convolution}, first_index(outputs_per_convolution), {l, inputs_per_convolution}, {i, H}, {j, W});
                  CONV2D_ASSERT(source_index < N);
                  presacrificing_image[dest_index] = image.coeff(source_index);
                  presacrificing_image_mac[dest_index] = macd_image.coeff(source_index);
                }
              }
            }
          }
        }
      }
    }
    timers["Images"].stop();

    auto write = [](T& x, auto const& y)
    {
#if CONV2D_SUMMING_CIPHERTEXTS
      x = y;
#else
      if constexpr (requires { x += y; })
      {
        x += y;
      }
      else
      {
        x += T{y};
      }
#endif
    };

    timers["Outputs"].start();
    for (int b = 0; b < batches_required; ++b)
    {
      for (int d = 0; d < outputs_required; ++d)
      {
#if CONV2D_SUMMING_CIPHERTEXTS
        assert(inputs_required <= params.get_matrix_dim());
        Ciphertext coutput_sum(params);
        coutput_sum.allocate();
        coutput_sum.Scale();
#endif
        for (int c = 0; c < inputs_required; ++c)
        {
          mul(coutput, cimages[b][c], cfilters[d][c], pk);
#if CONV2D_SUMMING_CIPHERTEXTS
          coutput_sum += coutput;
        }
        Reshare(output, ccouput, coutput_sum, true, P, outputEC, pk, dd);
#else
          Reshare(output, ccouput, coutput, true, P, outputEC, pk, dd);
#endif
          mul(cmacd_output, calpha, ccouput, pk);
          dd.reshare(macd_output, cmacd_output, outputEC);

          for (int k = 0; k < batches_per_convolution; ++k)
          {
            for (int l = 0; l < outputs_per_convolution; ++l)
            {
              auto batch = nDaccess({b, batches_required}, {k, batches_per_convolution});
              auto depth = nDaccess({d, outputs_required}, {l, outputs_per_convolution});
              for (int i = 0; i < H; ++i)
              {
                for (int j = 0; j < W; ++j)
                {
                  if (batch < dimensions.image_batch)
                  {
                    auto source_index = nDaccess({k, batches_per_convolution}, {l, outputs_per_convolution}, last_index(inputs_per_convolution), {i, H}, {j, W});
                    CONV2D_ASSERT(source_index < N);
                    if (depth < dimensions.output_depth)
                    {
                      auto dest_index = nDaccess({batch, dimensions.image_batch}, {i, H}, {j, W}, {depth, dimensions.output_depth});
                      write(presacrificing_output[dest_index], output.coeff(source_index));
                      write(presacrificing_output_mac[dest_index], macd_output.coeff(source_index));
                    }
                    else if (depth < 2 * dimensions.output_depth)
                    {
                      auto dest_index = nDaccess({batch, dimensions.image_batch}, {i, H}, {j, W}, {depth - dimensions.output_depth, dimensions.output_depth});
                      write(presacrificing_output_prime[dest_index], output.coeff(source_index));
                      write(presacrificing_output_prime_mac[dest_index], macd_output.coeff(source_index));
                    }
                  }
                }
              }
            }
          }
#if not CONV2D_SUMMING_CIPHERTEXTS
        }
#endif
      }
    }
    timers["Outputs"].stop();

#ifdef VERBOSE_CONV2D
    std::cerr << "Finished triple production for conv2d.\n";
    std::cerr << "Called filterEC " << filterEC.number_of_calls() << " times (" << filterEC.remaining_capacity() << " unprocessed ciphertexts)\n";
    std::cerr << "Called imageEC " << imageEC.number_of_calls() << " times (" << imageEC.remaining_capacity() << " unprocessed ciphertexts)\n";
    std::cerr << "Called outputEC " << outputEC.number_of_calls() << " times (" << outputEC.remaining_capacity() << " unprocessed ciphertexts)\n";
    auto required_ciphertexts = batches_required * outputs_required * inputs_required;
    std::cerr << "Output utilization: " << required_ciphertexts * N << " slots used for " << 2 * dimensions.full_output_size() << " output elements (" << 200.0 * dimensions.full_output_size() / (required_ciphertexts * N) << "%)\n";
    std::cerr << "       (spacially): " << batches_required * outputs_required * N << " slots used for " << 2 * dimensions.full_output_size() << " output elements (" << 200.0 * dimensions.full_output_size() / (batches_required * outputs_required * N) << "%)\n";
    std::cerr << std::flush;
#endif
}
#else
template<class FD>
void highgear_conv2d(convolution_dimensions dimensions, Player const& P, const FHE_PK& pk, const Ciphertext& calpha, DistDecrypt<FD>& dd, FD const& FieldD, map<string, Timer>& timers, SparseSummingEncCommit<FD>& imageEC, SparseSummingEncCommit<FD>& filterEC, EncCommitBase_<FD>& summingEC, std::span<typename FD::T> presacrificing_image, std::span<typename FD::T> presacrificing_filter, std::span<typename FD::T> presacrificing_filter_prime, std::span<typename FD::T> presacrificing_output, std::span<typename FD::T> presacrificing_output_prime, std::span<typename FD::T> presacrificing_image_mac, std::span<typename FD::T> presacrificing_filter_mac, std::span<typename FD::T> presacrificing_filter_prime_mac, std::span<typename FD::T> presacrificing_output_mac, std::span<typename FD::T> presacrificing_output_prime_mac)
{
    using T = typename FD::T;

    const FHE_Params& params = pk.get_params();

    CONV2D_ASSERT(params.get_plaintext_modulus() == FieldD.get_prime());
    CONV2D_ASSERT(params.phi_m() == FieldD.phi_m());
    CONV2D_ASSERT(FieldD.num_slots() == FieldD.phi_m());

    auto& outputEC = dynamic_cast<SummingEncCommit<FD>&>(summingEC);
#ifdef CONV2D_HIGHGEAR_GENERIC_EC
    auto& imageReshareEC = outputEC;
    auto& filterReshareEC = outputEC;
#else
    auto& imageReshareEC = imageEC;
    auto& filterReshareEC = filterEC;
#endif

    auto H = dimensions.full_output_height();
    auto W = dimensions.full_output_width();
    auto N = FieldD.num_slots();
    auto convs_per_ciphertext = N / (H * W);
    assert(convs_per_ciphertext >= 1);

    int convs_per_image_depth = DIV_CEIL(2 * dimensions.output_depth, convs_per_ciphertext);
#ifdef VERBOSE_CONV2D
    std::cerr << "Starting summing convolution "
#if CONV2D_SUMMING_CIPHERTEXTS
      "(summing over ciphertexts; at most " << CONV2D_MAX_IMAGE_DEPTH << " = " << CONV2D_EXTRA_SLACK << " ?= " << OnlineOptions::singleton.matrix_dimensions << " = " << params.get_matrix_dim() << " ciphertexts) "
#endif
#ifdef CONV2D_HIGHGEAR_GENERIC_EC
      "(using the generic EC for resharing) "
#endif
     << dimensions.as_string() << "\n";
    std::cerr << "H=" << H << ", W=" << W << ", N=" << N << ", conv_per_ciphertext=" << convs_per_ciphertext << ", convs_per_image_depth=" << convs_per_image_depth << "\n";
#endif

    Ciphertext coutput(params), ccouput(params);
    std::vector<std::vector<Ciphertext>> cfilters, cimages;
    Plaintext_<FD> image(FieldD), filter(FieldD), output(FieldD);
    Ciphertext cmacd_image(params), cmacd_filter(params), cmacd_output(params);
    Plaintext_<FD> macd_image(FieldD), macd_filter(FieldD), macd_output(FieldD);

    timers["Filters"].start();
    cfilters.resize(dimensions.image_depth);
    for (int c = 0; c < dimensions.image_depth; ++c)
    {
      cfilters[c].resize(convs_per_image_depth, Ciphertext(params));
      for (int d = 0; d < convs_per_image_depth; ++d)
      {
        filterEC.next(filter, cfilters[c][d]);
        mul(cmacd_filter, calpha, cfilters[c][d], pk);
        dd.reshare(macd_filter, cmacd_filter, filterReshareEC);

        for (int k = 0; k < convs_per_ciphertext; ++k)
        {
          auto depth = nDaccess({d, convs_per_image_depth}, {k, convs_per_ciphertext});
          for (int i = 0; i < dimensions.filter_height; ++i)
          {
            for (int j = 0; j < dimensions.filter_width; ++j)
            {
              auto source_index = nDaccess({k, convs_per_ciphertext}, reverse_index(i, dimensions.filter_height, H), reverse_index(j, dimensions.filter_width, W)); // reverse filter to get cross-correlation instead of a convolution
              if (depth < dimensions.output_depth)
              {
                auto dest_index = nDaccess({depth, dimensions.output_depth}, {i, dimensions.filter_height}, {j, dimensions.filter_width}, {c, dimensions.image_depth});
                presacrificing_filter[dest_index] = filter.coeff(source_index);
                presacrificing_filter_mac[dest_index] = macd_filter.coeff(source_index);
              }
              else if (depth < 2 * dimensions.output_depth)
              {
                auto dest_index = nDaccess({depth - dimensions.output_depth, dimensions.output_depth}, {i, dimensions.filter_height}, {j, dimensions.filter_width}, {c, dimensions.image_depth});
                presacrificing_filter_prime[dest_index] = filter.coeff(source_index);
                presacrificing_filter_prime_mac[dest_index] = macd_filter.coeff(source_index);
              }
            }
          }
        }
      }
    }
    timers["Filters"].stop();
    
    timers["Images"].start();
    cimages.resize(dimensions.image_batch);
    for (int b = 0; b < dimensions.image_batch; ++b)
    {
      cimages[b].resize(dimensions.image_depth, Ciphertext(params));
      for (int c = 0; c < dimensions.image_depth; ++c)
      {
        imageEC.next(image, cimages[b][c]);
        mul(cmacd_image, calpha, cimages[b][c], pk);
        dd.reshare(macd_image, cmacd_image, imageReshareEC);

        for (int i = 0; i < dimensions.image_height; ++i)
        {
          for (int j = 0; j < dimensions.image_width; ++j)
          {
            auto dest_index = nDaccess({b, dimensions.image_batch}, {i, dimensions.image_height}, {j, dimensions.image_width}, {c, dimensions.image_depth});
            auto source_index = nDaccess({i, H}, {j, W});
            presacrificing_image[dest_index] = image.coeff(source_index);
            presacrificing_image_mac[dest_index] = macd_image.coeff(source_index);
          }
        }
      }
    }
    timers["Images"].stop();

    auto write = [](T& x, auto const& y)
    {
#if CONV2D_SUMMING_CIPHERTEXTS
      x = y;
#else
      if constexpr (requires { x += y; })
      {
        x += y;
      }
      else
      {
        x += T{y};
      }
#endif
    };

    timers["Outputs"].start();
    for (int b = 0; b < dimensions.image_batch; ++b)
    {
      for (int d = 0; d < convs_per_image_depth; ++d)
      {
#if CONV2D_SUMMING_CIPHERTEXTS
        assert(dimensions.image_depth <= params.get_matrix_dim());
        Ciphertext coutput_sum(params);
        coutput_sum.allocate();
        coutput_sum.Scale();
#endif
        for (int c = 0; c < dimensions.image_depth; ++c)
        {
          mul(coutput, cimages[b][c], cfilters[c][d], pk);
#if CONV2D_SUMMING_CIPHERTEXTS
          coutput_sum += coutput;
        }
        Reshare(output, ccouput, coutput_sum, true, P, outputEC, pk, dd);
#else
          Reshare(output, ccouput, coutput, true, P, outputEC, pk, dd);
#endif
          mul(cmacd_output, calpha, ccouput, pk);
          dd.reshare(macd_output, cmacd_output, outputEC);
          for (int k = 0; k < convs_per_ciphertext; ++k)
          {
            auto depth = nDaccess({d, convs_per_image_depth}, {k, convs_per_ciphertext});
            for (int i = 0; i < H; ++i)
            {
              for (int j = 0; j < W; ++j)
              {
                auto source_index = nDaccess({k, convs_per_ciphertext}, {i, H}, {j, W});
                if (depth < dimensions.output_depth)
                {
                  auto dest_index = nDaccess({b, dimensions.image_batch}, {i, H}, {j, W}, {depth, dimensions.output_depth});
                  write(presacrificing_output[dest_index], output.coeff(source_index));
                  write(presacrificing_output_mac[dest_index], macd_output.coeff(source_index));
                }
                else if (depth < 2 * dimensions.output_depth)
                {
                  auto dest_index = nDaccess({b, dimensions.image_batch}, {i, H}, {j, W}, {depth - dimensions.output_depth, dimensions.output_depth});
                  write(presacrificing_output_prime[dest_index], output.coeff(source_index));
                  write(presacrificing_output_prime_mac[dest_index], macd_output.coeff(source_index));
                }
              }
            }
          }
#if not CONV2D_SUMMING_CIPHERTEXTS
        }
#endif
      }
    }
    timers["Outputs"].stop();

#ifdef VERBOSE_CONV2D
    std::cerr << "Finished triple production for conv2d.\n";
    std::cerr << "Called filterEC " << filterEC.number_of_calls() << " times (" << filterEC.remaining_capacity() << " unprocessed ciphertexts)\n";
    std::cerr << "Called imageEC " << imageEC.number_of_calls() << " times (" << imageEC.remaining_capacity() << " unprocessed ciphertexts)\n";
    std::cerr << "Called outputEC " << outputEC.number_of_calls() << " times (" << outputEC.remaining_capacity() << " unprocessed ciphertexts)\n";
    auto required_ciphertexts = dimensions.image_batch * dimensions.image_depth * convs_per_image_depth;
    std::cerr << "Output utilization: " << required_ciphertexts * N << " slots used for " << 2 * dimensions.full_output_size() << " output elements (" << 200.0 * dimensions.full_output_size() / (required_ciphertexts * N) << "%)\n";
    std::cerr << "       (spacially): " << dimensions.image_batch * convs_per_image_depth * N << " slots used for " << 2 * dimensions.full_output_size() << " output elements (" << 200.0 * dimensions.full_output_size() / (dimensions.image_batch * convs_per_image_depth * N) << "%)\n";
    std::cerr << std::flush;
#endif
}
#endif
template<typename FD, typename Shares, typename Macs>
void highgear_conv2d(depthwise_convolution_triple_dimensions dimensions, Player const& P, const FHE_PK& pk, const Ciphertext& calpha, DistDecrypt<FD>& dd, FD const& FieldD, map<string, Timer>& timers, SparseSummingEncCommit<FD>& imageEC, SparseSummingEncCommit<FD>& filterEC, EncCommitBase_<FD>& summingEC, indexed_view_of<Shares, 0> presacrificing_image, indexed_view_of<Shares, 1> presacrificing_image_prime, indexed_view_of<Shares, 2> presacrificing_filter, indexed_view_of<Shares, 3> presacrificing_output, indexed_view_of<Shares, 4> presacrificing_output_prime, indexed_view_of<Macs, 0> presacrificing_image_mac, indexed_view_of<Macs, 1> presacrificing_image_prime_mac, indexed_view_of<Macs, 2> presacrificing_filter_mac, indexed_view_of<Macs, 3> presacrificing_output_mac, indexed_view_of<Macs, 4> presacrificing_output_prime_mac)
{
    const FHE_Params& params = pk.get_params();

    CONV2D_ASSERT(params.get_plaintext_modulus() == FieldD.get_prime());
    CONV2D_ASSERT(params.phi_m() == FieldD.phi_m());
    CONV2D_ASSERT(FieldD.num_slots() == FieldD.phi_m());

    auto& outputEC = dynamic_cast<SummingEncCommit<FD>&>(summingEC);
#ifdef CONV2D_HIGHGEAR_GENERIC_EC
    auto& imageReshareEC = outputEC;
    auto& filterReshareEC = outputEC;
#else
    auto& imageReshareEC = imageEC;
    auto& filterReshareEC = filterEC;
#endif

    auto H = dimensions.full_output_height();
    auto W = dimensions.full_output_width();
    auto N = FieldD.num_slots();
    
    auto [batches_per_convolution, batches_required, outputs_per_convolution] = dimensions.depthwise_split(N);
    assert(batches_per_convolution > 0);
    assert(outputs_per_convolution > 0);
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_image.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_image_prime.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_filter.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_output.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_output_prime.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_image_mac.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_image_prime_mac.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_filter_mac.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_output_mac.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_output_prime_mac.size());
#ifdef VERBOSE_CONV2D
    std::cerr << "Starting summing depthwise convolution "
#ifdef CONV2D_HIGHGEAR_GENERIC_EC
      "(using the generic EC for resharing) "
#endif
     << dimensions.as_string() << "\n";
    std::cerr << "H=" << H << ", W=" << W << ", N=" << N << ", batches_per_convolution=" << batches_per_convolution << ", batches_required=" << batches_required << ", outputs_per_convolution=" << outputs_per_convolution << "\n";
#endif

    Ciphertext coutput(params), ccouput(params);
    Ciphertext cimage(params), cfilter(params);
    Plaintext_<FD> image(FieldD), filter(FieldD), output(FieldD);
    Ciphertext cmacd_image(params), cmacd_filter(params), cmacd_output(params);
    Plaintext_<FD> macd_image(FieldD), macd_filter(FieldD), macd_output(FieldD);


    timers["Filters"].start();
    filterEC.next(filter, cfilter);
    mul(cmacd_filter, calpha, cfilter, pk);
    dd.reshare(macd_filter, cmacd_filter, filterReshareEC);

    for (int l = 0; l < outputs_per_convolution; ++l)
    {
      for (int i = 0; i < dimensions.filter_height; ++i)
      {
        for (int j = 0; j < dimensions.filter_width; ++j)
        {
          auto source_index = nDaccess(/*first_index(batches_per_convolution),*/ {l, outputs_per_convolution}, first_index(outputs_per_convolution), reverse_index(i, dimensions.filter_height, H), reverse_index(j, dimensions.filter_width, W)); // reverse filter to get cross-correlation instead of a convolution
          CONV2D_ASSERT(source_index >= 0);
          CONV2D_ASSERT(source_index < N);
          auto dest_index = nDaccess({i, dimensions.filter_height}, {j, dimensions.filter_width});
          presacrificing_filter[l][dest_index] = filter.coeff(source_index);
          presacrificing_filter_mac[l][dest_index] = macd_filter.coeff(source_index);
        }
      }
    }
    timers["Filters"].stop();


    for (int b = 0; b < batches_required; ++b)
    {
      timers["Images"].start();
      imageEC.next(image, cimage);
      mul(cmacd_image, calpha, cimage, pk);
      dd.reshare(macd_image, cmacd_image, imageReshareEC);

      for (int k = 0; k < batches_per_convolution; ++k)
      {
        for (int l = 0; l < outputs_per_convolution; ++l)
        {
          for (int i = 0; i < dimensions.image_height; ++i)
          {
            for (int j = 0; j < dimensions.image_width; ++j)
            {
              auto source_index = nDaccess({k, batches_per_convolution}, first_index(outputs_per_convolution), {l, outputs_per_convolution}, {i, H}, {j, W});
              CONV2D_ASSERT(source_index >= 0);
              CONV2D_ASSERT(source_index < N);
              auto batch = nDaccess({b, batches_required}, {k, batches_per_convolution});
              if (batch < dimensions.image_batch)
              {
                auto dest_index = nDaccess({batch, dimensions.image_batch}, {i, dimensions.image_height}, {j, dimensions.image_width});
                presacrificing_image[l][dest_index] = image.coeff(source_index);
                presacrificing_image_mac[l][dest_index] = macd_image.coeff(source_index);
              }
              else
              {
                CONV2D_ASSERT(batch < 2 * dimensions.image_batch);
              
                auto dest_index = nDaccess({batch - dimensions.image_batch, dimensions.image_batch}, {i, dimensions.image_height}, {j, dimensions.image_width});
                presacrificing_image_prime[l][dest_index] = image.coeff(source_index);
                presacrificing_image_prime_mac[l][dest_index] = macd_image.coeff(source_index);
              }
            }
          }
        }
      }
      timers["Images"].stop();

      timers["Outputs"].start();
      mul(coutput, cimage, cfilter, pk);
      Reshare(output, ccouput, coutput, true, P, outputEC, pk, dd);
      mul(cmacd_output, calpha, ccouput, pk);
      dd.reshare(macd_output, cmacd_output, outputEC);

      for (int k = 0; k < batches_per_convolution; ++k)
      {
        for (int l = 0; l < outputs_per_convolution; ++l)
        {
          for (int i = 0; i < H; ++i)
          {
            for (int j = 0; j < W; ++j)
            {
              auto source_index = nDaccess({k, batches_per_convolution}, {l, outputs_per_convolution}, {l, outputs_per_convolution}, {i, H}, {j, W});
              CONV2D_ASSERT(source_index >= 0);
              CONV2D_ASSERT(source_index < N);
              auto batch = nDaccess({b, batches_required}, {k, batches_per_convolution});
              if (batch < dimensions.image_batch)
              {
                auto dest_index = nDaccess({batch, dimensions.image_batch}, {i, H}, {j, W});
                presacrificing_output[l][dest_index] = output.coeff(source_index);
                presacrificing_output_mac[l][dest_index] = macd_output.coeff(source_index);
              }
              else
              {
                CONV2D_ASSERT(batch < 2 * dimensions.image_batch);
              
                auto dest_index = nDaccess({batch - dimensions.image_batch, dimensions.image_batch}, {i, H}, {j, W});
                presacrificing_output_prime[l][dest_index] = output.coeff(source_index);
                presacrificing_output_prime_mac[l][dest_index] = macd_output.coeff(source_index);
              }
            }
          }
        }
      }
      timers["Outputs"].stop();
    }

#ifdef VERBOSE_CONV2D
    std::cerr << "Finished triple production for conv2d.\n";
    std::cerr << "Called filterEC " << filterEC.number_of_calls() << " times (" << filterEC.remaining_capacity() << " unprocessed ciphertexts)\n";
    std::cerr << "Called imageEC " << imageEC.number_of_calls() << " times (" << imageEC.remaining_capacity() << " unprocessed ciphertexts)\n";
    std::cerr << "Called outputEC " << outputEC.number_of_calls() << " times (" << outputEC.remaining_capacity() << " unprocessed ciphertexts)\n";
    auto required_ciphertexts = batches_required;
    std::cerr << "Output utilization: " << required_ciphertexts * N << " slots used for " << 2 * outputs_per_convolution * dimensions.full_output_size() << " output elements (" << 200.0 * outputs_per_convolution * dimensions.full_output_size() / (required_ciphertexts * N) << "%) for " << outputs_per_convolution << " output channels\n";
    std::cerr << std::flush;
#endif
}


template<class FD>
void SummingConv2dTripleProducer<FD>::run(const Player& P, const FHE_PK& pk, const Ciphertext& calpha, EncCommitBase_<FD>& summingEC, DistDecrypt<FD>& dd, const T&)
{
    auto& [imageEC, filterEC] = this->EC_ptrs;
    CONV2D_ASSERT(imageEC != this->template end<0>());
    CONV2D_ASSERT(filterEC != this->template end<1>());
    if (this->dimensions.is_depthwise())
    {
      highgear_conv2d(this->dimensions.as_depthwise(), P, pk, calpha, dd, this->FieldD, this->timers, imageEC->second, filterEC->second, summingEC, this->template get_presacrificing_shares<0>(), this->template get_presacrificing_shares<1>(), this->template get_presacrificing_shares<2>(), this->template get_presacrificing_shares<3>(), this->template get_presacrificing_shares<4>(), this->template get_presacrificing_macs<0>(), this->template get_presacrificing_macs<1>(), this->template get_presacrificing_macs<2>(), this->template get_presacrificing_macs<3>(), this->template get_presacrificing_macs<4>());
    }
    else
    {
      highgear_conv2d(this->dimensions, P, pk, calpha, dd, this->FieldD, this->timers, imageEC->second, filterEC->second, summingEC, this->presacrificing_shares[0][0], this->presacrificing_shares[0][1], this->presacrificing_shares[0][2], this->presacrificing_shares[0][3], this->presacrificing_shares[0][4], this->presacrificing_macs[0][0], this->presacrificing_macs[0][1], this->presacrificing_macs[0][2], this->presacrificing_macs[0][3], this->presacrificing_macs[0][4]);
    }
}

template<class FD>
template<typename ConvolutionDimensions>
int SummingConv2dTripleProducer<FD>::adapt(ConvolutionDimensions dimensions)
{
    CONV2D_ASSERT(this->FieldD.phi_m() == this->FieldD.num_slots());
    int N = this->FieldD.num_slots();

    this->template try_emplace<0>(dimensions.image_sparcity(N), this->P, this->pk, this->FieldD, this->timers, this->machine, 0, false);
    this->template try_emplace<1>(dimensions.filter_sparcity(N), this->P, this->pk, this->FieldD, this->timers, this->machine, 0, false);

    if constexpr (ConvolutionDimensions::is_always_depthwise)
    {
      auto [batches_per_convolution, batches_required, outputs_per_convolution] = dimensions.depthwise_split(N);
      return outputs_per_convolution;
    }
    else
    {
      return 1;
    }
}

#ifdef CONV2D_DIRECT_SUM
template<typename FD>
void lowgear_conv2d(convolution_dimensions dimensions, std::span<Multiplier<std::type_identity_t<FD>>*> multipliers, Plaintext_<FD> const& alphai, const FHE_Params& params, FD const& FieldD, map<string, Timer>& timers, ReusableSparseMultiEncCommit<FD>& imageEC, ReusableSparseMultiEncCommit<FD>& filterEC, std::span<typename FD::T> presacrificing_image, std::span<typename FD::T> presacrificing_filter, std::span<typename FD::T> presacrificing_filter_prime, std::span<typename FD::T> presacrificing_output, std::span<typename FD::T> presacrificing_output_prime, std::span<typename FD::T> presacrificing_image_mac, std::span<typename FD::T> presacrificing_filter_mac, std::span<typename FD::T> presacrificing_filter_prime_mac, std::span<typename FD::T> presacrificing_output_mac, std::span<typename FD::T> presacrificing_output_prime_mac)
{
    CONV2D_ASSERT(params.get_plaintext_modulus() == FieldD.get_prime());
    CONV2D_ASSERT(params.phi_m() == FieldD.phi_m());
    CONV2D_ASSERT(FieldD.num_slots() == FieldD.phi_m());

    auto H = dimensions.full_output_height();
    auto W = dimensions.full_output_width();
    auto N = FieldD.num_slots();

    auto [batches_per_convolution, batches_required, outputs_per_convolution, outputs_required, inputs_per_convolution, inputs_required] = dimensions.direct_sum_split(N);

#ifdef VERBOSE_CONV2D
    std::cerr << "Starting pairwise (direct sum) convolution "
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    "(with filter ciphertexts instead of image ciphertexts) "
#else
    "(with image ciphertexts instead of filter ciphertexs) "
#endif
      << dimensions.as_string() << "\n";
    std::cerr << "H=" << H << ", W=" << W << ", N=" << N << ", batches_per_convolution=" << batches_per_convolution << ", batches_required=" << batches_required << ", outputs_per_convolution=" << outputs_per_convolution << ", outputs_required=" << outputs_required << ", inputs_per_convolution=" << inputs_per_convolution << ", inputs_required=" << inputs_required << "\n";
#endif
    PRNG G;
    G.ReSeed();

    auto authenticate = [&multipliers, &alphai](Plaintext_<FD>& mac, Plaintext_<FD> const& share, Rq_Element const& share_mod_q)
    {
      mac.mul(alphai, share);
      for (auto& m : multipliers)
      {
        m->multiply_alpha_and_add(mac, share_mod_q);
      }
    };

    auto multiply_and_add = [&multipliers](Plaintext_<FD>& destination, Plaintext_<FD> const& share, Rq_Element const& share_mod_q, Plaintext_<FD> const& my_share, std::vector<Ciphertext> const& ciphertexts_of_others)
    {
      destination.mul(my_share, share);
      CONV2D_ASSERT(multipliers.size() == ciphertexts_of_others.size());
      for (int i = 0; auto& m : multipliers)
      {
        m->multiply_and_add(destination, ciphertexts_of_others[i++], share_mod_q);
      }
    };

    std::vector<std::vector<Plaintext_<FD>>> filters;
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    std::vector<std::vector<std::vector<Ciphertext>>> cfilters_of_others;
#else
    std::vector<std::vector<Rq_Element>> filters_mod_q;
    auto& proofEC = filterEC.get_proof();
#endif

    Plaintext_<FD> macd_filter(FieldD);

    timers["Filters"].start();
    filters.resize(outputs_required);
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    cfilters_of_others.resize(outputs_required);
#else
    filters_mod_q.resize(outputs_required);
#endif
    for (int c = 0; c < outputs_required; ++c)
    {
      filters[c].resize(inputs_required, FieldD);
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
      cfilters_of_others[c].resize(inputs_required);
#else
      filters_mod_q[c].resize(inputs_required, Rq_Element(params, evaluation, evaluation));
#endif
      for (int d = 0; d < inputs_required; ++d)
      {
        auto& filter = filters[c][d];
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
        auto filter_mod_q = Rq_Element(params, evaluation, evaluation);

        filterEC.next(filter, cfilters_of_others[c][d]);
#else
        auto& filter_mod_q = filters_mod_q[c][d];

        proofEC.randomize(G, filter);
#endif
        filter_mod_q.from(filter.get_iterator());
        authenticate(macd_filter, filter, filter_mod_q);

        for (int k = 0; k < outputs_per_convolution; ++k)
        {
          for (int l = 0; l < inputs_per_convolution; ++l)
          {
            auto depth = nDaccess({c, outputs_required}, {k, outputs_per_convolution});
            auto idepth = nDaccess({d, inputs_required}, {l, inputs_per_convolution});
            for (int i = 0; i < dimensions.filter_height; ++i)
            {
              for (int j = 0; j < dimensions.filter_width; ++j)
              {
                auto source_index = nDaccess({k, outputs_per_convolution}, reverse_index(l, inputs_per_convolution), reverse_index(i, dimensions.filter_height, H), reverse_index(j, dimensions.filter_width, W)); // reverse filter to get cross-correlation instead of a convolution; input channels have to be reversed for direct sum as well
                CONV2D_ASSERT(source_index >= 0);
                CONV2D_ASSERT(source_index < N);
                if (depth < dimensions.output_depth)
                {
                  auto dest_index = nDaccess({depth, dimensions.output_depth}, {i, dimensions.filter_height}, {j, dimensions.filter_width}, {idepth, dimensions.image_depth});
                  presacrificing_filter[dest_index] = filter.coeff(source_index);
                  presacrificing_filter_mac[dest_index] = macd_filter.coeff(source_index);
                }
                else if (depth < 2 * dimensions.output_depth)
                {
                  auto dest_index = nDaccess({depth - dimensions.output_depth, dimensions.output_depth}, {i, dimensions.filter_height}, {j, dimensions.filter_width}, {idepth, dimensions.image_depth});
                  presacrificing_filter_prime[dest_index] = filter.coeff(source_index);
                  presacrificing_filter_prime_mac[dest_index] = macd_filter.coeff(source_index);
                }
              }
            }
          }
        }
      }
    }
    timers["Filters"].stop();
    
    std::vector<std::vector<Plaintext_<FD>>> images;
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    std::vector<std::vector<Rq_Element>> images_mod_q;

    auto& proofEC = imageEC.get_proof();
#else
    std::vector<std::vector<std::vector<Ciphertext>>> cimages_of_others;
#endif
    Plaintext_<FD> macd_image(FieldD);

    timers["Images"].start();
    images.resize(batches_required);
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    images_mod_q.resize(batches_required);
#else
    cimages_of_others.resize(batches_required);
#endif
    for (int b = 0; b < batches_required; ++b)
    {
      images[b].resize(inputs_required, FieldD);
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
      images_mod_q[b].resize(inputs_required, Rq_Element(params, evaluation, evaluation));
#else
      cimages_of_others[b].resize(inputs_required);
#endif
      for (int c = 0; c < inputs_required; ++c)
      {
        auto& image = images[b][c];
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
        auto& image_mod_q = images_mod_q[b][c];

        proofEC.randomize(G, image);
#else
        auto image_mod_q = Rq_Element(params, evaluation, evaluation);
        
        imageEC.next(image, cimages_of_others[b][c]);
#endif

        image_mod_q.from(image.get_iterator());
        authenticate(macd_image, image, image_mod_q);

        for (int k = 0; k < batches_per_convolution; ++k)
        {
          for (int l = 0; l < inputs_per_convolution; ++l)
          {
            for (int i = 0; i < dimensions.image_height; ++i)
            {
              for (int j = 0; j < dimensions.image_width; ++j)
              {
                auto batch = nDaccess({b, batches_required}, {k, batches_per_convolution});
                auto idepth = nDaccess({c, inputs_required}, {l, inputs_per_convolution});

                if (batch < dimensions.image_batch)
                {
                  auto dest_index = nDaccess({batch, dimensions.image_batch}, {i, dimensions.image_height}, {j, dimensions.image_width}, {idepth, dimensions.image_depth});
                  auto source_index = nDaccess({k, batches_per_convolution}, first_index(outputs_per_convolution), {l, inputs_per_convolution}, {i, H}, {j, W});
                  CONV2D_ASSERT(source_index < N);
                  presacrificing_image[dest_index] = image.coeff(source_index);
                  presacrificing_image_mac[dest_index] = macd_image.coeff(source_index);
                }
              }
            }
          }
        }
      }
    }
    timers["Images"].stop();

    Plaintext_<FD> output(FieldD);
    Plaintext_<FD> macd_output(FieldD);
    Rq_Element output_mod_q(params, evaluation, evaluation);

    timers["Outputs"].start();
    for (int b = 0; b < batches_required; ++b)
    {
      for (int d = 0; d < outputs_required; ++d)
      {
        Plaintext_<FD> output_sum(FieldD);
        for (int c = 0; c < inputs_required; ++c)
        {
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
          multiply_and_add(output, images[b][c], images_mod_q[b][c], filters[d][c], cfilters_of_others[d][c]);
#else
          multiply_and_add(output, filters[d][c], filters_mod_q[d][c], images[b][c], cimages_of_others[b][c]);
#endif
          output_sum += output;
        }

        output = output_sum;
        output_mod_q.from(output.get_iterator());
        authenticate(macd_output, output, output_mod_q);

        for (int k = 0; k < batches_per_convolution; ++k)
        {
          for (int l = 0; l < outputs_per_convolution; ++l)
          {
            auto batch = nDaccess({b, batches_required}, {k, batches_per_convolution});
            auto depth = nDaccess({d, outputs_required}, {l, outputs_per_convolution});
            for (int i = 0; i < H; ++i)
            {
              for (int j = 0; j < W; ++j)
              {
                if (batch < dimensions.image_batch)
                {
                  auto source_index = nDaccess({k, batches_per_convolution}, {l, outputs_per_convolution}, last_index(inputs_per_convolution), {i, H}, {j, W});
                  CONV2D_ASSERT(source_index < N);
                  if (depth < dimensions.output_depth)
                  {
                    auto dest_index = nDaccess({batch, dimensions.image_batch}, {i, H}, {j, W}, {depth, dimensions.output_depth});
                    presacrificing_output[dest_index] = output.coeff(source_index);
                    presacrificing_output_mac[dest_index] = macd_output.coeff(source_index);
                  }
                  else if (depth < 2 * dimensions.output_depth)
                  {
                    auto dest_index = nDaccess({batch, dimensions.image_batch}, {i, H}, {j, W}, {depth - dimensions.output_depth, dimensions.output_depth});
                    presacrificing_output_prime[dest_index] = output.coeff(source_index);
                    presacrificing_output_prime_mac[dest_index] = macd_output.coeff(source_index);
                  }
                }
              }
            }
          }
        }
      }
    }
    timers["Outputs"].stop();
#ifdef VERBOSE_CONV2D
    std::cerr << "Finished triple production for conv2d.\n";
    std::cerr << "Called filterEC " << filterEC.number_of_calls() << " times (" << filterEC.remaining_capacity() << " unprocessed ciphertexts)\n";
    std::cerr << "Called imageEC " << imageEC.number_of_calls() << " times (" << imageEC.remaining_capacity() << " unprocessed ciphertexts)\n";
    auto output_sparcity = dimensions.output_sparcity(N);
    auto required_ciphertexts = batches_required * outputs_required * inputs_required;
    std::cerr << "Output utilization: " << required_ciphertexts * N << " slots used for " << 2 * dimensions.full_output_size() << " output elements (" << 200.0 * dimensions.full_output_size() / (required_ciphertexts * N) << "%)\n";
    std::cerr << "       (spacially): " << batches_required * outputs_required * N << " slots used for " << 2 * dimensions.full_output_size() << " output elements (" << 200.0 * dimensions.full_output_size() / (batches_required * outputs_required * N) << "%)\n";
    std::cerr << std::flush;
#endif
}
#else
template<typename FD>
void lowgear_conv2d(convolution_dimensions dimensions, std::span<Multiplier<std::type_identity_t<FD>>*> multipliers, Plaintext_<FD> const& alphai, const FHE_Params& params, FD const& FieldD, map<string, Timer>& timers, ReusableSparseMultiEncCommit<FD>& imageEC, ReusableSparseMultiEncCommit<FD>& filterEC, std::span<typename FD::T> presacrificing_image, std::span<typename FD::T> presacrificing_filter, std::span<typename FD::T> presacrificing_filter_prime, std::span<typename FD::T> presacrificing_output, std::span<typename FD::T> presacrificing_output_prime, std::span<typename FD::T> presacrificing_image_mac, std::span<typename FD::T> presacrificing_filter_mac, std::span<typename FD::T> presacrificing_filter_prime_mac, std::span<typename FD::T> presacrificing_output_mac, std::span<typename FD::T> presacrificing_output_prime_mac)
{
    CONV2D_ASSERT(params.get_plaintext_modulus() == FieldD.get_prime());
    CONV2D_ASSERT(params.phi_m() == FieldD.phi_m());
    CONV2D_ASSERT(FieldD.num_slots() == FieldD.phi_m());

    auto H = dimensions.full_output_height();
    auto W = dimensions.full_output_width();
    auto N = FieldD.num_slots();
    auto convs_per_ciphertext = N / (H * W);
    assert(convs_per_ciphertext >= 1);

    int convs_per_image_depth = DIV_CEIL(2 * dimensions.output_depth, convs_per_ciphertext);
#ifdef VERBOSE_CONV2D
    std::cerr << "Starting pairwise convolution "
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    "(with filter ciphertexts instead of image ciphertexts) "
#else
    "(with image ciphertext instead of filter ciphertexts) "
#endif
    << dimensions.as_string() << "\n";
    std::cerr << "H=" << H << ", W=" << W << ", N=" << N << ", conv_per_ciphertext=" << convs_per_ciphertext << ", convs_per_image_depth=" << convs_per_image_depth << "\n";
#endif
    PRNG G;
    G.ReSeed();

    auto authenticate = [&multipliers, &alphai](Plaintext_<FD>& mac, Plaintext_<FD> const& share, Rq_Element const& share_mod_q)
    {
      mac.mul(alphai, share);
      for (auto& m : multipliers)
      {
        m->multiply_alpha_and_add(mac, share_mod_q);
      }
    };

    auto multiply_and_add = [&multipliers](Plaintext_<FD>& destination, Plaintext_<FD> const& share, Rq_Element const& share_mod_q, Plaintext_<FD> const& my_share, std::vector<Ciphertext> const& ciphertexts_of_others)
    {
      destination.mul(my_share, share);
      CONV2D_ASSERT(multipliers.size() == ciphertexts_of_others.size());
      for (int i = 0; auto& m : multipliers)
      {
        m->multiply_and_add(destination, ciphertexts_of_others[i++], share_mod_q);
      }
    };

    std::vector<std::vector<Plaintext_<FD>>> filters;
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    std::vector<std::vector<std::vector<Ciphertext>>> cfilters_of_others;
#else
    std::vector<std::vector<Rq_Element>> filters_mod_q;
    auto& proofEC = filterEC.get_proof();
#endif

    Plaintext_<FD> macd_filter(FieldD);

    timers["Filters"].start();
    filters.resize(dimensions.image_depth);
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    cfilters_of_others.resize(dimensions.image_depth);
#else
    filters_mod_q.resize(dimensions.image_depth);
#endif
    for (int c = 0; c < dimensions.image_depth; ++c)
    {
      filters[c].resize(convs_per_image_depth, FieldD);
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
      cfilters_of_others[c].resize(convs_per_image_depth);
#else
      filters_mod_q[c].resize(convs_per_image_depth, Rq_Element(params, evaluation, evaluation));
#endif
      for (int d = 0; d < convs_per_image_depth; ++d)
      {
        auto& filter = filters[c][d];
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
        auto filter_mod_q = Rq_Element(params, evaluation, evaluation);

        filterEC.next(filter, cfilters_of_others[c][d]);
#else
        auto& filter_mod_q = filters_mod_q[c][d];

        proofEC.randomize(G, filter);
#endif
        filter_mod_q.from(filter.get_iterator());
        authenticate(macd_filter, filter, filter_mod_q);

        for (int k = 0; k < convs_per_ciphertext; ++k)
        {
          auto depth = nDaccess({d, convs_per_image_depth}, {k, convs_per_ciphertext});
          for (int i = 0; i < dimensions.filter_height; ++i)
          {
            for (int j = 0; j < dimensions.filter_width; ++j)
            {
              auto source_index = nDaccess({k, convs_per_ciphertext}, reverse_index(i, dimensions.filter_height, H), reverse_index(j, dimensions.filter_width, W)); // reverse filter to get cross-correlation instead of a convolution
              if (depth < dimensions.output_depth)
              {
                auto dest_index = nDaccess({depth, dimensions.output_depth}, {i, dimensions.filter_height}, {j, dimensions.filter_width}, {c, dimensions.image_depth});
                presacrificing_filter[dest_index] = filter.coeff(source_index);
                presacrificing_filter_mac[dest_index] = macd_filter.coeff(source_index);
              }
              else if (depth < 2 * dimensions.output_depth)
              {
                auto dest_index = nDaccess({depth - dimensions.output_depth, dimensions.output_depth}, {i, dimensions.filter_height}, {j, dimensions.filter_width}, {c, dimensions.image_depth});
                presacrificing_filter_prime[dest_index] = filter.coeff(source_index);
                presacrificing_filter_prime_mac[dest_index] = macd_filter.coeff(source_index);
              }
            }
          }
        }
      }
    }
    timers["Filters"].stop();
    
    std::vector<std::vector<Plaintext_<FD>>> images;
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    std::vector<std::vector<Rq_Element>> images_mod_q;

    auto& proofEC = imageEC.get_proof();
#else
    std::vector<std::vector<std::vector<Ciphertext>>> cimages_of_others;
#endif
    Plaintext_<FD> macd_image(FieldD);

    timers["Images"].start();
    images.resize(dimensions.image_batch);
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    images_mod_q.resize(dimensions.image_batch);
#else
    cimages_of_others.resize(dimensions.image_batch);
#endif
    for (int b = 0; b < dimensions.image_batch; ++b)
    {
      images[b].resize(dimensions.image_depth, FieldD);
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
      images_mod_q[b].resize(dimensions.image_depth, Rq_Element(params, evaluation, evaluation));
#else
      cimages_of_others[b].resize(dimensions.image_depth);
#endif
      for (int c = 0; c < dimensions.image_depth; ++c)
      {
        auto& image = images[b][c];
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
        auto& image_mod_q = images_mod_q[b][c];

        proofEC.randomize(G, image);
#else
        auto image_mod_q = Rq_Element(params, evaluation, evaluation);
        
        imageEC.next(image, cimages_of_others[b][c]);
#endif

        image_mod_q.from(image.get_iterator());
        authenticate(macd_image, image, image_mod_q);

        for (int i = 0; i < dimensions.image_height; ++i)
        {
          for (int j = 0; j < dimensions.image_width; ++j)
          {
            auto dest_index = nDaccess({b, dimensions.image_batch}, {i, dimensions.image_height}, {j, dimensions.image_width}, {c, dimensions.image_depth});
            auto source_index = nDaccess({i, H}, {j, W});
            presacrificing_image[dest_index] = image.coeff(source_index);
            presacrificing_image_mac[dest_index] = macd_image.coeff(source_index);
          }
        }
      }
    }
    timers["Images"].stop();

    Plaintext_<FD> output(FieldD);
    Plaintext_<FD> macd_output(FieldD);
    Rq_Element output_mod_q(params, evaluation, evaluation);

    timers["Outputs"].start();
    for (int b = 0; b < dimensions.image_batch; ++b)
    {
      for (int d = 0; d < convs_per_image_depth; ++d)
      {
        Plaintext_<FD> output_sum(FieldD);
        for (int c = 0; c < dimensions.image_depth; ++c)
        {
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
          multiply_and_add(output, images[b][c], images_mod_q[b][c], filters[c][d], cfilters_of_others[c][d]);
#else
          multiply_and_add(output, filters[c][d], filters_mod_q[c][d], images[b][c], cimages_of_others[b][c]);
#endif
          output_sum += output;
        }

        output = output_sum;
        output_mod_q.from(output.get_iterator());
        authenticate(macd_output, output, output_mod_q);

        for (int k = 0; k < convs_per_ciphertext; ++k)
        {
          auto depth = nDaccess({d, convs_per_image_depth}, {k, convs_per_ciphertext});
          for (int i = 0; i < H; ++i)
          {
            for (int j = 0; j < W; ++j)
            {
              auto source_index = nDaccess({k, convs_per_ciphertext}, {i, H}, {j, W});
              if (depth < dimensions.output_depth)
              {
                auto dest_index = nDaccess({b, dimensions.image_batch}, {i, H}, {j, W}, {depth, dimensions.output_depth});
                presacrificing_output[dest_index] = output.coeff(source_index);
                presacrificing_output_mac[dest_index] = macd_output.coeff(source_index);
              }
              else if (depth < 2 * dimensions.output_depth)
              {
                auto dest_index = nDaccess({b, dimensions.image_batch}, {i, H}, {j, W}, {depth - dimensions.output_depth, dimensions.output_depth});
                presacrificing_output_prime[dest_index] = output.coeff(source_index);
                presacrificing_output_prime_mac[dest_index] = macd_output.coeff(source_index);
              }
            }
          }
        }
      }
    }
    timers["Outputs"].stop();
#ifdef VERBOSE_CONV2D
    std::cerr << "Finished triple production for conv2d.\n";
    std::cerr << "Called filterEC " << filterEC.number_of_calls() << " times (" << filterEC.remaining_capacity() << " unprocessed ciphertexts)\n";
    std::cerr << "Called imageEC " << imageEC.number_of_calls() << " times (" << imageEC.remaining_capacity() << " unprocessed ciphertexts)\n";
    auto required_ciphertexts = dimensions.image_batch * dimensions.image_depth * convs_per_image_depth;
    std::cerr << "Output utilization: " << required_ciphertexts * N << " slots used for " << 2 * dimensions.full_output_size() << " output elements (" << 200.0 * dimensions.full_output_size() / (required_ciphertexts * N) << "%)\n";
    std::cerr << "       (spacially): " << dimensions.image_batch * convs_per_image_depth * N << " slots used for " << 2 * dimensions.full_output_size() << " output elements (" << 200.0 * dimensions.full_output_size() / (dimensions.image_batch * convs_per_image_depth * N) << "%)\n";
    std::cerr << std::flush;
#endif
}
#endif

#ifdef CONV2D_LOWGEAR_EXPANDED_BGV
template<typename FD, typename Shares, typename Macs>
void lowgear_conv2d(depthwise_convolution_triple_dimensions dimensions, std::span<Multiplier<std::type_identity_t<FD>>*> multipliers, Plaintext_<FD> const& alphai, const FHE_Params& params, FD const& FieldD, map<string, Timer>& timers, ReusableSparseMultiEncCommit<FD>&, ReusableSparseMultiEncCommit<FD>&, ReusableMultiEncCommit<FD>& EC, indexed_view_of<Shares, 0> presacrificing_image, indexed_view_of<Shares, 1> presacrificing_image_prime, indexed_view_of<Shares, 2> presacrificing_filter, indexed_view_of<Shares, 3> presacrificing_output, indexed_view_of<Shares, 4> presacrificing_output_prime, indexed_view_of<Macs, 0> presacrificing_image_mac, indexed_view_of<Macs, 1> presacrificing_image_prime_mac, indexed_view_of<Macs, 2> presacrificing_filter_mac, indexed_view_of<Macs, 3> presacrificing_output_mac, indexed_view_of<Macs, 4> presacrificing_output_prime_mac)
{
    CONV2D_ASSERT(params.get_plaintext_modulus() == FieldD.get_prime());
    CONV2D_ASSERT(params.phi_m() == FieldD.phi_m());
    CONV2D_ASSERT(FieldD.num_slots() == FieldD.phi_m());

    auto H = dimensions.full_output_height();
    auto W = dimensions.full_output_width();
    auto N = FieldD.num_slots();

    auto [batches_per_convolution, batches_required, outputs_per_convolution] = dimensions.depthwise_split(N);
    assert(batches_per_convolution > 0);
    assert(outputs_per_convolution > 0);
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_image.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_image_prime.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_filter.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_output.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_output_prime.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_image_mac.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_image_prime_mac.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_filter_mac.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_output_mac.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_output_prime_mac.size());
#ifdef VERBOSE_CONV2D
    std::cerr << "Starting pairwise depthwise convolution with expanded ciphertexts " << dimensions.as_string() << "\n";
    std::cerr << "H=" << H << ", W=" << W << ", N=" << N << ", batches_per_convolution=" << batches_per_convolution << ", batches_required=" << batches_required << ", outputs_per_convolution=" << outputs_per_convolution << "\n";
#endif
    PRNG G;
    G.ReSeed();

    auto authenticate = [&multipliers, &alphai](Plaintext_<FD>& mac, Plaintext_<FD> const& share, Rq_Element const& share_mod_q)
    {
      mac.mul(alphai, share);
      for (auto& m : multipliers)
      {
        m->multiply_alpha_and_add(mac, share_mod_q);
      }
    };

    auto conv_and_add = [&multipliers](Plaintext_<FD>& destination, MultiConvolution_PlaintextMatrix_<FD> const& share, MultiConvolution_Matrix const& share_mod_q, Plaintext_<FD> const& my_share, std::vector<Ciphertext> const& ciphertexts_of_others)
    {
      destination = share * my_share;
      CONV2D_ASSERT(multipliers.size() == ciphertexts_of_others.size());
      for (int i = 0; auto& m : multipliers)
      {
        m->conv_and_add(destination, ciphertexts_of_others[i++], share_mod_q);
      }
    };

    timers["Filters"].start();
    MultiConvolution_PlaintextMatrix_<FD> filter(FieldD, dimensions);
    MultiConvolution_Matrix filter_mod_q(params.FFTD()[0], dimensions);
    Plaintext_<FD> macd_filter(FieldD);

    filter.randomize(filter_mod_q, G);

    {
      auto plain_filter = static_cast<Plaintext_<FD>>(filter);
      auto plain_filter_mod_q =  Rq_Element(params, evaluation, evaluation);
      plain_filter_mod_q.from(plain_filter.get_iterator());
      authenticate(macd_filter, plain_filter, plain_filter_mod_q);
    }

    for (int l = 0; l < outputs_per_convolution; ++l)
    {
      for (int i = 0; i < dimensions.filter_height; ++i)
      {
        for (int j = 0; j < dimensions.filter_width; ++j)
        {
          auto source_index = nDaccess({l, outputs_per_convolution}, {i, dimensions.filter_height}, {j, dimensions.filter_width});
          CONV2D_ASSERT(source_index >= 0);
          CONV2D_ASSERT(static_cast<std::size_t>(source_index) < filter.size());
          auto dest_index = nDaccess({i, dimensions.filter_height}, {j, dimensions.filter_width});
          presacrificing_filter[l][dest_index] = filter.coeff(source_index);
          presacrificing_filter_mac[l][dest_index] = macd_filter.coeff(source_index);
        }
      }
    }
    timers["Filters"].stop();

    for (int b = 0; b < batches_required; ++b)
    {
      timers["Images"].start();
      Plaintext_<FD> image(FieldD), macd_image(FieldD);
      std::vector<Ciphertext> cimage_of_others;

      EC.next(image, cimage_of_others);

      {
        auto image_mod_q = Rq_Element(params, evaluation, evaluation);
        image_mod_q.from(image.get_iterator());
        authenticate(macd_image, image, image_mod_q);
      }

      for (int k = 0; k < batches_per_convolution; ++k)
      {
        for (int l = 0; l < outputs_per_convolution; ++l)
        {
          for (int i = 0; i < dimensions.image_height; ++i)
          {
            for (int j = 0; j < dimensions.image_width; ++j)
            {
              auto source_index = nDaccess({l, outputs_per_convolution}, {k, batches_per_convolution}, {i, H}, {j, W});
              CONV2D_ASSERT(source_index >= 0);
              CONV2D_ASSERT(source_index < N);
              auto batch = nDaccess({b, batches_required}, {k, batches_per_convolution});
              if (batch < dimensions.image_batch)
              {
                auto dest_index = nDaccess({batch, dimensions.image_batch}, {i, dimensions.image_height}, {j, dimensions.image_width});
                presacrificing_image[l][dest_index] = image.coeff(source_index);
                presacrificing_image_mac[l][dest_index] = macd_image.coeff(source_index);
              }
              else
              {
                CONV2D_ASSERT(batch < 2 * dimensions.image_batch);
              
                auto dest_index = nDaccess({batch - dimensions.image_batch, dimensions.image_batch}, {i, dimensions.image_height}, {j, dimensions.image_width});
                presacrificing_image_prime[l][dest_index] = image.coeff(source_index);
                presacrificing_image_prime_mac[l][dest_index] = macd_image.coeff(source_index);
              }
            }
          }
        }
      }
      timers["Images"].stop();

      timers["Outputs"].start();
      Plaintext_<FD> output(FieldD), macd_output(FieldD);
      conv_and_add(output, filter, filter_mod_q, image, cimage_of_others);
      auto output_mod_q = Rq_Element(params, evaluation, evaluation);
      output_mod_q.from(output.get_iterator());
      authenticate(macd_output, output, output_mod_q);
      
      for (int k = 0; k < batches_per_convolution; ++k)
      {
        for (int l = 0; l < outputs_per_convolution; ++l)
        {
          for (int i = 0; i < H; ++i)
          {
            for (int j = 0; j < W; ++j)
            {
              auto source_index = nDaccess({l, outputs_per_convolution}, {k, batches_per_convolution},  {i, H}, {j, W});
              CONV2D_ASSERT(source_index >= 0);
              CONV2D_ASSERT(source_index < N);
              auto batch = nDaccess({b, batches_required}, {k, batches_per_convolution});
              if (batch < dimensions.image_batch)
              {
                auto dest_index = nDaccess({batch, dimensions.image_batch}, {i, H}, {j, W});
                presacrificing_output[l][dest_index] = output.coeff(source_index);
                presacrificing_output_mac[l][dest_index] = macd_output.coeff(source_index);
              }
              else
              {
                CONV2D_ASSERT(batch < 2 * dimensions.image_batch);
              
                auto dest_index = nDaccess({batch - dimensions.image_batch, dimensions.image_batch}, {i, H}, {j, W});
                presacrificing_output_prime[l][dest_index] = output.coeff(source_index);
                presacrificing_output_prime_mac[l][dest_index] = macd_output.coeff(source_index);
              }
            }
          }
        }
      }
      timers["Outputs"].stop();
    }

#ifdef VERBOSE_CONV2D
    std::cerr << "Finished triple production for conv2d.\n";
    std::cerr << "Called EC " << EC.number_of_calls() << " times (" << EC.remaining_capacity() << " unprocessed ciphertexts)\n";
    auto required_ciphertexts = batches_required;
    std::cerr << "Output utilization: " << required_ciphertexts * N << " slots used for " << 2 * outputs_per_convolution * dimensions.full_output_size() << " output elements (" << 200.0 * outputs_per_convolution * dimensions.full_output_size() / (required_ciphertexts * N) << "%) for " << outputs_per_convolution << " output channels\n";
    std::cerr << std::flush;
#endif
}
#else
template<typename FD, typename ReusableEC, typename Shares, typename Macs>
void lowgear_conv2d(depthwise_convolution_triple_dimensions dimensions, std::span<Multiplier<std::type_identity_t<FD>>*> multipliers, Plaintext_<FD> const& alphai, const FHE_Params& params, FD const& FieldD, map<string, Timer>& timers, ReusableSparseMultiEncCommit<FD>& imageEC, ReusableSparseMultiEncCommit<FD>& filterEC, ReusableEC&, indexed_view_of<Shares, 0> presacrificing_image, indexed_view_of<Shares, 1> presacrificing_image_prime, indexed_view_of<Shares, 2> presacrificing_filter, indexed_view_of<Shares, 3> presacrificing_output, indexed_view_of<Shares, 4> presacrificing_output_prime, indexed_view_of<Macs, 0> presacrificing_image_mac, indexed_view_of<Macs, 1> presacrificing_image_prime_mac, indexed_view_of<Macs, 2> presacrificing_filter_mac, indexed_view_of<Macs, 3> presacrificing_output_mac, indexed_view_of<Macs, 4> presacrificing_output_prime_mac)
{
    CONV2D_ASSERT(params.get_plaintext_modulus() == FieldD.get_prime());
    CONV2D_ASSERT(params.phi_m() == FieldD.phi_m());
    CONV2D_ASSERT(FieldD.num_slots() == FieldD.phi_m());

    auto H = dimensions.full_output_height();
    auto W = dimensions.full_output_width();
    auto N = FieldD.num_slots();

    auto [batches_per_convolution, batches_required, outputs_per_convolution] = dimensions.depthwise_split(N);
    assert(batches_per_convolution > 0);
    assert(outputs_per_convolution > 0);
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_image.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_image_prime.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_filter.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_output.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_output_prime.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_image_mac.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_image_prime_mac.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_filter_mac.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_output_mac.size());
    CONV2D_ASSERT(static_cast<std::size_t>(outputs_per_convolution) == presacrificing_output_prime_mac.size());
#ifdef VERBOSE_CONV2D
    std::cerr << "Starting pairwise depthwise convolution "
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    "(with filter ciphertexts instead of image ciphertexts) "
#else
    "(with image ciphertext instead of filter ciphertexts) "
#endif
    << dimensions.as_string() << "\n";
    std::cerr << "H=" << H << ", W=" << W << ", N=" << N << ", batches_per_convolution=" << batches_per_convolution << ", batches_required=" << batches_required << ", outputs_per_convolution=" << outputs_per_convolution << "\n";
#endif
    PRNG G;
    G.ReSeed();

    auto authenticate = [&multipliers, &alphai](Plaintext_<FD>& mac, Plaintext_<FD> const& share, Rq_Element const& share_mod_q)
    {
      mac.mul(alphai, share);
      for (auto& m : multipliers)
      {
        m->multiply_alpha_and_add(mac, share_mod_q);
      }
    };

    auto multiply_and_add = [&multipliers](Plaintext_<FD>& destination, Plaintext_<FD> const& share, Rq_Element const& share_mod_q, Plaintext_<FD> const& my_share, std::vector<Ciphertext> const& ciphertexts_of_others)
    {
      destination.mul(my_share, share);
      CONV2D_ASSERT(multipliers.size() == ciphertexts_of_others.size());
      for (int i = 0; auto& m : multipliers)
      {
        m->multiply_and_add(destination, ciphertexts_of_others[i++], share_mod_q);
      }
    };

    Plaintext_<FD> filter(FieldD);
    auto filter_mod_q = Rq_Element(params, evaluation, evaluation);
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    std::vector<Ciphertext> cfilter_of_others;
#endif
    Plaintext_<FD> macd_filter(FieldD);

    timers["Filters"].start();

#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    filterEC.next(filter, cfilter_of_others);
#else
    filterEC.get_proof().randomize(G, filter);
#endif
    filter_mod_q.from(filter.get_iterator());
    authenticate(macd_filter, filter, filter_mod_q);

    for (int l = 0; l < outputs_per_convolution; ++l)
    {
      for (int i = 0; i < dimensions.filter_height; ++i)
      {
        for (int j = 0; j < dimensions.filter_width; ++j)
        {
          auto source_index = nDaccess(/*first_index(batches_per_convolution),*/ {l, outputs_per_convolution}, first_index(outputs_per_convolution), reverse_index(i, dimensions.filter_height, H), reverse_index(j, dimensions.filter_width, W)); // reverse filter to get cross-correlation instead of a convolution
          CONV2D_ASSERT(source_index >= 0);
          CONV2D_ASSERT(source_index < N);
          auto dest_index = nDaccess({i, dimensions.filter_height}, {j, dimensions.filter_width});
          presacrificing_filter[l][dest_index] = filter.coeff(source_index);
          presacrificing_filter_mac[l][dest_index] = macd_filter.coeff(source_index);
        }
      }
    }
    timers["Filters"].stop();
    

    for (int b = 0; b < batches_required; ++b)
    {
      Plaintext_<FD> image(FieldD);
      auto image_mod_q = Rq_Element(params, evaluation, evaluation);
#ifndef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
      std::vector<Ciphertext> cimage_of_others;
#endif
      Plaintext_<FD> macd_image(FieldD);

      timers["Images"].start();
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
      imageEC.get_proof().randomize(G, image);
#else
      imageEC.next(image, cimage_of_others);
#endif
      image_mod_q.from(image.get_iterator());
      authenticate(macd_image, image, image_mod_q);

      for (int k = 0; k < batches_per_convolution; ++k)
      {
        for (int l = 0; l < outputs_per_convolution; ++l)
        {
          for (int i = 0; i < dimensions.image_height; ++i)
          {
            for (int j = 0; j < dimensions.image_width; ++j)
            {
              auto source_index = nDaccess({k, batches_per_convolution}, first_index(outputs_per_convolution), {l, outputs_per_convolution}, {i, H}, {j, W});
              CONV2D_ASSERT(source_index >= 0);
              CONV2D_ASSERT(source_index < N);
              auto batch = nDaccess({b, batches_required}, {k, batches_per_convolution});
              if (batch < dimensions.image_batch)
              {
                auto dest_index = nDaccess({batch, dimensions.image_batch}, {i, dimensions.image_height}, {j, dimensions.image_width});
                presacrificing_image[l][dest_index] = image.coeff(source_index);
                presacrificing_image_mac[l][dest_index] = macd_image.coeff(source_index);
              }
              else
              {
                CONV2D_ASSERT(batch < 2 * dimensions.image_batch);
              
                auto dest_index = nDaccess({batch - dimensions.image_batch, dimensions.image_batch}, {i, dimensions.image_height}, {j, dimensions.image_width});
                presacrificing_image_prime[l][dest_index] = image.coeff(source_index);
                presacrificing_image_prime_mac[l][dest_index] = macd_image.coeff(source_index);
              }
            }
          }
        }
      }
      timers["Images"].stop();

      Plaintext_<FD> output(FieldD);
      auto output_mod_q = Rq_Element(params, evaluation, evaluation);
      Plaintext_<FD> macd_output(FieldD);
      timers["Outputs"].start();
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
      multiply_and_add(output, image, image_mod_q, filter, cfilter_of_others);
#else
      multiply_and_add(output, filter, filter_mod_q, image, cimage_of_others);
#endif
      output_mod_q.from(output.get_iterator());

      authenticate(macd_output, output, output_mod_q);

      for (int k = 0; k < batches_per_convolution; ++k)
      {
        for (int l = 0; l < outputs_per_convolution; ++l)
        {
          for (int i = 0; i < H; ++i)
          {
            for (int j = 0; j < W; ++j)
            {
              auto source_index = nDaccess({k, batches_per_convolution}, {l, outputs_per_convolution}, {l, outputs_per_convolution}, {i, H}, {j, W});
              CONV2D_ASSERT(source_index >= 0);
              CONV2D_ASSERT(source_index < N);
              auto batch = nDaccess({b, batches_required}, {k, batches_per_convolution});
              if (batch < dimensions.image_batch)
              {
                auto dest_index = nDaccess({batch, dimensions.image_batch}, {i, H}, {j, W});
                presacrificing_output[l][dest_index] = output.coeff(source_index);
                presacrificing_output_mac[l][dest_index] = macd_output.coeff(source_index);
              }
              else
              {
                CONV2D_ASSERT(batch < 2 * dimensions.image_batch);
              
                auto dest_index = nDaccess({batch - dimensions.image_batch, dimensions.image_batch}, {i, H}, {j, W});
                presacrificing_output_prime[l][dest_index] = output.coeff(source_index);
                presacrificing_output_prime_mac[l][dest_index] = macd_output.coeff(source_index);
              }
            }
          }
        }
      }
      timers["Outputs"].stop();
    }
#ifdef VERBOSE_CONV2D
    std::cerr << "Finished triple production for conv2d.\n";
    std::cerr << "Called filterEC " << filterEC.number_of_calls() << " times (" << filterEC.remaining_capacity() << " unprocessed ciphertexts)\n";
    std::cerr << "Called imageEC " << imageEC.number_of_calls() << " times (" << imageEC.remaining_capacity() << " unprocessed ciphertexts)\n";
    auto required_ciphertexts = batches_required;
    std::cerr << "Output utilization: " << required_ciphertexts * N << " slots used for " << 2 * outputs_per_convolution * dimensions.full_output_size() << " output elements (" << 200.0 * outputs_per_convolution * dimensions.full_output_size() / (required_ciphertexts * N) << "%) for " << outputs_per_convolution << " output channels\n";
    std::cerr << std::flush;
#endif
}
#endif

template<class FD>
void PairwiseConv2dTripleProducer<FD>::run(const Player&, const FHE_PK&, const Ciphertext&, EncCommitBase_<FD>&, DistDecrypt<FD>&, const T&)
{
    auto& [imageEC, filterEC, reusableEC] = this->EC_ptrs;
    CONV2D_ASSERT(imageEC != this->template end<0>());
    CONV2D_ASSERT(filterEC != this->template end<1>());
    CONV2D_ASSERT(reusableEC != this->template end<2>());
    auto& setup = this->generator.machine.template setup<FD>();
    auto& multipliers = this->generator.multipliers;
    if (this->dimensions.is_depthwise())
    {
      lowgear_conv2d(this->dimensions.as_depthwise(), multipliers, setup.alpha, setup.params, setup.FieldD, this->timers, imageEC->second, filterEC->second, reusableEC->second, this->template get_presacrificing_shares<0>(), this->template get_presacrificing_shares<1>(), this->template get_presacrificing_shares<2>(), this->template get_presacrificing_shares<3>(), this->template get_presacrificing_shares<4>(), this->template get_presacrificing_macs<0>(), this->template get_presacrificing_macs<1>(), this->template get_presacrificing_macs<2>(), this->template get_presacrificing_macs<3>(), this->template get_presacrificing_macs<4>());
    }
    else
    {
      lowgear_conv2d(this->dimensions, multipliers, setup.alpha, setup.params, setup.FieldD, this->timers, imageEC->second, filterEC->second, this->presacrificing_shares[0][0], this->presacrificing_shares[0][1], this->presacrificing_shares[0][2], this->presacrificing_shares[0][3], this->presacrificing_shares[0][4], this->presacrificing_macs[0][0], this->presacrificing_macs[0][1], this->presacrificing_macs[0][2], this->presacrificing_macs[0][3], this->presacrificing_macs[0][4]);
    }
}

template<class FD>
template<typename ConvolutionDimensions>
int PairwiseConv2dTripleProducer<FD>::adapt(ConvolutionDimensions dimensions)
{
    auto& generator = this->generator;
    auto& FieldD = generator.machine.template setup<FD>().FieldD;
    CONV2D_ASSERT(FieldD.phi_m() == FieldD.num_slots());
    int N = FieldD.num_slots();
    this->template try_emplace<0>(dimensions.image_sparcity(N), generator.P, generator.machine.other_pks, FieldD, this->timers, generator.machine, generator, false);
#ifdef CONV2D_LOWGEAR_EXPANDED_BGV
    constexpr auto is_expanded = true;
#else
    constexpr auto is_expanded = false;
#endif

    if constexpr (not (is_expanded and ConvolutionDimensions::is_always_depthwise))
    {
      this->template try_emplace<1>(dimensions.filter_sparcity(N), generator.P, generator.machine.other_pks, FieldD, this->timers, generator.machine, generator, false);
    }

    if constexpr (ConvolutionDimensions::is_always_depthwise)
    {
      auto [batches_per_convolution, batches_required, outputs_per_convolution] = dimensions.depthwise_split(N);
      return outputs_per_convolution;
    }
    else
    {
      return 1;
    }
}

template class BaseConv2dTripleProducer<FFT_Data>;
template void BaseConv2dTripleProducer<FFT_Data>::produce(convolution_dimensions, const Player&, MAC_Check<typename FFT_Data::T>&, const FHE_PK&, const Ciphertext&, EncCommitBase_<FFT_Data>&, DistDecrypt<FFT_Data>&, const typename FFT_Data::T&);
template void BaseConv2dTripleProducer<FFT_Data>::produce(depthwise_convolution_triple_dimensions, const Player&, MAC_Check<typename FFT_Data::T>&, const FHE_PK&, const Ciphertext&, EncCommitBase_<FFT_Data>&, DistDecrypt<FFT_Data>&, const typename FFT_Data::T&);
template class BaseConv2dTripleProducer<P2Data>;
template void BaseConv2dTripleProducer<P2Data>::produce(convolution_dimensions, const Player&, MAC_Check<typename P2Data::T>&, const FHE_PK&, const Ciphertext&, EncCommitBase_<P2Data>&, DistDecrypt<P2Data>&, const typename P2Data::T&);
template void BaseConv2dTripleProducer<P2Data>::produce(depthwise_convolution_triple_dimensions, const Player&, MAC_Check<typename P2Data::T>&, const FHE_PK&, const Ciphertext&, EncCommitBase_<P2Data>&, DistDecrypt<P2Data>&, const typename P2Data::T&);
template class SummingConv2dTripleProducer<FFT_Data>;
template int SummingConv2dTripleProducer<FFT_Data>::adapt(convolution_dimensions);
template int SummingConv2dTripleProducer<FFT_Data>::adapt(depthwise_convolution_triple_dimensions);
template class SummingConv2dTripleProducer<P2Data>;
template int SummingConv2dTripleProducer<P2Data>::adapt(convolution_dimensions);
template int SummingConv2dTripleProducer<P2Data>::adapt(depthwise_convolution_triple_dimensions);
template class PairwiseConv2dTripleProducer<FFT_Data>;
template int PairwiseConv2dTripleProducer<FFT_Data>::adapt(convolution_dimensions);
template int PairwiseConv2dTripleProducer<FFT_Data>::adapt(depthwise_convolution_triple_dimensions);
template class PairwiseConv2dTripleProducer<P2Data>;
template int PairwiseConv2dTripleProducer<P2Data>::adapt(convolution_dimensions);
template int PairwiseConv2dTripleProducer<P2Data>::adapt(depthwise_convolution_triple_dimensions);

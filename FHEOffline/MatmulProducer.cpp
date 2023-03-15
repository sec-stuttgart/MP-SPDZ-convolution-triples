#include "FHEOffline/MatmulProducer.h"
#include "Tools/Subroutines.h"
#include "FHEOffline/PairwiseGenerator.h"
#include "FHEOffline/PairwiseMachine.h"

template<class FD>
void BaseMatmulTripleProducer<FD>::clear_and_set_dimensions(matmul_dimensions dimensions)
{
  this->dimensions = dimensions;
  auto triple_count = this->adapt_for_dimensions(dimensions);
  this->clear();
  auto left_size = dimensions.left_size();
  auto right_size = dimensions.right_size();
  auto result_size = dimensions.result_size();

  this->presacrificing_shares = std::vector(triple_count, std::array{ std::vector<T>(left_size), std::vector<T>(right_size), std::vector<T>(right_size), std::vector<T>(result_size), std::vector<T>(result_size) });
  this->presacrificing_macs = std::vector(triple_count, std::array{ std::vector<T>(left_size), std::vector<T>(right_size), std::vector<T>(right_size), std::vector<T>(result_size), std::vector<T>(result_size) });
}

template<class FD>
void BaseMatmulTripleProducer<FD>::produce(matmul_dimensions dimensions, const Player& P, MAC_Check<T>& MC, const FHE_PK& pk, const Ciphertext& calpha, EncCommitBase_<FD>& EC, DistDecrypt<FD>& dd, const T& alphai)
{
  this->clear_and_set_dimensions(dimensions);
  this->run(P, pk, calpha, EC, dd, alphai);
  this->sacrifice(P, MC);
  MC.Check(P);
}

template<class FD>
int BaseMatmulTripleProducer<FD>::sacrifice(const Player& P, MAC_Check<T>& MC)
{
  check_field_size<T>();

  auto combine = [](auto const& share, auto const& mac) { Share<T> result; result.set_share(share); result.set_mac(mac); return result; };

  int triple_count = this->presacrificing_shares.size();
  CONV2D_ASSERT(this->presacrificing_macs.size() == static_cast<std::size_t>(triple_count));
  std::vector<T> ss(triple_count);

  std::size_t left_size = dimensions.left_size();
  std::size_t right_size = dimensions.right_size();
  std::size_t result_size = dimensions.result_size();

  std::vector<Share<T>> shared_rho(right_size * triple_count);
  std::vector<T> rho(right_size * triple_count);

  for (int j = 0; j < triple_count; ++j)
  {
    auto& [left, right, right_prime, result, result_prime] = this->presacrificing_shares[j];
    auto& [left_mac, right_mac, right_prime_mac, result_mac, result_prime_mac] = this->presacrificing_macs[j];
    CONV2D_ASSERT(left.size() == left_size);
    CONV2D_ASSERT(left.size() == left_mac.size());
    CONV2D_ASSERT(right.size() == right_size);
    CONV2D_ASSERT(right.size() == right_mac.size());
    CONV2D_ASSERT(result.size() == result_size);
    CONV2D_ASSERT(result.size() == result_mac.size());
    CONV2D_ASSERT(right.size() == right_prime.size());
    CONV2D_ASSERT(right_mac.size() == right_prime_mac.size());
    CONV2D_ASSERT(result.size() == result_prime.size());
    CONV2D_ASSERT(result_mac.size() == result_prime_mac.size());

    // sacrificing
    auto& s = ss[j];
    Create_Random(s, P);
    for (std::size_t i = 0; i < right_size; ++i)
    {
      auto b = combine(right[i], right_mac[i]);
      auto b_prime = combine(right_prime[i], right_prime_mac[i]);
      shared_rho[nDaccess({j, triple_count}, {static_cast<int>(i), static_cast<int>(right_size)})] = s * b - b_prime;
    }
  }
  MC.POpen(rho, shared_rho, P);

  std::vector<Share<T>> shared_sigma(result_size * triple_count);
  std::vector<T> sigma(result_size * triple_count);

  for (int j = 0; j < triple_count; ++j)
  {
    auto& [left, right, right_prime, result, result_prime] = this->presacrificing_shares[j];
    auto& [left_mac, right_mac, right_prime_mac, result_mac, result_prime_mac] = this->presacrificing_macs[j];
    auto& s = ss[j];
    for (int x = 0; x < dimensions.left_outer_dimension; ++x)
    {
      for (int y = 0; y < dimensions.right_outer_dimension; ++y)
      {
        auto result_index = nDaccess({x, dimensions.left_outer_dimension}, {y, dimensions.right_outer_dimension});
        auto& value = shared_sigma[nDaccess({j, triple_count}, {result_index, static_cast<int>(result_size)})];
        auto c = combine(result[result_index], result_mac[result_index]);
        auto c_prime = combine(result_prime[result_index], result_prime_mac[result_index]);

        value = s * c - c_prime;
        for (int k = 0; k < dimensions.inner_dimension; ++k)
        {
            auto left_index = nDaccess({x, dimensions.left_outer_dimension}, {k, dimensions.inner_dimension});
            auto right_index = nDaccess({k, dimensions.inner_dimension}, {y, dimensions.right_outer_dimension});
              
            auto a = combine(left[left_index], left_mac[left_index]);
            value -= rho[nDaccess({j, triple_count}, {right_index, static_cast<int>(right_size)})] * a;
        }
      }
    }
  }
  MC.POpen(sigma, shared_sigma, P);
  if (auto non_zero = std::find_if(begin(sigma), end(sigma), [](auto const&s) { return not s.is_zero(); }); non_zero != end(sigma))
  {
    // could use non_zero to debug where an element is non-zero
    throw Offline_Check_Error("Matmul sacrificing failed");
  }

  this->triples.reserve(this->triples.size() + triple_count);
  for (int j = 0; j < triple_count; ++j)
  {
    auto& [left, right, right_prime, result, result_prime] = this->presacrificing_shares[j];
    auto& [left_mac, right_mac, right_prime_mac, result_mac, result_prime_mac] = this->presacrificing_macs[j];
    auto& [a, b, c] = this->triples.emplace_back();

    // copy output
    using std::begin;
    using std::end;
    a.resize(left_size);
    std::transform(begin(left), end(left), begin(left_mac), begin(a), combine);

    b.resize(right_size);
    std::transform(begin(right), end(right), begin(right_mac), begin(b), combine);

    c.resize(result_size);
    std::transform(begin(result), end(result), begin(result_mac), begin(c), combine);
  }

  // write output
  // if (this->write_output)
  // {
  //   throw std::runtime_error("Writing to file not yet implemented");
  // }

  return 1;
}

#ifdef CONV2D_DIRECT_SUM
template<class FD>
void highgear_matmul(matmul_dimensions dimensions, Player const& P, const FHE_PK& pk, const Ciphertext& calpha, DistDecrypt<FD>& dd, FD const& FieldD, map<string, Timer>& timers, SparseSummingEncCommit<FD>& leftEC, SparseSummingEncCommit<FD>&, EncCommitBase_<FD>& summingEC, std::span<typename FD::T> presacrificing_left, std::span<typename FD::T> presacrificing_right, std::span<typename FD::T> presacrificing_right_prime, std::span<typename FD::T> presacrificing_result, std::span<typename FD::T> presacrificing_result_prime, std::span<typename FD::T> presacrificing_left_mac, std::span<typename FD::T> presacrificing_right_mac, std::span<typename FD::T> presacrificing_right_prime_mac, std::span<typename FD::T> presacrificing_result_mac, std::span<typename FD::T> presacrificing_result_prime_mac)
{
  const FHE_Params& params = pk.get_params();

  CONV2D_ASSERT(params.get_plaintext_modulus() == FieldD.get_prime());
  CONV2D_ASSERT(params.phi_m() == FieldD.phi_m());
  CONV2D_ASSERT(FieldD.num_slots() == FieldD.phi_m());
  auto N = FieldD.num_slots();

  auto& rightEC = dynamic_cast<SummingEncCommit<FD>&>(summingEC);
  auto& rightReshareEC = rightEC;
  auto& resultEC = rightEC;
#ifdef CONV2D_HIGHGEAR_GENERIC_EC
  auto& leftReshareEC = rightEC;
#else
  auto& leftReshareEC = leftEC;
#endif

  int products_per_multiplication = N / dimensions.inner_dimension;
  int multiplications_required = DIV_CEIL(2 * dimensions.right_outer_dimension, products_per_multiplication);

#ifdef VERBOSE_CONV2D
    std::cerr << "Starting summing (direct sum) matmul "
#ifdef CONV2D_HIGHGEAR_GENERIC_EC
      "(using the generic EC for resharing) "
#endif
#if CONV2D_SUMMING_CIPHERTEXTS
      "(summing over ciphertexts; at most " << CONV2D_MAX_IMAGE_DEPTH << " = " << CONV2D_EXTRA_SLACK << " ?= " << OnlineOptions::singleton.matrix_dimensions << " = " << params.get_matrix_dim() << " ciphertexts) "
#endif
     << dimensions.as_string() << "\n";
    std::cerr << "N=" << N << ", products_per_multiplication=" << products_per_multiplication << ", multiplications_required=" << multiplications_required << " * " << dimensions.left_outer_dimension << "\n";
#endif

  std::vector<Ciphertext> crights(multiplications_required, Ciphertext(params));

  timers["Right"].start();
  for (int n = 0; n < multiplications_required; ++n)
  {
    Plaintext_<FD> right(FieldD), macd_right(FieldD);
    Ciphertext cmacd_right(params);

    rightEC.next(right, crights[n]);
    mul(cmacd_right, calpha, crights[n], pk);
    dd.reshare(macd_right, cmacd_right, rightReshareEC);

    for (int m = 0; m < products_per_multiplication; ++m)
    {
      auto j = nDaccess({n, multiplications_required}, {m, products_per_multiplication});
      for (int k = 0; k < dimensions.inner_dimension; ++k)
      {
        auto source_index = nDaccess({m, products_per_multiplication}, {k, dimensions.inner_dimension});
        CONV2D_ASSERT(source_index < N);
        if (j < dimensions.right_outer_dimension)
        {
          auto dest_index = nDaccess({k, dimensions.inner_dimension}, {j, dimensions.right_outer_dimension});
          presacrificing_right[dest_index] = right.coeff(source_index);
          presacrificing_right_mac[dest_index] = macd_right.coeff(source_index);
        }
        else if (j < 2 * dimensions.right_outer_dimension)
        {
          auto dest_index = nDaccess({k, dimensions.inner_dimension}, {j - dimensions.right_outer_dimension, dimensions.right_outer_dimension});
          presacrificing_right_prime[dest_index] = right.coeff(source_index);
          presacrificing_right_prime_mac[dest_index] = macd_right.coeff(source_index);
        }
      }
    }
  }
  timers["Right"].stop();

  for (int i = 0; i < dimensions.left_outer_dimension; ++i)
  {
    timers["Left"].start();
    Plaintext_<FD> left(FieldD), macd_left(FieldD);
    Ciphertext cleft(params), cmacd_left(params);

    leftEC.next(left, cleft);
    mul(cmacd_left, calpha, cleft, pk);
    dd.reshare(macd_left, cmacd_left, leftReshareEC);

    for (int k = 0; k < dimensions.inner_dimension; ++k)
    {
      auto source_index = nDaccess(reverse_index(k, dimensions.inner_dimension));
      auto dest_index = nDaccess({i, dimensions.left_outer_dimension}, {k, dimensions.inner_dimension});

      presacrificing_left[dest_index] = left.coeff(source_index);
      presacrificing_left_mac[dest_index] = macd_left.coeff(source_index);
    }
    timers["Left"].stop();

    timers["Result"].start();
    for (int n = 0; n < multiplications_required; ++n)
    {
      Plaintext_<FD> result(FieldD), macd_result(FieldD);
      Ciphertext cresult(params), ccresult(params), cmacd_result(params);

      mul(cresult, cleft, crights[n], pk);
      Reshare(result, ccresult, cresult, true, P, resultEC, pk, dd);
      mul(cmacd_result, calpha, ccresult, pk);
      dd.reshare(macd_result, cmacd_result, resultEC);

      for (int m = 0; m < products_per_multiplication; ++m)
      {
        auto j = nDaccess({n, multiplications_required}, {m, products_per_multiplication});
        auto source_index = nDaccess({m, products_per_multiplication}, last_index(dimensions.inner_dimension));
        CONV2D_ASSERT(source_index < N);
        if (j < dimensions.right_outer_dimension)
        {
          auto dest_index = nDaccess({i, dimensions.left_outer_dimension}, {j, dimensions.right_outer_dimension});
          presacrificing_result[dest_index] = result.coeff(source_index);
          presacrificing_result_mac[dest_index] = macd_result.coeff(source_index);
        }
        else if (j < 2 * dimensions.right_outer_dimension)
        {
          auto dest_index = nDaccess({i, dimensions.left_outer_dimension}, {j - dimensions.right_outer_dimension, dimensions.right_outer_dimension});
          presacrificing_result_prime[dest_index] = result.coeff(source_index);
          presacrificing_result_prime_mac[dest_index] = macd_result.coeff(source_index);
        }
      }
    }
    timers["Result"].stop();
  }

#ifdef VERBOSE_CONV2D
  std::cerr << "Finished triple production for matmul.\n";
  std::cerr << "Called leftEC " << leftEC.number_of_calls() << " times (" << leftEC.remaining_capacity() << " unprocessed ciphertexts)\n";
  std::cerr << "Called rightEC = resultEC " << rightEC.number_of_calls() << " times (" << rightEC.remaining_capacity() << " unprocessed ciphertexts)\n";
  auto required_ciphertexts = multiplications_required * dimensions.left_outer_dimension;
  std::cerr << "Result utilization: " << required_ciphertexts * N << " slots used for " << 2 * dimensions.result_size() << " result elements (" << 200.0 * dimensions.result_size() / (required_ciphertexts * N) << "%)\n";
  std::cerr << std::flush;
#endif
}
#elif defined(CONV2D_BASIC_MATMUL)
template<class FD, typename Shares, typename Macs>
void highgear_matmul(matmul_dimensions dimensions, Player const& P, const FHE_PK& pk, const Ciphertext& calpha, DistDecrypt<FD>& dd, FD const& FieldD, map<string, Timer>& timers, EncCommitBase_<FD>& summingEC, indexed_view_of<Shares, 0> presacrificing_left, indexed_view_of<Shares, 1> presacrificing_right, indexed_view_of<Shares, 2> presacrificing_right_prime, indexed_view_of<Shares, 3> presacrificing_result, indexed_view_of<Shares, 4> presacrificing_result_prime, indexed_view_of<Macs, 0> presacrificing_left_mac, indexed_view_of<Macs, 1> presacrificing_right_mac, indexed_view_of<Macs, 2> presacrificing_right_prime_mac, indexed_view_of<Macs, 3> presacrificing_result_mac, indexed_view_of<Macs, 4> presacrificing_result_prime_mac)
{
  const FHE_Params& params = pk.get_params();

  CONV2D_ASSERT(params.get_plaintext_modulus() == FieldD.get_prime());
  CONV2D_ASSERT(params.phi_m() == FieldD.phi_m());
  CONV2D_ASSERT(FieldD.num_slots() == FieldD.phi_m());
  auto N = FieldD.num_slots();

  auto triple_count = N / dimensions.inner_dimension;
  assert(triple_count > 0);

  auto& EC = dynamic_cast<SummingEncCommit<FD>&>(summingEC);

#ifdef VERBOSE_CONV2D
    std::cerr << "Starting summing (basic) matmul "
     << dimensions.as_string() << "\n";
    std::cerr << "N=" << N << ", triple_count=" << triple_count << "\n";
#endif

  std::vector<Ciphertext> cleft(dimensions.left_outer_dimension, Ciphertext(params));
  std::vector<Ciphertext> cright(2 * dimensions.right_outer_dimension, Ciphertext(params));

  timers["Left"].start();
  for (int i = 0; i < dimensions.left_outer_dimension; ++i)
  {
    Plaintext_<FD> left(FieldD), macd_left(FieldD);
    Ciphertext cmacd_left(params);

    EC.next(left, cleft[i]);
    mul(cmacd_left, calpha, cleft[i], pk);
    dd.reshare(macd_left, cmacd_left, EC);

    for (int l = 0; l < triple_count; ++l)
    {
      for (int k = 0; k < dimensions.inner_dimension; ++k)
      {
        auto source_index = nDaccess({l, triple_count}, {k, dimensions.inner_dimension});
        auto dest_index = nDaccess({i, dimensions.left_outer_dimension}, {k, dimensions.inner_dimension});
        CONV2D_ASSERT(source_index < N);
        presacrificing_left[l][dest_index] = left.element(source_index);
        presacrificing_left_mac[l][dest_index] = macd_left.element(source_index);
      }
    }
  }
  timers["Left"].stop();

  timers["Right"].start();
  for (int j = 0; j < 2 * dimensions.right_outer_dimension; ++j)
  {
    Plaintext_<FD> right(FieldD), macd_right(FieldD);
    Ciphertext cmacd_right(params);

    EC.next(right, cright[j]);
    mul(cmacd_right, calpha, cright[j], pk);
    dd.reshare(macd_right, cmacd_right, EC);

    for (int l = 0; l < triple_count; ++l)
    {
      for (int k = 0; k < dimensions.inner_dimension; ++k)
      {
        auto source_index = nDaccess({l, triple_count}, {k, dimensions.inner_dimension});
        CONV2D_ASSERT(source_index < N);
        if (j < dimensions.right_outer_dimension)
        {
          auto dest_index = nDaccess({k, dimensions.inner_dimension}, {j, dimensions.right_outer_dimension});
          presacrificing_right[l][dest_index] = right.element(source_index);
          presacrificing_right_mac[l][dest_index] = macd_right.element(source_index);
        }
        else
        { 
          CONV2D_ASSERT(j < 2 * dimensions.right_outer_dimension);
          auto dest_index = nDaccess({k, dimensions.inner_dimension}, {j - dimensions.right_outer_dimension, dimensions.right_outer_dimension});
          presacrificing_right_prime[l][dest_index] = right.element(source_index);
          presacrificing_right_prime_mac[l][dest_index] = macd_right.element(source_index);
        }
      }
    }
  }
  timers["Right"].stop();

  timers["Result"].start();
  for (int i = 0; i < dimensions.left_outer_dimension; ++i)
  {
    for (int j = 0; j < 2 * dimensions.right_outer_dimension; ++j)
    {
      Plaintext_<FD> result(FieldD), macd_result(FieldD);
      Ciphertext cresult(params), ccresult(params), cmacd_result(params);

      mul(cresult, cleft[i], cright[j], pk);
      Reshare(result, ccresult, cresult, true, P, EC, pk, dd);
      mul(cmacd_result, calpha, ccresult, pk);
      dd.reshare(macd_result, cmacd_result, EC);

      for (int l = 0; l < triple_count; ++l)
      {
        for (int k = 0; k < dimensions.inner_dimension; ++k)
        {
          auto source_index = nDaccess({l, triple_count}, {k, dimensions.inner_dimension});
          CONV2D_ASSERT(source_index < N);
          if (j < dimensions.right_outer_dimension)
          {
            auto dest_index = nDaccess({i, dimensions.left_outer_dimension}, {j, dimensions.right_outer_dimension});
            presacrificing_result[l][dest_index] += result.element(source_index);
            presacrificing_result_mac[l][dest_index] += macd_result.element(source_index);
          }
          else
          { 
            CONV2D_ASSERT(j < 2 * dimensions.right_outer_dimension);
            auto dest_index = nDaccess({i, dimensions.left_outer_dimension}, {j - dimensions.right_outer_dimension, dimensions.right_outer_dimension});
            presacrificing_result_prime[l][dest_index] += result.element(source_index);
            presacrificing_result_prime_mac[l][dest_index] += macd_result.element(source_index);
          }
        }
      }
    }
  }
  timers["Result"].stop();

#ifdef VERBOSE_CONV2D
  std::cerr << "Finished triple production for matmul.\n";
  std::cerr << "Called summingEC " << EC.number_of_calls() << " times (" << EC.remaining_capacity() << " unprocessed ciphertexts)\n";
  auto required_ciphertexts = 2 * dimensions.left_outer_dimension * dimensions.right_outer_dimension;
  std::cerr << "Result utilization: " << required_ciphertexts * N << " slots used for " << 2 * triple_count * dimensions.result_size() << " result elements (" << 200.0 * triple_count * dimensions.result_size() / (required_ciphertexts * N) << "%) for " << triple_count << " triples\n";
  std::cerr << std::flush;
#endif
}
#else
template<class FD>
void highgear_matmul(matmul_dimensions dimensions, Player const& P, const FHE_PK& pk, const Ciphertext& calpha, DistDecrypt<FD>& dd, FD const& FieldD, map<string, Timer>& timers, SparseSummingEncCommit<FD>& leftEC, SparseSummingEncCommit<FD>& rightEC, EncCommitBase_<FD>& summingEC, std::span<typename FD::T> presacrificing_left, std::span<typename FD::T> presacrificing_right, std::span<typename FD::T> presacrificing_right_prime, std::span<typename FD::T> presacrificing_result, std::span<typename FD::T> presacrificing_result_prime, std::span<typename FD::T> presacrificing_left_mac, std::span<typename FD::T> presacrificing_right_mac, std::span<typename FD::T> presacrificing_right_prime_mac, std::span<typename FD::T> presacrificing_result_mac, std::span<typename FD::T> presacrificing_result_prime_mac)
{
  using T = typename FD::T;

  const FHE_Params& params = pk.get_params();

  CONV2D_ASSERT(params.get_plaintext_modulus() == FieldD.get_prime());
  CONV2D_ASSERT(params.phi_m() == FieldD.phi_m());
  CONV2D_ASSERT(FieldD.num_slots() == FieldD.phi_m());
  auto N = FieldD.num_slots();

  auto& resultEC = dynamic_cast<SummingEncCommit<FD>&>(summingEC);
#ifdef CONV2D_HIGHGEAR_GENERIC_EC
  auto& leftReshareEC = resultEC;
  auto& rightReshareEC = resultEC;
#else
  auto& leftReshareEC = leftEC;
  auto& rightReshareEC = rightEC;
#endif

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

  int products_per_multiplication = N / dimensions.left_outer_dimension;
  int multiplications_required = DIV_CEIL(2 * dimensions.right_outer_dimension, products_per_multiplication);

#ifdef VERBOSE_CONV2D
    std::cerr << "Starting summing matmul "
#ifdef CONV2D_HIGHGEAR_GENERIC_EC
      "(using the generic EC for resharing) "
#endif
#if CONV2D_SUMMING_CIPHERTEXTS
      "(summing over ciphertexts; at most " << CONV2D_MAX_IMAGE_DEPTH << " = " << CONV2D_EXTRA_SLACK << " ?= " << OnlineOptions::singleton.matrix_dimensions << " = " << params.get_matrix_dim() << " ciphertexts) "
#endif
     << dimensions.as_string() << "\n";
    std::cerr << "N=" << N << ", products_per_multiplication=" << products_per_multiplication << ", multiplications_required=" << multiplications_required << " * " << dimensions.inner_dimension << "\n";
#endif

#if CONV2D_SUMMING_CIPHERTEXTS
  assert(dimensions.inner_dimension <= params.get_matrix_dim());
  std::vector<Ciphertext> cresult_sums(multiplications_required, Ciphertext(params));
  for (auto& c : cresult_sums)
  {
    c.allocate();
    c.Scale();
  }
#endif

  Plaintext_<FD> result(FieldD), macd_result(FieldD);
  Ciphertext cresult(params), ccresult(params), cmacd_result(params);

  for (int k = 0; k < dimensions.inner_dimension; ++k)
  {
    timers["Left"].start();
    Plaintext_<FD> left(FieldD), macd_left(FieldD);
    Ciphertext cleft(params), cmacd_left(params);

    leftEC.next(left, cleft);
    mul(cmacd_left, calpha, cleft, pk);
    dd.reshare(macd_left, cmacd_left, leftReshareEC);

    for (int i = 0; i < dimensions.left_outer_dimension; ++i)
    {
      auto source_index = nDaccess({i, dimensions.left_outer_dimension});
      auto dest_index = nDaccess({i, dimensions.left_outer_dimension}, {k, dimensions.inner_dimension});

      presacrificing_left[dest_index] = left.coeff(source_index);
      presacrificing_left_mac[dest_index] = macd_left.coeff(source_index);
    }
    timers["Left"].stop();

    for (int n = 0; n < multiplications_required; ++n)
    {
      timers["Right"].start();
      Plaintext_<FD> right(FieldD), macd_right(FieldD);
      Ciphertext cright(params), cmacd_right(params);

      rightEC.next(right, cright);
      mul(cmacd_right, calpha, cright, pk);
      dd.reshare(macd_right, cmacd_right, rightReshareEC);

      for (int m = 0; m < products_per_multiplication; ++m)
      {
        auto j = nDaccess({n, multiplications_required}, {m, products_per_multiplication});
        auto source_index = nDaccess({m, products_per_multiplication}, first_index(dimensions.left_outer_dimension));
        CONV2D_ASSERT(source_index < N);
        if (j < dimensions.right_outer_dimension)
        {
          auto dest_index = nDaccess({k, dimensions.inner_dimension}, {j, dimensions.right_outer_dimension});
          presacrificing_right[dest_index] = right.coeff(source_index);
          presacrificing_right_mac[dest_index] = macd_right.coeff(source_index);
        }
        else if (j < 2 * dimensions.right_outer_dimension)
        {
          auto dest_index = nDaccess({k, dimensions.inner_dimension}, {j - dimensions.right_outer_dimension, dimensions.right_outer_dimension});
          presacrificing_right_prime[dest_index] = right.coeff(source_index);
          presacrificing_right_prime_mac[dest_index] = macd_right.coeff(source_index);
        }
      }
      timers["Right"].stop();
      
      timers["Result"].start();
      mul(cresult, cleft, cright, pk);
#if CONV2D_SUMMING_CIPHERTEXTS
      cresult_sums[n] += cresult;
      timers["Result"].stop();
    }
  }

  for (int n = 0; n < multiplications_required; ++n)
  {
    timers["Result"].start();
    Reshare(result, ccresult, cresult_sums[n], true, P, resultEC, pk, dd);
#else
      Reshare(result, ccresult, cresult, true, P, resultEC, pk, dd);
#endif
      mul(cmacd_result, calpha, ccresult, pk);
      dd.reshare(macd_result, cmacd_result, resultEC);

      for (int m = 0; m < products_per_multiplication; ++m)
      {
        auto j = nDaccess({n, multiplications_required}, {m, products_per_multiplication});
        for (int i = 0; i < dimensions.left_outer_dimension; ++i)
        {
          auto source_index = nDaccess({m, products_per_multiplication}, {i, dimensions.left_outer_dimension});
          CONV2D_ASSERT(source_index < N);
          if (j < dimensions.right_outer_dimension)
          {
            auto dest_index = nDaccess({i, dimensions.left_outer_dimension}, {j, dimensions.right_outer_dimension});
            write(presacrificing_result[dest_index], result.coeff(source_index));
            write(presacrificing_result_mac[dest_index], macd_result.coeff(source_index));
          }
          else if (j < 2 * dimensions.right_outer_dimension)
          {
            auto dest_index = nDaccess({i, dimensions.left_outer_dimension}, {j - dimensions.right_outer_dimension, dimensions.right_outer_dimension});
            write(presacrificing_result_prime[dest_index], result.coeff(source_index));
            write(presacrificing_result_prime_mac[dest_index], macd_result.coeff(source_index));
          }
        }
      }
      timers["Result"].stop();
    }
#if not CONV2D_SUMMING_CIPHERTEXTS
  }
#endif

#ifdef VERBOSE_CONV2D
  std::cerr << "Finished triple production for matmul.\n";
  std::cerr << "Called leftEC " << leftEC.number_of_calls() << " times (" << leftEC.remaining_capacity() << " unprocessed ciphertexts)\n";
  std::cerr << "Called rightEC " << rightEC.number_of_calls() << " times (" << rightEC.remaining_capacity() << " unprocessed ciphertexts)\n";
  std::cerr << "Called resultEC " << resultEC.number_of_calls() << " times (" << resultEC.remaining_capacity() << " unprocessed ciphertexts)\n";
  auto required_ciphertexts = multiplications_required * dimensions.inner_dimension;
  std::cerr << "Result utilization: " << required_ciphertexts * N << " slots used for " << 2 * dimensions.result_size() << " result elements (" << 200.0 * dimensions.result_size() / (required_ciphertexts * N) << "%)\n";
  std::cerr << std::flush;
#endif
}
#endif

template<class FD>
void SummingMatmulTripleProducer<FD>::run(const Player& P, const FHE_PK& pk, const Ciphertext& calpha, EncCommitBase_<FD>& summingEC, DistDecrypt<FD>& dd, const T& /*alphai*/)
{
#ifdef CONV2D_BASIC_MATMUL
  highgear_matmul(this->dimensions, P, pk, calpha, dd, this->FieldD, this->timers, summingEC, this->template get_presacrificing_shares<0>(), this->template get_presacrificing_shares<1>(), this->template get_presacrificing_shares<2>(), this->template get_presacrificing_shares<3>(), this->template get_presacrificing_shares<4>(), this->template get_presacrificing_macs<0>(), this->template get_presacrificing_macs<1>(), this->template get_presacrificing_macs<2>(), this->template get_presacrificing_macs<3>(), this->template get_presacrificing_macs<4>());
#else
  auto& [leftEC, rightEC] = this->EC_ptrs;
  CONV2D_ASSERT(leftEC != this->template end<0>());
  CONV2D_ASSERT(rightEC != this->template end<1>());
  highgear_matmul(this->dimensions, P, pk, calpha, dd, this->FieldD, this->timers, leftEC->second, rightEC->second, summingEC, this->presacrificing_shares[0][0], this->presacrificing_shares[0][1], this->presacrificing_shares[0][2], this->presacrificing_shares[0][3], this->presacrificing_shares[0][4], this->presacrificing_macs[0][0], this->presacrificing_macs[0][1], this->presacrificing_macs[0][2], this->presacrificing_macs[0][3], this->presacrificing_macs[0][4]);
#endif
}

template<class FD>
int SummingMatmulTripleProducer<FD>::adapt_for_dimensions(matmul_dimensions dimensions)
{
    CONV2D_ASSERT(this->FieldD.phi_m() == this->FieldD.num_slots());
    int N = this->FieldD.num_slots();
#ifndef CONV2D_BASIC_MATMUL
    this->template try_emplace<0>(dimensions.left_sparcity(N), this->P, this->pk, this->FieldD, this->timers, this->machine, 0, false);
  #ifndef CONV2D_DIRECT_SUM
    this->template try_emplace<1>(dimensions.right_sparcity(N), this->P, this->pk, this->FieldD, this->timers, this->machine, 0, false);
  #endif
    return 1;

#else
    return N / dimensions.inner_dimension;
#endif
}

#ifdef CONV2D_DIRECT_SUM
template<typename FD, typename RightEC>
void lowgear_matmul(matmul_dimensions dimensions, std::span<Multiplier<std::type_identity_t<FD>>*> multipliers, Plaintext_<FD> const& alphai, const FHE_Params& params, FD const& FieldD, map<string, Timer>& timers, ReusableSparseMultiEncCommit<FD>& leftEC, RightEC& rightEC, std::span<typename FD::T> presacrificing_left, std::span<typename FD::T> presacrificing_right, std::span<typename FD::T> presacrificing_right_prime, std::span<typename FD::T> presacrificing_result, std::span<typename FD::T> presacrificing_result_prime, std::span<typename FD::T> presacrificing_left_mac, std::span<typename FD::T> presacrificing_right_mac, std::span<typename FD::T> presacrificing_right_prime_mac, std::span<typename FD::T> presacrificing_result_mac, std::span<typename FD::T> presacrificing_result_prime_mac)
{
  CONV2D_ASSERT(params.get_plaintext_modulus() == FieldD.get_prime());
  CONV2D_ASSERT(params.phi_m() == FieldD.phi_m());
  CONV2D_ASSERT(FieldD.num_slots() == FieldD.phi_m());

  auto N = FieldD.num_slots();

  int products_per_multiplication = N / dimensions.inner_dimension;
  int multiplications_required = DIV_CEIL(2 * dimensions.right_outer_dimension, products_per_multiplication);
#ifdef VERBOSE_CONV2D
    std::cerr << "Starting pairwise (direct sum) matmul "
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    "(with right ciphertexts instead of left ciphertexts) "
#else
    "(with left ciphertexts instead of right ciphertexts) "
#endif
     << dimensions.as_string() << "\n";
    std::cerr << "N=" << N << ", products_per_multiplication=" << products_per_multiplication << ", multiplications_required=" << multiplications_required << " * " << dimensions.left_outer_dimension << "\n";
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

  std::vector<Plaintext_<FD>> rights(multiplications_required, Plaintext_<FD>(FieldD));
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
  std::vector<std::vector<Ciphertext>> cright_of_others(multiplications_required);
#else
  std::vector<Rq_Element> rights_mod_q(multiplications_required, Rq_Element(params, evaluation, evaluation));
#endif

  timers["Right"].start();
  for (int n = 0; n < multiplications_required; ++n)
  {
    auto& right = rights[n];
    Plaintext_<FD> macd_right(FieldD);
#if CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    auto right_mod_q = Rq_Element(params, evaluation, evaluation);
    rightEC.next(right, cright_of_others[n]);
#else
    auto& right_mod_q = rights_mod_q[n];
    rightEC.get_proof().randomize(G, right);
#endif
    right_mod_q.from(right.get_iterator());
    authenticate(macd_right, right, right_mod_q);

    for (int m = 0; m < products_per_multiplication; ++m)
    {
      auto j = nDaccess({n, multiplications_required}, {m, products_per_multiplication});
      for (int k = 0; k < dimensions.inner_dimension; ++k)
      {
        auto source_index = nDaccess({m, products_per_multiplication}, {k, dimensions.inner_dimension});
        CONV2D_ASSERT(source_index < N);
        if (j < dimensions.right_outer_dimension)
        {
          auto dest_index = nDaccess({k, dimensions.inner_dimension}, {j, dimensions.right_outer_dimension});
          presacrificing_right[dest_index] = right.coeff(source_index);
          presacrificing_right_mac[dest_index] = macd_right.coeff(source_index);
        }
        else if (j < 2 * dimensions.right_outer_dimension)
        {
          auto dest_index = nDaccess({k, dimensions.inner_dimension}, {j - dimensions.right_outer_dimension, dimensions.right_outer_dimension});
          presacrificing_right_prime[dest_index] = right.coeff(source_index);
          presacrificing_right_prime_mac[dest_index] = macd_right.coeff(source_index);
        }
      }
    }
  }
  timers["Right"].stop();

  for (int i = 0; i < dimensions.left_outer_dimension; ++i)
  {
    timers["Left"].start();
    Plaintext_<FD> left(FieldD), macd_left(FieldD);
    auto left_mod_q = Rq_Element(params, evaluation, evaluation);
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    leftEC.get_proof().randomize(G, left);
#else
    std::vector<Ciphertext> cleft_of_others;
    leftEC.next(left, cleft_of_others);
#endif

    left_mod_q.from(left.get_iterator());
    authenticate(macd_left, left, left_mod_q);

    for (int k = 0; k < dimensions.inner_dimension; ++k)
    {
      auto source_index = nDaccess(reverse_index(k, dimensions.inner_dimension));
      auto dest_index = nDaccess({i, dimensions.left_outer_dimension}, {k, dimensions.inner_dimension});

      presacrificing_left[dest_index] = left.coeff(source_index);
      presacrificing_left_mac[dest_index] = macd_left.coeff(source_index);
    }
    timers["Left"].stop();

    timers["Result"].start();
    for (int n = 0; n < multiplications_required; ++n)
    {
      Plaintext_<FD> result(FieldD), macd_result(FieldD);
      auto result_mod_q = Rq_Element(params, evaluation, evaluation);

#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
      multiply_and_add(result, left, left_mod_q, rights[n], cright_of_others[n]);
#else
      multiply_and_add(result, rights[n], rights_mod_q[n], left, cleft_of_others);
#endif
      result_mod_q.from(result.get_iterator());
      authenticate(macd_result, result, result_mod_q);

      for (int m = 0; m < products_per_multiplication; ++m)
      {
        auto j = nDaccess({n, multiplications_required}, {m, products_per_multiplication});
        auto source_index = nDaccess({m, products_per_multiplication}, last_index(dimensions.inner_dimension));
        CONV2D_ASSERT(source_index < N);
        if (j < dimensions.right_outer_dimension)
        {
          auto dest_index = nDaccess({i, dimensions.left_outer_dimension}, {j, dimensions.right_outer_dimension});
          presacrificing_result[dest_index] = result.coeff(source_index);
          presacrificing_result_mac[dest_index] = macd_result.coeff(source_index);
        }
        else if (j < 2 * dimensions.right_outer_dimension)
        {
          auto dest_index = nDaccess({i, dimensions.left_outer_dimension}, {j - dimensions.right_outer_dimension, dimensions.right_outer_dimension});
          presacrificing_result_prime[dest_index] = result.coeff(source_index);
          presacrificing_result_prime_mac[dest_index] = macd_result.coeff(source_index);
        }
      }
    }
    timers["Result"].stop();
  }

#ifdef VERBOSE_CONV2D
  std::cerr << "Finished triple production for matmul.\n";
  std::cerr << "Called leftEC " << leftEC.number_of_calls() << " times (" << leftEC.remaining_capacity() << " unprocessed ciphertexts)\n";
  std::cerr << "Called rightEC " << rightEC.number_of_calls() << " times (" << rightEC.remaining_capacity() << " unprocessed ciphertexts)\n";
  auto required_ciphertexts = multiplications_required * dimensions.left_outer_dimension;
  std::cerr << "Result utilization: " << required_ciphertexts * N << " slots used for " << 2 * dimensions.result_size() << " result elements (" << 200.0 * dimensions.result_size() / (required_ciphertexts * N) << "%)\n";
  std::cerr << std::flush;
#endif
}
#elif defined(CONV2D_BASIC_MATMUL)
template<typename FD, typename Shares, typename Macs>
void lowgear_matmul(matmul_dimensions dimensions, std::span<Multiplier<std::type_identity_t<FD>>*> multipliers, Plaintext_<FD> const& alphai, const FHE_Params& params, FD const& FieldD, map<string, Timer>& timers, ReusableMultiEncCommit<FD>& EC, indexed_view_of<Shares, 0> presacrificing_left, indexed_view_of<Shares, 1> presacrificing_right, indexed_view_of<Shares, 2> presacrificing_right_prime, indexed_view_of<Shares, 3> presacrificing_result, indexed_view_of<Shares, 4> presacrificing_result_prime, indexed_view_of<Macs, 0> presacrificing_left_mac, indexed_view_of<Macs, 1> presacrificing_right_mac, indexed_view_of<Macs, 2> presacrificing_right_prime_mac, indexed_view_of<Macs, 3> presacrificing_result_mac, indexed_view_of<Macs, 4> presacrificing_result_prime_mac)
{
  CONV2D_ASSERT(params.get_plaintext_modulus() == FieldD.get_prime());
  CONV2D_ASSERT(params.phi_m() == FieldD.phi_m());
  CONV2D_ASSERT(FieldD.num_slots() == FieldD.phi_m());

  auto N = FieldD.num_slots();

  int triple_count = N / dimensions.inner_dimension;
  assert(triple_count > 0);
#ifdef VERBOSE_CONV2D
    std::cerr << "Starting pairwise (basic) matmul "
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    "(with right ciphertexts instead of left ciphertexts) "
#else
    "(with left ciphertexts instead of right ciphertexts) "
#endif
     << dimensions.as_string() << "\n";
    std::cerr << "N=" << N << ", triple_count=" << triple_count << "\n";
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

  std::vector<Plaintext_<FD>> lefts(dimensions.left_outer_dimension, Plaintext_<FD>(FieldD));
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
  std::vector<Rq_Element> lefts_mod_q(dimensions.left_outer_dimension, Rq_Element(params, evaluation, evaluation));
#else
  std::vector<std::vector<Ciphertext>> clefts_of_others(dimensions.left_outer_dimension);
#endif
  timers["Left"].start();
  for (int i = 0; i < dimensions.left_outer_dimension; ++i)
  {
    Plaintext_<FD> macd_left(FieldD);
    auto& left = lefts[i];
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    auto& left_mod_q = lefts_mod_q[i];
    EC.get_proof().randomize(G, left);
#else
    EC.next(left, clefts_of_others[i]);
    auto left_mod_q = Rq_Element(params, evaluation, evaluation);
#endif
    left_mod_q.from(left.get_iterator());
    authenticate(macd_left, left, left_mod_q);

    for (int l = 0; l < triple_count; ++l)
    {
      for (int k = 0; k < dimensions.inner_dimension; ++k)
      {
        auto source_index = nDaccess({l, triple_count}, {k, dimensions.inner_dimension});
        auto dest_index = nDaccess({i, dimensions.left_outer_dimension}, {k, dimensions.inner_dimension});
        CONV2D_ASSERT(source_index < N);
        presacrificing_left[l][dest_index] = left.element(source_index);
        presacrificing_left_mac[l][dest_index] = macd_left.element(source_index);
      }
    }
  }
  timers["Left"].stop();

  std::vector<Plaintext_<FD>> rights(2 * dimensions.right_outer_dimension, Plaintext_<FD>(FieldD));
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
  std::vector<std::vector<Ciphertext>> crights_of_others(2 * dimensions.right_outer_dimension);
#else
  std::vector<Rq_Element> rights_mod_q(2 * dimensions.right_outer_dimension, Rq_Element(params, evaluation, evaluation));
#endif

  timers["Right"].start();
  for (int j = 0; j < 2 * dimensions.right_outer_dimension; ++j)
  {
    Plaintext_<FD> macd_right(FieldD);
    auto& right = rights[j];
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    EC.next(right, crights_of_others[j]);
    auto right_mod_q = Rq_Element(params, evaluation, evaluation);
#else
    auto& right_mod_q = rights_mod_q[j];
    EC.get_proof().randomize(G, right);
#endif
    right_mod_q.from(right.get_iterator());
    authenticate(macd_right, right, right_mod_q);

    for (int l = 0; l < triple_count; ++l)
    {
      for (int k = 0; k < dimensions.inner_dimension; ++k)
      {
        auto source_index = nDaccess({l, triple_count}, {k, dimensions.inner_dimension});
        CONV2D_ASSERT(source_index < N);
        if (j < dimensions.right_outer_dimension)
        {
          auto dest_index = nDaccess({k, dimensions.inner_dimension}, {j, dimensions.right_outer_dimension});
          presacrificing_right[l][dest_index] = right.element(source_index);
          presacrificing_right_mac[l][dest_index] = macd_right.element(source_index);
        }
        else
        { 
          CONV2D_ASSERT(j < 2 * dimensions.right_outer_dimension);
          auto dest_index = nDaccess({k, dimensions.inner_dimension}, {j - dimensions.right_outer_dimension, dimensions.right_outer_dimension});
          presacrificing_right_prime[l][dest_index] = right.element(source_index);
          presacrificing_right_prime_mac[l][dest_index] = macd_right.element(source_index);
        }
      }
    }
  }
  timers["Right"].stop();

  timers["Result"].start();
  for (int i = 0; i < dimensions.left_outer_dimension; ++i)
  {
    for (int j = 0; j < 2 * dimensions.right_outer_dimension; ++j)
    {
      Plaintext_<FD> result(FieldD), macd_result(FieldD);
      auto result_mod_q = Rq_Element(params, evaluation, evaluation);

      auto& left = lefts[i];
      auto& right = rights[j];
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
      auto& left_mod_q = lefts_mod_q[i];
      auto& cright_of_others = crights_of_others[j];
      multiply_and_add(result, left, left_mod_q, right, cright_of_others);
#else
      auto& cleft_of_others = clefts_of_others[i];
      auto& right_mod_q = rights_mod_q[j];
      multiply_and_add(result, right, right_mod_q, left, cleft_of_others);
#endif
      result_mod_q.from(result.get_iterator());
      authenticate(macd_result, result, result_mod_q);

      for (int l = 0; l < triple_count; ++l)
      {
        for (int k = 0; k < dimensions.inner_dimension; ++k)
        {
          auto source_index = nDaccess({l, triple_count}, {k, dimensions.inner_dimension});
          CONV2D_ASSERT(source_index < N);
          if (j < dimensions.right_outer_dimension)
          {
            auto dest_index = nDaccess({i, dimensions.left_outer_dimension}, {j, dimensions.right_outer_dimension});
            presacrificing_result[l][dest_index] += result.element(source_index);
            presacrificing_result_mac[l][dest_index] += macd_result.element(source_index);
          }
          else
          { 
            CONV2D_ASSERT(j < 2 * dimensions.right_outer_dimension);
            auto dest_index = nDaccess({i, dimensions.left_outer_dimension}, {j - dimensions.right_outer_dimension, dimensions.right_outer_dimension});
            presacrificing_result_prime[l][dest_index] += result.element(source_index);
            presacrificing_result_prime_mac[l][dest_index] += macd_result.element(source_index);
          }
        }
      }
    }
  }
  timers["Result"].stop();

#ifdef VERBOSE_CONV2D
  std::cerr << "Finished triple production for matmul.\n";
  std::cerr << "Called EC " << EC.number_of_calls() << " times (" << EC.remaining_capacity() << " unprocessed ciphertexts)\n";
  auto required_ciphertexts = 2 * dimensions.left_outer_dimension * dimensions.right_outer_dimension;
  std::cerr << "Result utilization: " << required_ciphertexts * N << " slots used for " << 2 * triple_count * dimensions.result_size() << " result elements (" << 200.0 * triple_count * dimensions.result_size() / (required_ciphertexts * N) << "%) for " << triple_count << " triples\n";
  std::cerr << std::flush;
#endif
}
#else
template<typename FD, typename RightEC>
void lowgear_matmul(matmul_dimensions dimensions, std::span<Multiplier<std::type_identity_t<FD>>*> multipliers, Plaintext_<FD> const& alphai, const FHE_Params& params, FD const& FieldD, map<string, Timer>& timers, ReusableSparseMultiEncCommit<FD>& leftEC, RightEC& rightEC, std::span<typename FD::T> presacrificing_left, std::span<typename FD::T> presacrificing_right, std::span<typename FD::T> presacrificing_right_prime, std::span<typename FD::T> presacrificing_result, std::span<typename FD::T> presacrificing_result_prime, std::span<typename FD::T> presacrificing_left_mac, std::span<typename FD::T> presacrificing_right_mac, std::span<typename FD::T> presacrificing_right_prime_mac, std::span<typename FD::T> presacrificing_result_mac, std::span<typename FD::T> presacrificing_result_prime_mac)
{
  CONV2D_ASSERT(params.get_plaintext_modulus() == FieldD.get_prime());
  CONV2D_ASSERT(params.phi_m() == FieldD.phi_m());
  CONV2D_ASSERT(FieldD.num_slots() == FieldD.phi_m());

  auto N = FieldD.num_slots();

  int products_per_multiplication = N / dimensions.left_outer_dimension;
  int multiplications_required = DIV_CEIL(2 * dimensions.right_outer_dimension, products_per_multiplication);
#ifdef VERBOSE_CONV2D
    std::cerr << "Starting pairwise matmul "
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    "(with right ciphertexts instead of left ciphertexts) "
#else
    "(with left ciphertexts instead of right ciphertexts) "
#endif
     << dimensions.as_string() << "\n";
    std::cerr << "N=" << N << ", products_per_multiplication=" << products_per_multiplication << ", multiplications_required=" << multiplications_required << " * " << dimensions.inner_dimension << "\n";
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

  std::vector<Plaintext_<FD>> result_sums(multiplications_required, Plaintext_<FD>(FieldD));

  for (int k = 0; k < dimensions.inner_dimension; ++k)
  {
    Plaintext_<FD> left(FieldD), macd_left(FieldD);
#ifndef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    std::vector<Ciphertext> cleft_of_others;
#endif
    auto left_mod_q = Rq_Element(params, evaluation, evaluation);
    
    timers["Left"].start();
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
    leftEC.get_proof().randomize(G, left);
#else
    leftEC.next(left, cleft_of_others);
#endif
    left_mod_q.from(left.get_iterator());
    authenticate(macd_left, left, left_mod_q);

    for (int i = 0; i < dimensions.left_outer_dimension; ++i)
    {
      auto source_index = nDaccess({i, dimensions.left_outer_dimension});
      auto dest_index = nDaccess({i, dimensions.left_outer_dimension}, {k, dimensions.inner_dimension});

      presacrificing_left[dest_index] = left.coeff(source_index);
      presacrificing_left_mac[dest_index] = macd_left.coeff(source_index);
    }
    timers["Left"].stop();

    for (int n = 0; n < multiplications_required; ++n)
    {
      Plaintext_<FD> right(FieldD), macd_right(FieldD);
      auto right_mod_q = Rq_Element(params, evaluation, evaluation);

      timers["Right"].start();
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
      std::vector<Ciphertext> cright_of_others;
      rightEC.next(right, cright_of_others);
#else
      rightEC.get_proof().randomize(G, right);
#endif
      right_mod_q.from(right.get_iterator());
      authenticate(macd_right, right, right_mod_q);

      for (int m = 0; m < products_per_multiplication; ++m)
      {
        auto j = nDaccess({n, multiplications_required}, {m, products_per_multiplication});
        auto source_index = nDaccess({m, products_per_multiplication}, first_index(dimensions.left_outer_dimension));
        CONV2D_ASSERT(source_index < N);
        if (j < dimensions.right_outer_dimension)
        {
          auto dest_index = nDaccess({k, dimensions.inner_dimension}, {j, dimensions.right_outer_dimension});
          presacrificing_right[dest_index] = right.coeff(source_index);
          presacrificing_right_mac[dest_index] = macd_right.coeff(source_index);
        }
        else if (j < 2 * dimensions.right_outer_dimension)
        {
          auto dest_index = nDaccess({k, dimensions.inner_dimension}, {j - dimensions.right_outer_dimension, dimensions.right_outer_dimension});
          presacrificing_right_prime[dest_index] = right.coeff(source_index);
          presacrificing_right_prime_mac[dest_index] = macd_right.coeff(source_index);
        }
      }
      timers["Right"].stop();

      timers["Result"].start();
      Plaintext_<FD> result(FieldD);
#ifdef CONV2D_LOWGEAR_FILTER_CIPHERTEXTS
      multiply_and_add(result, left, left_mod_q, right, cright_of_others);
#else
      multiply_and_add(result, right, right_mod_q, left, cleft_of_others);
#endif
      result_sums[n] += result;
      timers["Result"].stop();
    }
  }

  timers["Result"].start();
  for (int n = 0; n < multiplications_required; ++n)
  {
    auto& result = result_sums[n];
    Plaintext_<FD> macd_result(FieldD);
    Rq_Element result_mod_q(params, evaluation, evaluation);

    result_mod_q.from(result.get_iterator());
    authenticate(macd_result, result, result_mod_q);

    for (int m = 0; m < products_per_multiplication; ++m)
    {
      auto j = nDaccess({n, multiplications_required}, {m, products_per_multiplication});
      for (int i = 0; i < dimensions.left_outer_dimension; ++i)
      {
        auto source_index = nDaccess({m, products_per_multiplication}, {i, dimensions.left_outer_dimension});
        CONV2D_ASSERT(source_index < N);
        if (j < dimensions.right_outer_dimension)
        {
          auto dest_index = nDaccess({i, dimensions.left_outer_dimension}, {j, dimensions.right_outer_dimension});
          presacrificing_result[dest_index] = result.coeff(source_index);
          presacrificing_result_mac[dest_index] = macd_result.coeff(source_index);
        }
        else if (j < 2 * dimensions.right_outer_dimension)
        {
          auto dest_index = nDaccess({i, dimensions.left_outer_dimension}, {j - dimensions.right_outer_dimension, dimensions.right_outer_dimension});
          presacrificing_result_prime[dest_index] = result.coeff(source_index);
          presacrificing_result_prime_mac[dest_index] = macd_result.coeff(source_index);
        }
      }
    }
  }
  timers["Result"].stop();

#ifdef VERBOSE_CONV2D
  std::cerr << "Finished triple production for matmul.\n";
  std::cerr << "Called leftEC " << leftEC.number_of_calls() << " times (" << leftEC.remaining_capacity() << " unprocessed ciphertexts)\n";
  std::cerr << "Called rightEC " << rightEC.number_of_calls() << " times (" << rightEC.remaining_capacity() << " unprocessed ciphertexts)\n";
  auto required_ciphertexts = multiplications_required * dimensions.left_outer_dimension;
  std::cerr << "Result utilization: " << required_ciphertexts * N << " slots used for " << 2 * dimensions.result_size() << " result elements (" << 200.0 * dimensions.result_size() / (required_ciphertexts * N) << "%)\n";
  std::cerr << std::flush;
#endif
}
#endif

template<class FD>
void PairwiseMatmulTripleProducer<FD>::run(const Player&, const FHE_PK&, const Ciphertext&, EncCommitBase_<FD>&, DistDecrypt<FD>&, const T&)
{
  auto& setup = this->generator.machine.template setup<FD>();
  auto& multipliers = this->generator.multipliers;
  auto& [leftEC, rightEC] = this->EC_ptrs;
  CONV2D_ASSERT(leftEC != this->template end<0>());
  CONV2D_ASSERT(rightEC != this->template end<1>());
#ifdef CONV2D_BASIC_MATMUL
  lowgear_matmul(this->dimensions, multipliers, setup.alpha, setup.params, setup.FieldD, this->timers, rightEC->second, this->template get_presacrificing_shares<0>(), this->template get_presacrificing_shares<1>(), this->template get_presacrificing_shares<2>(), this->template get_presacrificing_shares<3>(), this->template get_presacrificing_shares<4>(), this->template get_presacrificing_macs<0>(), this->template get_presacrificing_macs<1>(), this->template get_presacrificing_macs<2>(), this->template get_presacrificing_macs<3>(), this->template get_presacrificing_macs<4>());
#else
  lowgear_matmul(this->dimensions, multipliers, setup.alpha, setup.params, setup.FieldD, this->timers, leftEC->second, rightEC->second, this->presacrificing_shares[0][0], this->presacrificing_shares[0][1], this->presacrificing_shares[0][2], this->presacrificing_shares[0][3], this->presacrificing_shares[0][4], this->presacrificing_macs[0][0], this->presacrificing_macs[0][1], this->presacrificing_macs[0][2], this->presacrificing_macs[0][3], this->presacrificing_macs[0][4]);
#endif
}

template<class FD>
int PairwiseMatmulTripleProducer<FD>::adapt_for_dimensions(matmul_dimensions dimensions)
{
    auto& generator = this->generator;
    auto& FieldD = generator.machine.template setup<FD>().FieldD;
    CONV2D_ASSERT(FieldD.phi_m() == FieldD.num_slots());
    int N = FieldD.num_slots();
#ifndef CONV2D_BASIC_MATMUL
    this->template try_emplace<0>(dimensions.left_sparcity(N), generator.P, generator.machine.other_pks, FieldD, this->timers, generator.machine, generator, false);
  #ifndef CONV2D_DIRECT_SUM
    this->template try_emplace<1>(dimensions.right_sparcity(N), generator.P, generator.machine.other_pks, FieldD, this->timers, generator.machine, generator, false);
  #endif
    
    return 1;
#else
    return N / dimensions.inner_dimension;
#endif
}

template class BaseMatmulTripleProducer<FFT_Data>;
template class BaseMatmulTripleProducer<P2Data>;
template class SummingMatmulTripleProducer<FFT_Data>;
template class SummingMatmulTripleProducer<P2Data>;
template class PairwiseMatmulTripleProducer<FFT_Data>;
template class PairwiseMatmulTripleProducer<P2Data>;

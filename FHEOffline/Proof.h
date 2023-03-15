#ifndef _Proof
#define _Proof

#include <math.h>
#include <vector>
#include <numeric>
#include <functional>
using namespace std;

#include "Math/bigint.h"
#include "FHE/Ciphertext.h"
#include "FHE/AddableVector.h"
#include "Protocols/CowGearOptions.h"

#include "config.h"
#include "FHE/P2Data.h"

enum SlackType
{
  NONINTERACTIVE_SPDZ1_SLACK = -1,
  INTERACTIVE_SPDZ1_SLACK = -2,
  COVERT_SPDZ2_SLACK = -3,
  ACTIVE_SPDZ2_SLACK = -4,
};

template<typename Field>
auto check_decoding_mod2(Field const&)
{
  if constexpr (std::is_same_v<Field, P2Data>)
  {
    return std::true_type{};
  }
  else
  {
    return std::false_type{};
  }
}

class Proof
{
  unsigned int sec;

protected:
  bool diagonal;
private:

  Proof();   // Private to avoid default 

  public:

  typedef AddableVector< Int_Random_Coins > Randomness;
  typedef typename FFT_Data::poly_type bound_type;

  class Preimages
  {
    typedef Int_Random_Coins::value_type::value_type r_type;

    bound_type m_tmp;
    AddableVector<r_type> r_tmp;

  public:
    Preimages(int size, const FHE_PK& pk, const bigint& p, int n_players);
    AddableMatrix<bound_type> m;
    Randomness r;
    void add(octetStream& os);
    void pack(octetStream& os);
    void unpack(octetStream& os);
    void check_sizes();
    size_t report_size(ReportType type) { return m.report_size(type) + r.report_size(type); }
  };

  bigint tau,rho;

  unsigned int phim;
  int B_plain_length, B_rand_length;
  bigint plain_check, rand_check;
  unsigned int V;
  unsigned int U;

  const FHE_PK* pk;

  int n_proofs;

  vector<int> e;
  vector<vector<int>> W;
  bool top_gear;

  static double dist;

  protected:
    typedef AddableVector<bound_type> T;
    typedef AddableMatrix<typename Int_Random_Coins::rand_type> X;

    Proof(int sc, const bigint& Tau, const bigint& Rho, const FHE_PK& pk,
            int n_proofs = 1, bool diagonal = false) :
              diagonal(diagonal),
              B_plain_length(0), B_rand_length(0), pk(&pk), n_proofs(n_proofs)
    { sec=sc;
      assert(sec > 0);
      tau=Tau; rho=Rho;

      phim=(pk.get_params()).phi_m();

      top_gear = use_top_gear(pk);
      if (top_gear)
        {
          V = ceil((sec + 2) / log2(2 * phim + 1));
          U = 2 * V;
#ifdef VERBOSE
          cerr << "Using " << U << " ciphertexts per proof" << endl;
#endif
        }
      else
        {
          U = sec;
          V = 2 * sec - 1;
        }
    }

  Proof(int sec, const FHE_PK& pk, int n_proofs = 1, bool diagonal = false) :
      Proof(sec, pk.p() / 2,
          pk.get_params().get_DG().get_NewHopeB(), pk,
          n_proofs, diagonal) {}

  virtual ~Proof() {}

  public:
  static bigint slack(int slack, int sec, int phim);

  bool use_top_gear(const FHE_PK& pk)
  {
    return CowGearOptions::singleton.top_gear() and pk.p() > 2 and
        not diagonal;
  }

  bool get_diagonal() const
  {
    return diagonal;
  }

  static int n_ciphertext_per_proof(int sec, const FHE_PK& pk, bool diagonal =
      false)
  {
    return Proof(sec, pk, 1, diagonal).U;
  }

  void set_challenge(const octetStream& ciphertexts);
  void set_challenge(PRNG& G);
  void generate_challenge(const Player& P);

  bool check_bounds(T& z, X& t, int i) const;

  template<class T, class U>
  void apply_challenge(int i, T& output, const U& input, const FHE_PK& pk) const
  {
    if (top_gear)
      {
        for (unsigned j = 0; j < this->U; j++)
          if (W[i][j] >= 0)
            output += input[j].mul_by_X_i(W[i][j], pk);
      }
    else
      for (unsigned k = 0; k < sec; k++)
        {
          unsigned j = (i + 1) - (k + 1);
          if (j < sec && e.at(j))
            output += input.at(j);
        }
  }

  template<typename Value, bool CheckDecodingMod2 = false>
  [[nodiscard]] bool check_sparse(Value const& value, std::bool_constant<CheckDecodingMod2> = {}) const
  {
    if (diagonal)
    {
      if constexpr (std::is_same_v<Value, Plaintext_<P2Data>> or std::is_same_v<Value, Plaintext_<FFT_Data>>)
      {
        return value.is_diagonal();
      }
      else
      {
        using std::begin;
        using std::end;
        return std::all_of(++begin(value), end(value), [](auto const& x) 
        { 
          if constexpr (CheckDecodingMod2)
          {
            return (x.get_limb(0) % 2) == 0;
          }
          else
          {
            return x == 0;
          }
        });
      }
    }
    return true;
  }

  template<typename Collection>
  [[nodiscard]] bool check_all_sparse(Collection const& collection) const
  {
    if (diagonal)
    {
      using std::begin;
      using std::end;
      return std::all_of(begin(collection), end(collection), [this](auto const& x) { return check_sparse(x); });
    }
    return true;
  }

  template<typename Value>
  void randomize(PRNG& prng, Value& value) const
  {
    if (diagonal)
    {
      value.randomize(prng, Diagonal);
    }
    else
    {
      value.randomize(prng);
    }
  }

  template<typename Value>
  void randomize(PRNG& prng, Value& value, int bits) const
  {
    value.randomize(prng, bits, diagonal);
  }
};

class NonInteractiveProof : public Proof
{
public:
  bigint static slack(int sec, int phim)
  { return bigint(phim * sec * sec) << (sec / 2 + 8); }

  NonInteractiveProof(int sec, const FHE_PK& pk,
      int extra_slack, bool diagonal = false) :
      Proof(sec, pk, 1, diagonal)
  {
    bigint B;
    B=128*sec*sec;
    B <<= extra_slack;
    B_plain_length = numBits(B*phim*tau);
    B_rand_length = numBits(B*3*phim*rho);
    plain_check = (bigint(1) << B_plain_length) - sec * tau;
    rand_check = (bigint(1) << B_rand_length) - sec * rho;
  }
};

class InteractiveProof : public Proof
{
public:
  bigint static slack(int sec, int phim)
  { (void)phim; return pow(2, 1.5 * sec + 1); }

  InteractiveProof(int sec, const FHE_PK& pk,
      int n_proofs = 1, bool diagonal = false) :
      Proof(sec, pk, n_proofs, diagonal)
  {
    bigint B;
    B = bigint(1) << sec;
    B_plain_length = numBits(B * tau);
    B_rand_length = numBits(B * rho);
    // leeway for completeness
    plain_check = (bigint(2) << B_plain_length);
    rand_check = (bigint(2) << B_rand_length);
  }
};

template<typename BaseProof>
class SparseProof : public BaseProof
{
  std::vector<int> sparcity;

public:
  SparseProof(std::vector<int> const& sparse, int sec, const FHE_PK& pk,
      int n_proofs = 1, bool diagonal = false) :
        BaseProof(sec, pk, n_proofs, true) ,
        sparcity(sparse)
  {
    this->diagonal = diagonal;
  }

  std::vector<int> const& get_sparcity() const
  {
    return sparcity;
  }

  bool use_top_gear(const FHE_PK&)
  {
    assert(this->top_gear == false);
    return this->top_gear;
  }

  static int n_ciphertext_per_proof(int sec, const FHE_PK& pk, bool =
      false)
  {
    return BaseProof::n_ciphertext_per_proof(sec, pk, true);
  }

  template<typename Value, bool CheckDecodingMod2 = false>
  [[nodiscard]] bool check_sparse(Value const& value, std::bool_constant<CheckDecodingMod2> check = {}) const
  {
    if (this->get_diagonal())
    {
      return BaseProof::check_sparse(value, check);
    }
    else
    {
      if constexpr (std::is_same_v<Value, Plaintext_<P2Data>> or std::is_same_v<Value, Plaintext_<FFT_Data>>)
      {
        assert(sparcity.size() == value.num_slots());
        for (unsigned int i = 0; i < value.num_slots(); ++i)
        {
          if (sparcity[i] and (value.coeff(i) != 0))
          {
            return false;
          }
        }
      }
      else
      {
        assert(sparcity.size() == value.size());
        using std::begin;
        using std::end;
        return std::transform_reduce(begin(value), end(value), begin(sparcity), true, std::logical_and{}, [](auto const& x, auto sparse)
        {
          if (sparse)
          {
            if constexpr (CheckDecodingMod2)
            {
              if ((x.get_limb(0) % 2) != 0)
              {
                return false;
              }
            }
            else
            {
              if (x != 0)
              {
                return false;
              }
            }
          }
          return true;
        });
      }
      return true;
    }
  }

  template<typename Collection>
  [[nodiscard]] bool check_all_sparse(Collection const& collection) const
  {
    using std::begin;
    using std::end;
    return std::all_of(begin(collection), end(collection), [this](auto const& x) { return check_sparse(x); });
  }

  template<typename Value>
  void randomize(PRNG& prng, Value& value) const
  {
    if (this->get_diagonal())
    {
      BaseProof::randomize(prng, value);
    }
    else
    {
      value.allocate(Polynomial);
      assert(sparcity.size() == value.num_slots());
      for (unsigned int i = 0; i < value.num_slots(); i++)
      {
        using FD = std::remove_cvref_t<decltype(value.get_field())>;
        using poly_type = typename FD::poly_type;
        poly_type x = {};
        if (not sparcity[i])
        {
          prng.randomBnd(bigint::tmp, value.get_field().get_prime(), true);
          x = bigint::tmp;
        }
        value.set_coeff(i, x);
      }
    }
  }

  template<typename Value>
  void randomize(PRNG& prng, Value& value, int bits) const
  {
    if (this->get_diagonal())
    {
      BaseProof::randomize(prng, value, bits);
    }
    else
    {
      value.allocate(Polynomial);
      assert(sparcity.size() == value.num_slots());
      for (unsigned int i = 0; i < value.num_slots(); i++)
      {
        using FD = std::remove_cvref_t<decltype(value.get_field())>;
        using poly_type = typename FD::poly_type;
        poly_type x = {};
        if (not sparcity[i])
        {
          x.generateUniform(prng, bits, false);
        }
        value.set_coeff(i, x);
      }
    }
  }
};

using SparseInteractiveProof = SparseProof<InteractiveProof>;
using SparseNonInteractiveProof = SparseProof<NonInteractiveProof>;

#endif

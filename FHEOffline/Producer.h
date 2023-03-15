/*
 * Producer.h
 *
 */

#ifndef FHEOFFLINE_PRODUCER_H_
#define FHEOFFLINE_PRODUCER_H_

#include "Networking/Player.h"
#include "FHE/Plaintext.h"
#include "FHEOffline/EncCommit.h"
#include "FHEOffline/DistDecrypt.h"
#include "FHEOffline/Sacrificing.h"
#include "Protocols/Share.h"
#include "Math/Setup.h"
#include "Protocols/mac_key.hpp"
#include "FHE/P2Data.h"
#include "FHE/FFT_Data.h"

#include "FHEOffline/SimpleEncCommit.h"

#include <memory>
#include <optional>

template <class T>
string prep_filename(string type, int my_num, int thread_num,
    bool initial, string dir = PREP_DIR);
template <class T>
string open_prep_file(ofstream& outf, string type, int my_num, int thread_num,
    bool initial, bool clear, string dir = PREP_DIR);

template <class FD>
class Producer
{
protected:
  int n_slots;
  int output_thread;
  bool write_output;
  string dir;

public:
  typedef typename FD::T T;
  typedef typename FD::S S;

  map<string, Timer> timers;

  Producer(int output_thread = 0, bool write_output = true);
  virtual ~Producer() {}
  virtual string data_type() = 0;
  virtual condition required_randomness() { return Full; }

  virtual string open_file(ofstream& outf, int my_num, int thread_num, bool initial,
      bool clear);
  virtual void clear_file(int my_num, int thread_num = 0, bool initial = true);

  virtual void clear() = 0;
  virtual void run(const Player& P, const FHE_PK& pk,
      const Ciphertext& calpha, EncCommitBase<T, FD, S>& EC,
      DistDecrypt<FD>& dd, const T& alphai) = 0;
  virtual int sacrifice(const Player& P, MAC_Check<T>& MC) = 0;
  int num_slots() { return n_slots; }

  virtual size_t report_size(ReportType type) { (void)type; return 0; }
};

template <class T, class FD, class S>
class TripleProducer : public TripleSacriFactory< Share<T> >, public Producer<FD>
{
  unsigned int i;

public:
  Plaintext_<FD> values[3], macs[3];
  Plaintext<T,FD,S> &ai, &bi, &ci;
  Plaintext<T,FD,S> &gam_ai, &gam_bi, &gam_ci;

  string data_type() { return "Triples"; }

  TripleProducer(const FD& Field, int my_num, int output_thread = 0,
      bool write_output = true, string dir = PREP_DIR);

  void clear() { this->triples.clear(); }
  void run(const Player& P, const FHE_PK& pk,
      const Ciphertext& calpha, EncCommitBase<T, FD, S>& EC,
      DistDecrypt<FD>& dd, const T& alphai);

  int sacrifice(const Player& P, MAC_Check<T>& MC);
  void get(Share<T>& a, Share<T>& b, Share<T>& c);
  void reset() { i = 0; }

  int num_slots() { return ai.num_slots(); }

  size_t report_size(ReportType type);
};

template <class FD>
using TripleProducer_ = TripleProducer<typename FD::T, FD, typename FD::S>;

template <class FD>
class TupleProducer : public Producer<FD>
{
protected:
  typedef typename FD::T T;

  unsigned int i;

public:
  Plaintext_<FD> values[2], macs[2];

  TupleProducer(const FD& FieldD, int output_thread = 0,
      bool write_output = true);
  virtual ~TupleProducer() {}

  void compute_macs(const FHE_PK& pk, const Ciphertext& calpha,
      EncCommitBase_<FD>& EC, DistDecrypt<FD>& dd, Ciphertext& ca,
      Ciphertext & cb);

  virtual void get(Share<T>& a, Share<T>& b);
};

template <class FD>
class InverseProducer : public TupleProducer<FD>,
    public TupleSacriFactory< Share<typename FD::T> >
{
  typedef typename FD::T T;

  Plaintext_<FD> ab;

  TripleProducer_<FD> triple_producer;
  bool produce_triples;

public:
  Plaintext_<FD> &ai, &bi;
  Plaintext_<FD> &gam_ai, &gam_bi;

  InverseProducer(const FD& FieldD, int my_num, int output_thread = 0,
      bool write_output = true, bool produce_triples = true,
      string dir = PREP_DIR);
  string data_type() { return "Inverses"; }

  void clear() { this->tuples.clear(); }
  void run(const Player& P, const FHE_PK& pk, const Ciphertext& calpha, 
      EncCommitBase_<FD>& EC, DistDecrypt<FD>& dd, const T& alphai);
  void get(Share<T>& a, Share<T>& b);
  int sacrifice(const Player& P, MAC_Check<T>& MC);
};

template <class FD>
class SquareProducer : public TupleProducer<FD>,
    public TupleSacriFactory< Share<typename FD::T> >
{
public:
  typedef typename FD::T T;

  Plaintext_<FD> &ai, &bi;
  Plaintext_<FD> &gam_ai, &gam_bi;

  SquareProducer(const FD& FieldD, int my_num, int output_thread = 0,
      bool write_output = true, string dir = PREP_DIR);
  string data_type() { return "Squares"; }

  void clear() { this->tuples.clear(); }
  void run(const Player& P, const FHE_PK& pk, const Ciphertext& calpha,
      EncCommitBase_<FD>& EC, DistDecrypt<FD>& dd, const T& alphai);
  void get(Share<T>& a, Share<T>& b) { TupleProducer<FD>::get(a, b); }
  int sacrifice(const Player& P, MAC_Check<T>& MC);
};

class gfpBitProducer : public Producer<FFT_Data>,
    public SingleSacriFactory< Share<gfp> >
{
  typedef FFT_Data FD;

  unsigned int i;
  Plaintext_<FD> vi, gam_vi;

  SquareProducer<FFT_Data> square_producer;
  bool produce_squares;

public:
  vector< Share<gfp> > bits;

  gfpBitProducer(const FD& FieldD, int my_num, int output_thread = 0,
      bool write_output = true, bool produce_squares = true,
      string dir = PREP_DIR);
  string data_type() { return "Bits"; }

  void clear() { bits.clear(); }
  void run(const Player& P, const FHE_PK& pk, const Ciphertext& calpha,
      EncCommitBase_<FD>& EC, DistDecrypt<FD>& dd, const gfp& alphai);
  int output(const Player& P, int thread, int output_thread = 0);

  void get(Share<gfp>& a);
  int sacrifice(const Player& P, MAC_Check<T>& MC);
};

// glue
template<class FD>
Producer<FD>* new_bit_producer(const FD& FieldD, const Player& P,
    const FHE_PK& pk, int covert,
    bool produce_squares = true, int thread_num = 0, bool write_output = true,
    string dir = PREP_DIR);

class gf2nBitProducer : public Producer<P2Data>
{
  typedef P2Data FD;
  typedef gf2n_short T;

  ofstream outf;
  bool write_output;

  EncCommit_<FD> ECB;

public:
  gf2nBitProducer(const Player& P, const FHE_PK& pk, int covert,
      int output_thread = 0, bool write_output = true, string dir = PREP_DIR);
  string data_type() { return "Bits"; }

  void clear() {}
  void run(const Player& P, const FHE_PK& pk, const Ciphertext& calpha,
      EncCommitBase_<FD>& EC, DistDecrypt<FD>& dd, const T& alphai);

  int sacrifice(const Player& P, MAC_Check<T>& MC);
};

template <class FD>
class InputProducer : public Producer<FD>
{
  typedef typename FD::T T;
  typedef typename FD::S S;

  ofstream* outf;
  const Player& P;
  bool write_output;

public:
  vector<vector<InputTuple<Share<T>>>> inputs;

  InputProducer(const Player& P, int output_thread = 0, bool write_output = true,
      string dir = PREP_DIR);
  ~InputProducer();

  string data_type() { return "Inputs"; }

  void clear() {}

  void run(const Player& P, const FHE_PK& pk, const Ciphertext& calpha,
      EncCommitBase_<FD>& EC, DistDecrypt<FD>& dd, const T& alphai)
  {
      run(P, pk, calpha, EC, dd, alphai, -1);
  }

  void run(const Player& P, const FHE_PK& pk, const Ciphertext& calpha,
      EncCommitBase_<FD>& EC, DistDecrypt<FD>& dd, const T& alphai,
      int player);

  int sacrifice(const Player& P, MAC_Check<T>& MC);

  // no ops
  string open_file(ofstream& outf, int my_num, int thread_num, bool initial,
      bool clear);
  void clear_file(int my_num, int thread_num = 0, bool initial = 0);

};

template<typename Container, std::size_t Index>
struct indexed_view_of
{
  Container& container;
  decltype(auto) operator [](std::size_t i)
  {
    return container[i][Index];
  }

  std::size_t size() const
    requires requires { container.size(); }
  {
    return container.size();
  }
};

template<class FD>
class BaseVectorTripleProducer : public Producer<FD>, public VectorTripleSacriFactory<Share<typename FD::T>>
{
public:
  using T = typename FD::T;
protected:
  std::vector<std::array<std::vector<T>, 5>> presacrificing_shares;
  std::vector<std::array<std::vector<T>, 5>> presacrificing_macs;

  template<std::size_t I>
  auto get_presacrificing_shares() { return indexed_view_of<decltype(presacrificing_shares), I>(presacrificing_shares); }
  template<std::size_t I>
  auto get_presacrificing_macs() { return indexed_view_of<decltype(presacrificing_macs), I>(presacrificing_macs); }

public:
  BaseVectorTripleProducer(int my_num, int output_thread = 0, bool write_output = true, string dir = PREP_DIR)
    : Producer<FD>(output_thread, write_output)
  {
    this->dir = dir;
    (void)my_num;
  }

  void clear()
  { 
    this->triples.clear();
    // don't clear presacrificing_* as it is overwritten in clear_and_set_dimensions anyways
  }
  void get(vector<Share<T>>& a, vector<Share<T>>& b, vector<Share<T>>& c) override 
  {
    (void)a; (void)b; (void)c; 
    throw std::runtime_error("get is not available for BaseVectorTripleProducer"); 
  }

  template<typename ShareT>
  void transfer_triples(std::vector<std::array<std::vector<ShareT>, 3>>& triple_target)
  {
    assert(not this->triples.empty());
    triple_target.reserve(triple_target.size() + this->triples.size());
    for (auto& [a, b, c] : this->triples)
    {
      auto& [out_a, out_b, out_c] = triple_target.emplace_back();

      out_a.reserve(a.size());
      for (auto& x : a)
      {
        out_a.emplace_back(x);
      }

      out_b.reserve(b.size());
      for (auto& x : b)
      {
        out_b.emplace_back(x);
      }

      out_c.reserve(c.size());
      for (auto& x : c)
      {
        out_c.emplace_back(x);
      }
    }
    this->triples.clear();
  }
};

class DummyEC {};

template<typename... EC>
class BaseProducerWithECs
{
protected:
  using EC_types = std::tuple<std::map<std::vector<int>, EC>...>;

  using EC_ptr_types = std::tuple<typename std::map<std::vector<int>, EC>::iterator...>;
  
  static constexpr std::size_t size = sizeof...(EC);

  EC_types ECs;
  EC_ptr_types EC_ptrs;

  template<std::size_t I, typename... Args>
  bool try_emplace(std::vector<int> key, Args&&... args)
  {
    bool emplaced;
    std::tie(std::get<I>(EC_ptrs), emplaced) = std::get<I>(ECs).try_emplace(key, key, std::forward<Args>(args)...);
    return emplaced;
  }

  template<std::size_t I, typename... Args>
  bool try_emplace(std::nullopt_t, Args&&... args)
  {
    bool emplaced;
    std::tie(std::get<I>(EC_ptrs), emplaced) = std::get<I>(ECs).try_emplace({}, std::forward<Args>(args)...);
    return emplaced;
  }

  template<std::size_t I>
  auto end()
  {
    return std::get<I>(ECs).end();
  }

  template<std::size_t I>
  void clear()
  {
    std::get<I>(ECs).clear();
    std::get<I>(EC_ptrs) = end<I>();
  }

  void clear()
  {
    EC_ptrs = std::apply([](auto&... maps)
    {
      (maps.clear(), ...);
      return std::make_tuple(maps.end()...);
    }, ECs);
  }

  BaseProducerWithECs()
    : ECs(), EC_ptrs(std::apply([](auto&... maps) { return std::make_tuple(maps.end()...); }, ECs))
  {
  }
};

template<class FD, typename... EC>
class BaseSummingTripleProducer : public BaseProducerWithECs<EC...>
{
protected:
  Player const& P;
  FHE_PK const& pk;
  MachineBase& machine;
  FD const& FieldD;

  BaseSummingTripleProducer(Player const& P, FHE_PK const& pk, MachineBase& machine, FD const& FieldD)
    : P(P)
    , pk(pk)
    , machine(machine)
    , FieldD(FieldD)
  {
  }


};

template<class FD, typename... EC>
class BasePairwiseTripleProducer : public BaseProducerWithECs<EC...>
{
protected:
  PairwiseGenerator<FD>& generator;

  BasePairwiseTripleProducer(PairwiseGenerator<FD>& generator)
    : generator(generator)
  {
  }
};

#endif /* FHEOFFLINE_PRODUCER_H_ */

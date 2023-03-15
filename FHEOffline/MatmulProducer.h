#pragma once

#include "FHEOffline/Producer.h"

template<class FD>
class BaseMatmulTripleProducer : public BaseVectorTripleProducer<FD>
{
public:
  using T = typename FD::T;
protected:
  matmul_dimensions dimensions;
public:

  BaseMatmulTripleProducer(int my_num, int output_thread = 0, bool write_output = true, string dir = PREP_DIR)
    : BaseVectorTripleProducer<FD>(my_num, output_thread, write_output, dir)
  {
  }

  virtual int adapt_for_dimensions(matmul_dimensions) = 0;
  void clear_and_set_dimensions(matmul_dimensions dimensions);

  string data_type() { return "Matmul"; }
  int sacrifice(const Player& P, MAC_Check<T>& MC) override;

  void produce(matmul_dimensions dimensions, const Player& P, MAC_Check<T>& MC, const FHE_PK& pk, const Ciphertext& calpha, EncCommitBase_<FD>& EC, DistDecrypt<FD>& dd, const T& alphai);
};

template<class FD>
class SummingMatmulTripleProducer : public BaseMatmulTripleProducer<FD>, public BaseSummingTripleProducer<FD, SparseSummingEncCommit<FD>, SparseSummingEncCommit<FD>>
{
public:
  using T = typename FD::T;

  SummingMatmulTripleProducer(const Player& P, const FHE_PK& pk, MachineBase& machine, const FD& Field, int my_num, int output_thread = 0, bool write_output = true, string dir = PREP_DIR)
    : BaseMatmulTripleProducer<FD>(my_num, output_thread, write_output, dir)
    , BaseSummingTripleProducer<FD, SparseSummingEncCommit<FD>, SparseSummingEncCommit<FD>>(P, pk, machine, Field)
  {
  }
  void run(const Player&, const FHE_PK& pk, const Ciphertext& calpha, EncCommitBase_<FD>&, DistDecrypt<FD>& dd, const T& /*alphai*/) override;
  
  int adapt_for_dimensions(matmul_dimensions dimensions) override;
};

template<typename FD>
#if defined(CONV2D_DIRECT_SUM) or defined(CONV2D_BASIC_MATMUL)
using RightPairwiseMatmulEC = ReusableMultiEncCommit<FD>;
#else
using RightPairwiseMatmulEC = ReusableSparseMultiEncCommit<FD>;
#endif

template<class FD>
class PairwiseMatmulTripleProducer : public BaseMatmulTripleProducer<FD>, public BasePairwiseTripleProducer<FD, ReusableSparseMultiEncCommit<FD>, RightPairwiseMatmulEC<FD>>
{
public:
  using T = typename FD::T;

  PairwiseMatmulTripleProducer(PairwiseGenerator<FD>& generator, int my_num, int output_thread = 0, bool write_output = true, string dir = PREP_DIR)
    : BaseMatmulTripleProducer<FD>(my_num, output_thread, write_output, dir)
    , BasePairwiseTripleProducer<FD, ReusableSparseMultiEncCommit<FD>, RightPairwiseMatmulEC<FD>>(generator)
    {
#if defined(CONV2D_DIRECT_SUM) or defined(CONV2D_BASIC_MATMUL)
      this->template try_emplace<1>(std::nullopt, generator.P, generator.machine.other_pks, generator.machine.template setup<FD>().FieldD, this->timers, generator.machine, generator, false);
#endif
    }

  void run(const Player&, const FHE_PK&, const Ciphertext&, EncCommitBase_<FD>&, DistDecrypt<FD>&, const T&) override;

  int adapt_for_dimensions(matmul_dimensions dimensions) override;
};
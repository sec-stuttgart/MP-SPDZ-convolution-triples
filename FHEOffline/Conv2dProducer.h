#pragma once

#include "FHEOffline/Producer.h"

template<class FD>
class BaseConv2dTripleProducer : public BaseVectorTripleProducer<FD>
{
public:
  using T = typename FD::T;
protected:
  convolution_dimensions dimensions;

  int sacrifice(convolution_dimensions dimensions, const Player& P, MAC_Check<T>& MC);
  int sacrifice(depthwise_convolution_triple_dimensions dimensions, const Player& P, MAC_Check<T>& MC);
public:

  BaseConv2dTripleProducer(int my_num, int output_thread = 0, bool write_output = true, string dir = PREP_DIR)
    : BaseVectorTripleProducer<FD>(my_num, output_thread, write_output, dir)
  {
  }

  virtual int adapt_for_dimensions(convolution_dimensions) = 0;
  virtual int adapt_for_dimensions(depthwise_convolution_triple_dimensions) = 0;

  template<typename ConvolutionDimensions>
  void clear_and_set_dimensions(ConvolutionDimensions dimensions);

  string data_type() { return "Conv2d"; }
  int sacrifice(const Player& P, MAC_Check<T>& MC) override;

  template<typename ConvolutionDimensions>
  void produce(ConvolutionDimensions dimensions, const Player& P, MAC_Check<T>& MC, const FHE_PK& pk, const Ciphertext& calpha, EncCommitBase_<FD>& EC, DistDecrypt<FD>& dd, const T& alphai);
};

template<class FD>
class SummingConv2dTripleProducer : public BaseConv2dTripleProducer<FD>, public BaseSummingTripleProducer<FD, SparseSummingEncCommit<FD>, SparseSummingEncCommit<FD>>
{
private:
  template<typename ConvolutionDimensions>
  int adapt(ConvolutionDimensions dimensions);

public:
  using T = BaseConv2dTripleProducer<FD>::T;

  SummingConv2dTripleProducer(const Player& P, const FHE_PK& pk, MachineBase& machine, const FD& Field, int my_num, int output_thread = 0, bool write_output = true, string dir = PREP_DIR)
    : BaseConv2dTripleProducer<FD>(my_num, output_thread, write_output, dir)
    , BaseSummingTripleProducer<FD, SparseSummingEncCommit<FD>, SparseSummingEncCommit<FD>>(P, pk, machine, Field)
  {
  }
  void run(const Player&, const FHE_PK& pk, const Ciphertext& calpha, EncCommitBase_<FD>&, DistDecrypt<FD>& dd, const T& /*alphai*/) override;
  
  int adapt_for_dimensions(convolution_dimensions dimensions) override { return adapt(dimensions); }
  int adapt_for_dimensions(depthwise_convolution_triple_dimensions dimensions) override { return adapt(dimensions); }
};

template<class FD>
#ifdef CONV2D_LOWGEAR_EXPANDED_BGV
using ThirdPairwiseEC = ReusableMultiEncCommit<FD>;
#else
using ThirdPairwiseEC = DummyEC;
#endif

template<class FD>
class PairwiseConv2dTripleProducer : public BaseConv2dTripleProducer<FD>, public BasePairwiseTripleProducer<FD, ReusableSparseMultiEncCommit<FD>, ReusableSparseMultiEncCommit<FD>, ThirdPairwiseEC<FD>>
{
private:
  template<typename ConvolutionDimensions>
  int adapt(ConvolutionDimensions dimensions);

public:
  using T = typename FD::T;

  PairwiseConv2dTripleProducer(PairwiseGenerator<FD>& generator, int my_num, int output_thread = 0, bool write_output = true, string dir = PREP_DIR)
    : BaseConv2dTripleProducer<FD>(my_num, output_thread, write_output, dir)
    , BasePairwiseTripleProducer<FD, ReusableSparseMultiEncCommit<FD>, ReusableSparseMultiEncCommit<FD>, ThirdPairwiseEC<FD>>(generator)
    {
#ifdef CONV2D_LOWGEAR_EXPANDED_BGV
      this->template try_emplace<2>(std::nullopt, generator.P, generator.machine.other_pks, generator.machine.template setup<FD>().FieldD, this->timers, generator.machine, generator, false);
#else
      this->template try_emplace<2>(std::nullopt);
#endif
    }

  void run(const Player&, const FHE_PK&, const Ciphertext&, EncCommitBase_<FD>&, DistDecrypt<FD>&, const T&) override;

  int adapt_for_dimensions(convolution_dimensions dimensions) override { return adapt(dimensions); }
  int adapt_for_dimensions(depthwise_convolution_triple_dimensions dimensions) override { return adapt(dimensions); }
};
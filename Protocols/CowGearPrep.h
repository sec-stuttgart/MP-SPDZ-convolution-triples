/*
 * CowGearPrep.h
 *
 */

#ifndef PROTOCOLS_COWGEARPREP_H_
#define PROTOCOLS_COWGEARPREP_H_

#include "Protocols/ReplicatedPrep.h"
#include "FHEOffline/Producer.h"
#include "FHEOffline/MatmulProducer.h"
#include "FHEOffline/Conv2dProducer.h"

class PairwiseMachine;
template<class FD> class PairwiseGenerator;

/**
 * LowGear/CowGear preprocessing
 */
template<class T>
class CowGearPrep : public MaliciousRingPrep<T>
{
    typedef typename T::mac_key_type mac_key_type;
    typedef typename T::clear::FD FD;

    static PairwiseMachine* pairwise_machine;
    static Lock lock;

    PairwiseGenerator<typename T::clear::FD>* pairwise_generator;
    std::unique_ptr<BaseConv2dTripleProducer<FD>> conv2d_producer;
    std::unique_ptr<BaseMatmulTripleProducer<FD>> matmul_producer;

    PairwiseGenerator<FD>& get_generator();

    template<int>
    void buffer_bits(true_type);
    template<int>
    void buffer_bits(false_type);

public:
    static void basic_setup(Player& P);
    static void key_setup(Player& P, mac_key_type alphai);
    static void setup(Player& P, mac_key_type alphai);
    static void teardown();

    CowGearPrep(SubProcessor<T>* proc, DataPositions& usage) :
            BufferPrep<T>(usage),
            BitPrep<T>(proc, usage), RingPrep<T>(proc, usage),
            MaliciousDabitOnlyPrep<T>(proc, usage),
            MaliciousRingPrep<T>(proc, usage),
            pairwise_generator(0)
    {
    }
    ~CowGearPrep();

    void buffer_triples();
    void buffer_bits();
    void buffer_inputs(int player);
    void buffer_matmul_triples(matmul_dimensions dimensions) override;
    void buffer_conv2d_triples(convolution_dimensions dimensions) override;
    void buffer_conv2d_triples(depthwise_convolution_triple_dimensions dimensions) override;
};

#endif /* PROTOCOLS_COWGEARPREP_H_ */

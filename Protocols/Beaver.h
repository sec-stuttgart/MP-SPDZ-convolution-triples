/*
 * Beaver.h
 *
 */

#ifndef PROTOCOLS_BEAVER_H_
#define PROTOCOLS_BEAVER_H_

#include <vector>
#include <array>
using namespace std;

#include "Replicated.h"
#include "Processor/Data_Files.h"

template<class T> class SubProcessor;
template<class T> class MAC_Check_Base;
class Player;

/**
 * Beaver multiplication
 */
template<class T>
class Beaver : public ProtocolBase<T>
{
protected:
    vector<T> shares;
    vector<typename T::open_type> opened;
    vector<array<T, 3>> triples;
    vector<array<vector<T>, 3>> vtriples;
    typename vector<typename T::open_type>::iterator it;
    typename vector<array<T, 3>>::iterator triple;
    typename vector<array<vector<T>, 3>>::iterator vtriple;
    Preprocessing<T>* prep;
    typename T::MAC_Check* MC;

public:
    static const bool uses_triples = true;

    Player& P;

    Beaver(Player& P) : prep(0), MC(0), P(P) {}

    typename T::Protocol branch();

    void init(Preprocessing<T>& prep, typename T::MAC_Check& MC);

    void init_mul();
    void prepare_mul(const T& x, const T& y, int n = -1);
    void exchange();
    T finalize_mul(int n = -1);

    void init_matmul() { init_mul(); }
    void prepare_matmul(std::span<T const> registers, matmul_desc matmul);
    void finalize_matmul(std::span<T> registers, matmul_desc matmul);

    void init_conv2d() { init_mul(); }
    template<typename ConvoltionDesc>
    void prepare_conv2d(std::span<T const> registers, ConvoltionDesc conv);
    void finalize_conv2d(std::span<T> registers, convolution_desc conv);
    void finalize_conv2d(std::span<T> registers, depthwise_convolution_desc conv);

    void check();

    void start_exchange();
    void stop_exchange();

    int get_n_relevant_players() { return 1 + T::threshold(P.num_players()); }
};

#endif /* PROTOCOLS_BEAVER_H_ */

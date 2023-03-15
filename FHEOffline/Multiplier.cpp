/*
 * Multiplier.cpp
 *
 */

#include <FHEOffline/Multiplier.h>
#include "FHEOffline/PairwiseGenerator.h"
#include "FHEOffline/PairwiseMachine.h"

#include "Math/modp.hpp"

template <class FD>
Multiplier<FD>::Multiplier(int offset, PairwiseGenerator<FD>& generator) :
        Multiplier(offset, generator.machine, generator.P, generator.timers)
{
}

template <class FD>
Multiplier<FD>::Multiplier(int offset, PairwiseMachine& machine, Player& P,
        map<string, Timer>& timers) :
    machine(machine),
    P(P, offset),
    num_players(P.num_players()),
    my_num(P.my_num()),
    other_pk(machine.other_pks[(my_num + num_players - offset) % num_players]),
    other_enc_alpha(machine.enc_alphas[(my_num + num_players - offset) % num_players]),
    timers(timers),
    C(machine.pk), mask(machine.pk),
#ifdef CONV2D_LOWGEAR_EXPANDED_BGV
    xC(machine.pk.get_params()), xMask(machine.pk.get_params()),
#endif
    product_share(machine.setup<FD>().FieldD), rc(machine.pk),
    volatile_capacity(0)
{
    product_share.allocate_slots(machine.pk.p() << 64);
}

template <class FD>
void Multiplier<FD>::multiply_and_add(Plaintext_<FD>& res,
        const Ciphertext& enc_a, const Plaintext_<FD>& b)
{
    Rq_Element bb(enc_a.get_params(), evaluation, evaluation);
    bb.from(b.get_iterator());
    multiply_and_add(res, enc_a, bb);
}

template <class FD>
void Multiplier<FD>::multiply_and_add(Plaintext_<FD>& res,
        const Ciphertext& enc_a, const Rq_Element& b, OT_ROLE role)
{
    if (role & SENDER)
    {
        timers["Ciphertext multiplication"].start();
        C.mul(enc_a, b);
        timers["Ciphertext multiplication"].stop();
    }

    add(res, C, role);
}

template <class FD>
void Multiplier<FD>::add(Plaintext_<FD>& res, const Ciphertext& c,
        OT_ROLE role, int)
{
    o.reset_write_head();

    if (role & SENDER)
    {
        PRNG G;
        G.ReSeed();
        timers["Mask randomization"].start();
        product_share.randomize(G);
        mask = c;
        mask.rerandomize(other_pk);
        timers["Mask randomization"].stop();
        mask += product_share;
        mask.pack(o);
        res -= product_share;
    }

    timers["Multiplied ciphertext sending"].start();
    if (role == BOTH)
        P.reverse_exchange(o);
    else if (role == SENDER)
        P.reverse_send(o);
    else if (role == RECEIVER)
        P.receive(o);
    timers["Multiplied ciphertext sending"].stop();

    if (role & RECEIVER)
    {
        timers["Decryption"].start();
        C.unpack(o);
        machine.sk.decrypt_any(product_share, C);
        res += product_share;
        timers["Decryption"].stop();
    }

    memory_usage.update("multiplied ciphertext", C.report_size(CAPACITY));
    memory_usage.update("mask ciphertext", mask.report_size(CAPACITY));
    memory_usage.update("product shares", product_share.report_size(CAPACITY));
    memory_usage.update("masking random coins", rc.report_size(CAPACITY));
}

#ifdef CONV2D_LOWGEAR_EXPANDED_BGV
template <class FD>
void Multiplier<FD>::conv_and_add(Plaintext_<FD>& res, Ciphertext const& enc_a, MultiConvolution_Matrix const& b, OT_ROLE role)
{
    if (role & SENDER)
    {
        timers["Ciphertext convolution"].start();
#ifdef VERBOSE_CONV2D
        std::cout << CONV2D_NOW << " start: ciphertext convolution" << std::endl;
        b.mul(xC, enc_a);
        std::cout << CONV2D_NOW << " end:   ciphertext convolution" << std::endl;
#else
        b.mul(xC, enc_a);
#endif
        timers["Ciphertext convolution"].stop();
    }

    add(res, xC, role);
}

template <class FD>
void Multiplier<FD>::add(Plaintext_<FD>& res, ExpandedCiphertext const& c, OT_ROLE role)
{
    o.reset_write_head();

    if (role & SENDER)
    {
        PRNG G;
        G.ReSeed();
        timers["Mask randomization"].start();
        product_share.randomize(G);
        xMask = c;
#ifdef VERBOSE_CONV2D
        std::cout << CONV2D_NOW << " start: ciphertext rerandomization" << std::endl;
        xMask.rerandomize(other_pk);
        std::cout << CONV2D_NOW << " end:   ciphertext rerandomization" << std::endl;
#else
        xMask.rerandomize(other_pk);
#endif
        timers["Mask randomization"].stop();
        xMask += product_share;
        xMask.pack(o);
        res -= product_share;
    }

#ifdef VERBOSE_CONV2D
    std::cout << CONV2D_NOW << " start: ciphertext sending" << std::endl;
#endif
    timers["Convolved ciphertext sending"].start();
    if (role == BOTH)
        P.reverse_exchange(o);
    else if (role == SENDER)
        P.reverse_send(o);
    else if (role == RECEIVER)
        P.receive(o);
    timers["Convolved ciphertext sending"].stop();
#ifdef VERBOSE_CONV2D
    std::cout << CONV2D_NOW << " end:   ciphertext sending" << std::endl;
#endif

    if (role & RECEIVER)
    {
        timers["Decryption"].start();
        xC.unpack(o);
#ifdef VERBOSE_CONV2D
        std::cout << CONV2D_NOW << " start: ciphertext decryption" << std::endl;
        machine.sk.decrypt(product_share, xC);
        std::cout << CONV2D_NOW << " end:   ciphertext decryption" << std::endl;
#else
        machine.sk.decrypt(product_share, xC);
#endif
        res += product_share;
        timers["Decryption"].stop();
    }

    memory_usage.update("convolved ciphertext", xC.report_size(CAPACITY));
    memory_usage.update("extended mask ciphertext", xMask.report_size(CAPACITY));
    memory_usage.update("product shares", product_share.report_size(CAPACITY));
    memory_usage.update("masking random coins", rc.report_size(CAPACITY));
}
#endif

template <class FD>
void Multiplier<FD>::multiply_alpha_and_add(Plaintext_<FD>& res,
        const Rq_Element& b, OT_ROLE role)
{
    multiply_and_add(res, other_enc_alpha, b, role);
}

template <class FD>
size_t Multiplier<FD>::report_size(ReportType type)
{
    return C.report_size(type) + mask.report_size(type)
            + product_share.report_size(type) + rc.report_size(type);
}

template <class FD>
void Multiplier<FD>::report_size(ReportType type, MemoryUsage& res)
{
    (void)type;
    res += memory_usage;
}

template<class FD>
const vector<Ciphertext>& Multiplier<FD>::get_multiplicands(
        const vector<vector<Ciphertext> >& others_ct, const FHE_PK&)
{
    return others_ct[P.get_full_player().get_player(-P.get_offset())];
}


template class Multiplier<FFT_Data>;
template class Multiplier<P2Data>;

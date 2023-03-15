/*
 * SimpleEncCommit.cpp
 *
 */

#include <FHEOffline/SimpleEncCommit.h>
#include "FHEOffline/SimpleMachine.h"
#include "FHEOffline/Multiplier.h"
#include "FHEOffline/PairwiseGenerator.h"
#include "Tools/Subroutines.h"
#include "Protocols/MAC_Check.h"

#include "Protocols/MAC_Check.hpp"
#include "Math/modp.hpp"

template<class T, class FD, class S>
SimpleEncCommitBase<T, FD, S>::SimpleEncCommitBase(const MachineBase& machine) :
        EncCommitBase_<FD>(&machine),
        extra_slack(machine.extra_slack), n_rounds(0)
{
}

template<class T,class FD,class S>
void SimpleEncCommitBase<T, FD, S>::generate_ciphertexts(
        AddableVector<Ciphertext>& c, const vector<Plaintext_<FD> >& m,
        Proof::Randomness& r, const FHE_PK& pk, TimerMap& timers,
        Proof& proof)
{
    timers["Generating"].start();
    PRNG G;
    G.ReSeed();
    prepare_plaintext(G);
    Random_Coins rc(pk.get_params());
    c.resize(proof.U, pk);
    r.resize(proof.U, pk);
    for (unsigned i = 0; i < proof.U; i++)
    {
        r[i].sample(G);
        rc.assign(r[i]);
        pk.encrypt(c[i], m.at(i), rc);
    }
    timers["Generating"].stop();
    memory_usage.update("random coins", rc.report_size(CAPACITY));
}

template<class T,class FD,class S,class This>
void SimpleEncCommitImpl<T,FD,S,This>::create_more()
{
    cout << "Generating more ciphertexts in round " << this->n_rounds << endl;
    octetStream ciphertexts, cleartexts;
    size_t prover_memory = this->generate_proof(this->c, this->m, ciphertexts, cleartexts);
    size_t verifier_memory =
            NonInteractiveProofSimpleEncCommitImpl<FD, This>::create_more(ciphertexts,
                    cleartexts);
    cout << "Done checking proofs in round " << this->n_rounds << endl;
    this->n_rounds++;
    this->cnt = static_cast<This*>(this)->get_proof().U - 1;
    this->memory_usage.update("serialized ciphertexts",
            ciphertexts.get_max_length());
    this->memory_usage.update("serialized cleartexts", cleartexts.get_max_length());
    this->volatile_memory = max(prover_memory, verifier_memory)
            + ciphertexts.get_max_length() + cleartexts.get_max_length();
}

template<class T, class FD, class S,class This>
void SimpleEncCommitImpl<T, FD, S, This>::add_ciphertexts(
        vector<Ciphertext>& ciphertexts, int offset)
{
    (void)offset;
    auto& proof = static_cast<This*>(this)->get_proof();
    for (unsigned j = 0; j < proof.U; j++)
        add(this->c[j], this->c[j], ciphertexts[j]);
}

template<class FD, class This>
void SummingEncCommitImpl<FD, This>::create_more()
{
    octetStream cleartexts;
    const Player& P = this->P;
    AddableVector<Ciphertext> commitments;
    size_t prover_size;
    MemoryUsage& memory_usage = this->memory_usage;
    TreeSum<Ciphertext> tree_sum(2, 2, thread_num % P.num_players());
    octetStream& ciphertexts = tree_sum.get_buffer();

    auto& proof = static_cast<This*>(this)->get_proof();
    {
#ifdef LESS_ALLOC_MORE_MEM
        Proof::Randomness& r = preimages.r;
#else
        Proof::Randomness r(proof.U, this->pk.get_params());
        Prover<FD, Plaintext_<FD> > prover(proof, this->FTD);
#endif
        this->generate_ciphertexts(this->c, this->m, r, pk, timers, proof);
        this->timers["Stage 1 of proof"].start();
        prover.Stage_1(proof, ciphertexts, this->c, this->pk);
        this->timers["Stage 1 of proof"].stop();

        this->c.unpack(ciphertexts, this->pk);
        commitments.unpack(ciphertexts, this->pk);

#ifdef VERBOSE_HE
        cout << "Tree-wise sum of ciphertexts with "
                << 1e-9 * ciphertexts.get_length() << " GB" << endl;
#endif
        this->timers["Exchanging ciphertexts"].start();
        tree_sum.run(this->c, P);
        tree_sum.run(commitments, P);
        this->timers["Exchanging ciphertexts"].stop();

        proof.generate_challenge(P);

        this->timers["Stage 2 of proof"].start();
        prover.Stage_2(proof, cleartexts, this->m, r, pk);
        this->timers["Stage 2 of proof"].stop();

        prover_size = prover.report_size(CAPACITY) + r.report_size(CAPACITY)
                + prover.volatile_memory;
        memory_usage.update("prover", prover.report_size(CAPACITY));
        memory_usage.update("randomness", r.report_size(CAPACITY));
    }

#ifndef LESS_ALLOC_MORE_MEM
    Proof::Preimages preimages(proof.V, this->pk, this->FTD.get_prime(),
        P.num_players());
#endif
    preimages.unpack(cleartexts);

    this->timers["Committing"].start();
    AllCommitments cleartext_commitments(P);
    cleartext_commitments.commit_and_open(cleartexts);
    this->timers["Committing"].stop();

    for (int i = 1; i < P.num_players(); i++)
    {
#ifdef VERBOSE_HE
        cout << "Sending cleartexts with " << 1e-9 * cleartexts.get_length()
                << " GB in round " << i << endl;
#endif
        TimeScope(this->timers["Exchanging cleartexts"]);
        P.pass_around(cleartexts);
        preimages.add(cleartexts);
        cleartext_commitments.check_relative(i, cleartexts);
    }

    ciphertexts.reset_write_head();
    this->c.pack(ciphertexts);
    commitments.pack(ciphertexts);
    cleartexts.clear();
    cleartexts.resize_precise(preimages.report_size(USED));
    preimages.pack(cleartexts);
    this->timers["Verifying"].start();
#ifdef LESS_ALLOC_MORE_MEM
    Verifier<FD>& verifier = this->verifier;
#else
    Verifier<FD> verifier(proof);
#endif
    verifier.Stage_2(proof, this->c, ciphertexts, cleartexts,
            this->pk);
    this->timers["Verifying"].stop();
    this->cnt = proof.U - 1;

    this->volatile_memory =
            + commitments.report_size(CAPACITY) + ciphertexts.get_max_length()
            + cleartexts.get_max_length()
            + max(prover_size, preimages.report_size(CAPACITY))
            + tree_sum.report_size(CAPACITY);
    memory_usage.update("verifier", verifier.report_size(CAPACITY));
    memory_usage.update("preimages", preimages.report_size(CAPACITY));
    memory_usage.update("commitments", commitments.report_size(CAPACITY));
    memory_usage.update("received cleartexts", cleartexts.get_max_length());
    memory_usage.update("tree sum", tree_sum.report_size(CAPACITY));
}

template <class FD, class This>
size_t SimpleEncCommitFactoryImpl<FD, This>::report_size(ReportType type)
{
    return m.report_size(type) + c.report_size(type);
}

template<class FD, class This>
void SimpleEncCommitFactoryImpl<FD, This>::report_size(ReportType type, MemoryUsage& res)
{
    res.add("my plaintexts", m.report_size(type));
    res.add("my ciphertexts", c.report_size(type));
}

template<class FD, class This>
size_t SummingEncCommitImpl<FD, This>::report_size(ReportType type)
{
#ifdef LESS_ALLOC_MORE_MEM
    return prover.report_size(type) + preimages.report_size(type);
#else
    (void)type;
    return 0;
#endif
}

template class SimpleEncCommitBase<gfp, FFT_Data, bigint>;
template class SimpleEncCommit<gfp, FFT_Data, bigint>;
template class SimpleEncCommitFactory<FFT_Data>;
template class SummingEncCommit<FFT_Data>;
template class SparseSummingEncCommit<FFT_Data>;
template class MultiEncCommit<FFT_Data>;
template class ReusableMultiEncCommit<FFT_Data>;
template class ReusableSparseMultiEncCommit<FFT_Data>;

template class SimpleEncCommitBase<gf2n_short, P2Data, int>;
template class SimpleEncCommit<gf2n_short, P2Data, int>;
template class SimpleEncCommitFactory<P2Data>;
template class SummingEncCommit<P2Data>;
template class SparseSummingEncCommit<P2Data>;
template class MultiEncCommit<P2Data>;
template class ReusableMultiEncCommit<P2Data>;
template class ReusableSparseMultiEncCommit<P2Data>;

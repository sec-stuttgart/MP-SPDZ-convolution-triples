/*
 * SimpleEncCommit.h
 *
 */

#ifndef FHEOFFLINE_SIMPLEENCCOMMIT_H_
#define FHEOFFLINE_SIMPLEENCCOMMIT_H_

#include "FHEOffline/EncCommit.h"
#include "FHEOffline/Proof.h"
#include "FHEOffline/Prover.h"
#include "FHEOffline/Verifier.h"
#include "FHEOffline/SimpleMachine.h"
#include "Tools/MemoryUsage.h"
#include "Tools/config.h"

#include <memory>

class MachineBase;

typedef map<string, Timer> TimerMap;

template<class T,class FD,class S>
class SimpleEncCommitBase : public EncCommitBase<T,FD,S>
{
protected:
    int extra_slack;

    int n_rounds;

    void generate_ciphertexts(AddableVector<Ciphertext>& c,
            const vector<Plaintext_<FD> >& m, Proof::Randomness& r,
            const FHE_PK& pk, map<string, Timer>& timers, Proof& proof);

    virtual void prepare_plaintext(PRNG& G) = 0;

public:
    MemoryUsage memory_usage;

    SimpleEncCommitBase(const MachineBase& machine);
    virtual ~SimpleEncCommitBase() {}
    void report_size(ReportType type, MemoryUsage& res) { (void)type; res += memory_usage; }
};

template <class FD>
using SimpleEncCommitBase_ = SimpleEncCommitBase<typename FD::T, FD, typename FD::S>;

template <class FD, class This>
class NonInteractiveProofSimpleEncCommitImpl : public SimpleEncCommitBase_<FD>
{
public:
    using proof_type = NonInteractiveProof;
public:
    proof_type& get_proof() { return proof; }

protected:
    const PlayerBase& P;
    const FHE_PK& pk;
    const FD& FTD;

    virtual const FHE_PK& get_pk_for_verification(int offset) = 0;
    virtual void add_ciphertexts(vector<Ciphertext>& ciphertexts, int offset) = 0;

private:
    proof_type proof;
public:
#ifdef LESS_ALLOC_MORE_MEM
    Proof::Randomness r;
    Prover<FD, Plaintext_<FD> > prover;
    Verifier<FD> verifier;
#endif

    map<string, Timer>& timers;

    NonInteractiveProofSimpleEncCommitImpl(const PlayerBase& P, const FHE_PK& pk,
            const FD& FTD, map<string, Timer>& timers,
				       const MachineBase& machine, bool diagonal = false)
            :
            SimpleEncCommitBase_<FD>(machine),
            P(P), pk(pk), FTD(FTD),
            proof(machine.sec, pk, machine.extra_slack, diagonal),
#ifdef LESS_ALLOC_MORE_MEM
            r(proof.U, this->pk.get_params()), prover(proof, FTD),
            verifier(proof, FTD),
#endif
            timers(timers)
                {
                }
    virtual ~NonInteractiveProofSimpleEncCommitImpl() {}
    size_t generate_proof(AddableVector<Ciphertext>& c,
            vector<Plaintext_<FD> >& m, octetStream& ciphertexts,
            octetStream& cleartexts)
    {
        auto& proof = static_cast<This*>(this)->get_proof();

        timers["Proving"].start();
#ifndef LESS_ALLOC_MORE_MEM
        Proof::Randomness r(proof.U, pk.get_params());
#endif
        this->generate_ciphertexts(c, m, r, pk, timers, proof);
#ifndef LESS_ALLOC_MORE_MEM
        Prover<FD, Plaintext_<FD> > prover(proof, FTD);
#endif
        size_t prover_memory = prover.NIZKPoK(proof, ciphertexts, cleartexts,
                pk, c, m, r);
        timers["Proving"].stop();

        if (proof.top_gear)
        {
            c += c;
            for (auto& mm : m)
                mm += mm;
        }

        // cout << "Checking my own proof" << endl;
        // if (!Verifier<gfp>().NIZKPoK(c[P.my_num()], proofs[P.my_num()], pk, false, false))
        // 	throw runtime_error("proof check failed");
        this->memory_usage.update("randomness", r.report_size(CAPACITY));
        prover.report_size(CAPACITY, this->memory_usage);
        return r.report_size(CAPACITY) + prover_memory;
    }
    size_t create_more(octetStream& ciphertexts, octetStream& cleartexts)
    {
        auto& proof = static_cast<This*>(this)->get_proof();

        AddableVector<Ciphertext> others_ciphertexts;
        others_ciphertexts.resize(proof.U, pk.get_params());
        for (int i = 1; i < P.num_players(); i++)
        {
#ifdef VERBOSE_HE
            cerr << "Sending proof with " << 1e-9 * ciphertexts.get_length() << "+"
                    << 1e-9 * cleartexts.get_length() << " GB" << endl;
#endif
            timers["Sending"].start();
            P.pass_around(ciphertexts);
            P.pass_around(cleartexts);
            timers["Sending"].stop();
#ifndef LESS_ALLOC_MORE_MEM
            Verifier<FD,S> verifier(proof);
#endif
#ifdef VERBOSE_HE
            cerr << "Checking proof of player " << i << endl;
#endif
            timers["Verifying"].start();
            verifier.NIZKPoK(proof, others_ciphertexts, ciphertexts,
                    cleartexts, get_pk_for_verification(i));
            timers["Verifying"].stop();
            add_ciphertexts(others_ciphertexts, i);
            this->memory_usage.update("verifier", verifier.report_size(CAPACITY));
        }
        this->memory_usage.update("cleartexts", cleartexts.get_max_length());
        this->memory_usage.update("others' ciphertexts", others_ciphertexts.report_size(CAPACITY));
#ifdef LESS_ALLOC_MORE_MEM
        return others_ciphertexts.report_size(CAPACITY);
#else
        return others_ciphertexts.report_size(CAPACITY)
                + this->memory_usage.get("verifier");
#endif
    }
    virtual size_t report_size(ReportType type)
    {
#ifdef LESS_ALLOC_MORE_MEM
        return r.report_size(type) +
            prover.report_size(type) + verifier.report_size(type);
#else
        (void)type;
        return 0;
#endif
    }
    using SimpleEncCommitBase_<FD>::report_size;
};

template <class FD, class This>
class SimpleEncCommitFactoryImpl
{
protected:
    int cnt;
    AddableVector<Ciphertext> c;
    AddableVector< Plaintext_<FD> > m;

    int n_calls;

    const FHE_PK& pk;

    void prepare_plaintext(PRNG& G)
    {
        for (auto& mess : m)
            static_cast<This*>(this)->get_proof().randomize(G, mess);
    }

    virtual void create_more() = 0;

public:
    SimpleEncCommitFactoryImpl(const FHE_PK& pk, const FD& FTD,
            const MachineBase& machine, bool diagonal = false) :
        cnt(-1), n_calls(0), pk(pk)
    {
        int sec = This::proof_type::n_ciphertext_per_proof(machine.sec, pk, diagonal);
        c.resize(sec, pk.get_params());
        m.resize(sec, FTD);
        for (int i = 0; i < sec; i++)
        {
            m[i].assign_zero(Polynomial);
        }
    }
    virtual ~SimpleEncCommitFactoryImpl() {
#ifdef VERBOSE_HE
        if (n_calls > 0)
            cout << "EncCommit called " << n_calls << " times" << endl;
#endif
    }
    bool has_left() { return cnt >= 0; }
    int remaining_capacity() { return cnt + 1; }
    int number_of_calls() { return n_calls; }
    void next(Plaintext_<FD>& mess, Ciphertext& C)
    {
        if (not has_left())
            create_more();
        mess = m[cnt];
        C = c[cnt];

        if (static_cast<This*>(this)->get_proof().use_top_gear(pk))
        {
            mess = mess + mess;
            C = C + C;
        }

        cnt--;
        n_calls++;
    }
    virtual size_t report_size(ReportType type);
    void report_size(ReportType type, MemoryUsage& res);
};

template<class FD>
class SimpleEncCommitFactory : public SimpleEncCommitFactoryImpl<FD, SimpleEncCommitFactory<FD>>{};

template<class T,class FD,class S, class This>
class SimpleEncCommitImpl: public NonInteractiveProofSimpleEncCommitImpl<FD, This>,
        public SimpleEncCommitFactoryImpl<FD, This>
{
protected:
    const FHE_PK& get_pk_for_verification(int)
    { return NonInteractiveProofSimpleEncCommitImpl<FD, This>::pk; }
    void prepare_plaintext(PRNG& G)
    { SimpleEncCommitFactoryImpl<FD, This>::prepare_plaintext(G); }
    void add_ciphertexts(vector<Ciphertext>& ciphertexts, int offset);

public:
    SimpleEncCommitImpl(const PlayerBase& P, const FHE_PK& pk, const FD& FTD,
            map<string, Timer>& timers, const MachineBase& machine,
            int thread_num, bool diagonal = false) :
                NonInteractiveProofSimpleEncCommitImpl<FD, This>(P, pk, FTD, timers, machine,
                        diagonal),
                SimpleEncCommitFactoryImpl<FD, This>(pk, FTD, machine, diagonal)
                {
                    (void)thread_num;
                }
    void next(Plaintext_<FD>& mess, Ciphertext& C) { SimpleEncCommitFactoryImpl<FD, This>::next(mess, C); }
    void create_more();
    size_t report_size(ReportType type)
    { return SimpleEncCommitFactoryImpl<FD, This>::report_size(type) + EncCommitBase_<FD>::report_size(type); }
    void report_size(ReportType type, MemoryUsage& res)
    { SimpleEncCommitFactoryImpl<FD, This>::report_size(type, res); SimpleEncCommitBase_<FD>::report_size(type, res); }
};

template<class T, class FD, class S>
class SimpleEncCommit : public SimpleEncCommitImpl<T, FD, S, SimpleEncCommit<T, FD, S>>
{
    using SimpleEncCommitImpl<T, FD, S, SimpleEncCommit<T, FD, S>>::SimpleEncCommitImpl;
};

template <class FD>
using SimpleEncCommit_ = SimpleEncCommit<typename FD::T, FD, typename FD::S>;

template <class FD, class This>
class SummingEncCommitImpl: public SimpleEncCommitFactoryImpl<FD, This>,
        public SimpleEncCommitBase_<FD>
{
    InteractiveProof proof;
    const FHE_PK& pk;
    const FD& FTD;
    const Player& P;
    int thread_num;

protected:
#ifdef LESS_ALLOC_MORE_MEM
    Prover<FD, Plaintext_<FD> > prover;
    Verifier<FD> verifier;
    Proof::Preimages preimages;
#endif
private:

    void prepare_plaintext(PRNG& G)
    { SimpleEncCommitFactoryImpl<FD, This>::prepare_plaintext(G); }

public:
    using proof_type = InteractiveProof;
    map<string, Timer>& timers;

    SummingEncCommitImpl(const Player& P, const FHE_PK& pk, const FD& FTD,
            map<string, Timer>& timers, const MachineBase& machine,
            int thread_num, bool diagonal = false) :
                SimpleEncCommitFactoryImpl<FD, This>(pk, FTD, machine, diagonal), SimpleEncCommitBase_<FD>(
                machine), proof(machine.sec, pk, P.num_players(), diagonal), pk(pk), FTD(
                FTD), P(P), thread_num(thread_num),
#ifdef LESS_ALLOC_MORE_MEM
                prover(proof, FTD), verifier(proof, FTD), preimages(proof.V,
                        this->pk, FTD.get_prime(), P.num_players()),
#endif
                timers(timers)
                {
                }

    void next(Plaintext_<FD>& mess, Ciphertext& C) { SimpleEncCommitFactoryImpl<FD, This>::next(mess, C); }
    void create_more();
    size_t report_size(ReportType type);
    void report_size(ReportType type, MemoryUsage& res)
    { SimpleEncCommitFactoryImpl<FD, This>::report_size(type, res); SimpleEncCommitBase_<FD>::report_size(type, res); }
    InteractiveProof& get_proof() { return proof; }
};

template<class FD>
class SummingEncCommit : public SummingEncCommitImpl<FD, SummingEncCommit<FD>>
{
    using SummingEncCommitImpl<FD, SummingEncCommit<FD>>::SummingEncCommitImpl;
};

template <class FD>
class Multiplier;
template <class FD>
class PairwiseGenerator;

template <class FD, class This>
class MultiEncCommitImpl : public NonInteractiveProofSimpleEncCommitImpl<FD, This>
{
    friend PairwiseGenerator<FD>;

protected:
    const vector<FHE_PK>& pks;
    const Player& P;
    PairwiseGenerator<FD>& generator;

    void prepare_plaintext(PRNG& G) { (void)G; }
    const FHE_PK& get_pk_for_verification(int offset)
    { return pks[(P.my_num() - offset + P.num_players()) % P.num_players()]; }
    void add_ciphertexts(vector<Ciphertext>& ciphertexts, int offset)
    {
        auto& proof = static_cast<This*>(this)->get_proof();
        for (unsigned i = 0; i < proof.U; i++)
            generator.multipliers[offset - 1]->multiply_and_add(generator.c.at(i),
                    ciphertexts.at(i), generator.b_mod_q.at(i));
    }

public:
    MultiEncCommitImpl(const Player& P, const vector<FHE_PK>& pks,
            const FD& FTD,
            map<string, Timer>& timers, MachineBase& machine,
            PairwiseGenerator<FD>& generator, bool diagonal = false) :
        NonInteractiveProofSimpleEncCommitImpl<FD, This>(P, pks[P.my_real_num()], FTD,
                timers, machine, diagonal), pks(pks), P(P), generator(generator)
    {
    }
};

template<class FD>
class MultiEncCommit : public MultiEncCommitImpl<FD, MultiEncCommit<FD>>
{ 
public:
    using MultiEncCommitImpl<FD, MultiEncCommit<FD>>::MultiEncCommitImpl;
};

template<typename FD, typename This>
class SparseSummingEncCommitImpl : public SummingEncCommitImpl<FD, This>
{
public:
    using proof_type = SparseInteractiveProof;
private:
    proof_type proof;
public:
    proof_type& get_proof() { return proof; }

    SparseSummingEncCommitImpl(std::vector<int> const& sparcity, const Player& P, const FHE_PK& pk, const FD& FTD,
            map<string, Timer>& timers, const MachineBase& machine,
            int thread_num, bool diagonal = false)
            : SummingEncCommitImpl<FD, This>(P, pk, FTD, timers, machine, thread_num, diagonal)
            , proof(sparcity, machine.sec, pk, P.num_players(), diagonal)
    {
#ifdef LESS_ALLOC_MORE_MEM
        this->prover = Prover<FD, Plaintext_<FD>>(proof, FTD);
        this->preimages = Proof::Preimages(proof.V, pk, FTD.get_prime(), P.num_players());
#endif
    }

    std::vector<int> const& get_sparcity() const
    {
        return proof.get_sparcity();
    }
};

template<typename FD>
class SparseSummingEncCommit : public SparseSummingEncCommitImpl<FD, SparseSummingEncCommit<FD>>
{
public:
    using SparseSummingEncCommitImpl<FD, SparseSummingEncCommit<FD>>::SparseSummingEncCommitImpl;
};

template<typename FD, typename This>
class ReusableMultiEncCommitImpl : public MultiEncCommitImpl<FD, This>, public SimpleEncCommitFactoryImpl<FD, This>
{
protected:
    std::vector<std::vector<Ciphertext>> other_ciphertexts;

public:
    ReusableMultiEncCommitImpl(const Player& P, const vector<FHE_PK>& pks, const FD& FTD, map<string, Timer>& timers, MachineBase& machine, PairwiseGenerator<FD>& generator, bool diagonal = false)
            : MultiEncCommitImpl<FD, This>(P, pks, FTD, timers, machine, generator, diagonal)
            , SimpleEncCommitFactoryImpl<FD, This>(this->MultiEncCommitImpl<FD, This>::pk, FTD, machine, diagonal)
    {
    }

    void prepare_plaintext(PRNG& G) override
    {
        SimpleEncCommitFactoryImpl<FD, This>::prepare_plaintext(G);
    }

    void next(Plaintext_<FD>& mess, std::vector<Ciphertext>& C)
    {
        if (not this->has_left())
        {
            this->create_more();
        }
        mess = this->m[this->cnt];
        C = other_ciphertexts[this->cnt];

        if (static_cast<This*>(this)->get_proof().use_top_gear(this->MultiEncCommitImpl<FD, This>::pk))
        {
            mess = mess + mess;
            for (auto& c : C)
            {
                c = c + c;
            }
        }

        this->cnt--;
        this->n_calls++;
    }

    void add_ciphertexts(vector<Ciphertext>& ciphertexts, int offset) override
    {
        if (other_ciphertexts.empty())
        {
            other_ciphertexts.resize(ciphertexts.size());
        }
        for (std::size_t i = 0; i < ciphertexts.size(); ++i)
        {
            assert(std::cmp_equal(other_ciphertexts[i].size(), offset - 1));
            other_ciphertexts[i].push_back(ciphertexts[i]);
        }
    }

    void create_more()
    {
        other_ciphertexts.clear();
        octetStream ciphertexts;
        octetStream cleartexts;
        this->generate_proof(this->c, this->m, ciphertexts, cleartexts);
        this->NonInteractiveProofSimpleEncCommitImpl<FD, This>::create_more(ciphertexts, cleartexts);
        this->cnt = static_cast<This*>(this)->get_proof().U - 1;
    }
};

template<typename FD, typename This>
class ReusableSparseMultiEncCommitImpl : public ReusableMultiEncCommitImpl<FD, This>
{
public:
    using proof_type = SparseNonInteractiveProof;
private:
    proof_type proof;
public:
    proof_type& get_proof() { return proof; }

    ReusableSparseMultiEncCommitImpl(std::vector<int> const& sparcity, const Player& P, const vector<FHE_PK>& pks, const FD& FTD, map<string, Timer>& timers, MachineBase& machine, PairwiseGenerator<FD>& generator, bool diagonal = false)
            : ReusableMultiEncCommitImpl<FD, This>(P, pks, FTD, timers, machine, generator, diagonal)
            , proof(sparcity, machine.sec, this->MultiEncCommitImpl<FD, This>::pk, P.num_players(), diagonal)
    {
#ifdef LESS_ALLOC_MORE_MEM
        this->r = Proof::Randomness(proof.U, this->MultiEncCommitImpl<FD, This>::pk.get_params());
        this->prover = Prover<FD, Plaintext_<FD>>(proof, FTD);
#endif
    }

    std::vector<int> const& get_sparcity() const
    {
        return proof.get_sparcity();
    }
};

template<typename FD>
class ReusableMultiEncCommit : public ReusableMultiEncCommitImpl<FD, ReusableMultiEncCommit<FD>>
{
public:
    using ReusableMultiEncCommitImpl<FD, ReusableMultiEncCommit<FD>>::ReusableMultiEncCommitImpl;
};

template<typename FD>
class ReusableSparseMultiEncCommit : public ReusableSparseMultiEncCommitImpl<FD, ReusableSparseMultiEncCommit<FD>>
{
public:
    using ReusableSparseMultiEncCommitImpl<FD, ReusableSparseMultiEncCommit<FD>>::ReusableSparseMultiEncCommitImpl;
};

#endif /* FHEOFFLINE_SIMPLEENCCOMMIT_H_ */

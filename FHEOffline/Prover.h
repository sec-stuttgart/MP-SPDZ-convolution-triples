#ifndef _Prover
#define _Prover

#include "Proof.h"
#include "Tools/MemoryUsage.h"
#include "FHE/AddableVector.hpp"

/* Class for the prover */

template<class FD, class U>
class Prover
{
  /* Provers state */
  Proof::Randomness s;
  AddableVector< Plaintext_<FD> > y;

#ifdef LESS_ALLOC_MORE_MEM
  AddableVector<typename Proof::bound_type> z;
  AddableMatrix<Int_Random_Coins::value_type::value_type> t;
#endif

public:
  size_t volatile_memory;

  Prover(Proof& proof, const FD& FieldD);

  template<typename ProofT>
  void Stage_1(const ProofT& P, octetStream& ciphertexts, const AddableVector<Ciphertext>& c,
      const FHE_PK& pk)
  {
    size_t allocate = 3 * c.size() * c[0].report_size(USED);
    ciphertexts.resize_precise(allocate);
    ciphertexts.reset_write_head();
    c.pack(ciphertexts);

    int V=P.V;

    // AElement<T> AE;
    // ZZX rd;
    // ZZ pr=(*AE.A).prime();
    // ZZ bd=B_plain/(pr+1);
    PRNG G;
    G.ReSeed();
    Random_Coins rc(pk.get_params());
    Ciphertext ciphertext(pk.get_params());
    ciphertexts.store(V);
    for (int i=0; i<V; i++)
      {
        // AE.randomize(Diag,binary);
        // rd=RandPoly(phim,bd<<1);
        // y[i]=AE.plaintext()+pr*rd;
        P.randomize(G, y[i], P.B_plain_length);
        assert(P.check_sparse(y[i]));
        s[i].resize(3, P.phim);
        s[i].generateUniform(G, P.B_rand_length);
        rc.assign(s[i][0], s[i][1], s[i][2]);
        pk.encrypt(ciphertext,y[i],rc);
        ciphertext.pack(ciphertexts);
      }
  }

  template<typename ProofT>
  bool Stage_2(ProofT& P, octetStream& cleartexts,
               const vector<U>& x,
               const Proof::Randomness& r,
               const FHE_PK& pk)
  {
    size_t allocate = P.V * P.phim
        * (5 + numBytes(P.plain_check) + 3 * (5 + numBytes(P.rand_check)));
    cleartexts.resize_precise(allocate);
    cleartexts.reset_write_head();

    unsigned int i;
#ifndef LESS_ALLOC_MORE_MEM
    AddableVector<fixint<gfp::N_LIMBS>> z;
    AddableMatrix<fixint<gfp::N_LIMBS>> t;
#endif
    cleartexts.reset_write_head();
    cleartexts.store(P.V);
    assert(P.check_all_sparse(x));
    for (i=0; i<P.V; i++)
      { z=y[i];
        t=s[i];
        P.apply_challenge(i, z, x, pk);
        assert(P.check_sparse(z, check_decoding_mod2(x[0].get_field())));
        P.apply_challenge(i, t, r, pk);
        if (not P.check_bounds(z, t, i))
            return false;
        z.pack(cleartexts);
        t.pack(cleartexts);
    }
#ifndef LESS_ALLOC_MORE_MEM
    volatile_memory = t.report_size(CAPACITY) + z.report_size(CAPACITY);
#endif
#ifdef PRINT_MIN_DIST
    cout << "Minimal distance (log) " << log2(P.dist) << ", compare to " <<
        log2(P.plain_check.get_d() / pow(2, P.B_plain_length))  << endl;
#endif
    return true;
  }

  /* Only has a non-interactive version using the ROM 
      - If Diag is true then the plaintexts x are assumed to be
        diagonal elements, i.e. x=(x_1,x_1,...,x_1)
  */
  template<typename ProofT>
  size_t NIZKPoK(ProofT& P, octetStream& ciphertexts, octetStream& cleartexts,
	       const FHE_PK& pk,
               const AddableVector<Ciphertext>& c,
               const vector<U>& x,
               const Proof::Randomness& r)
  {
  //  AElement<T> AE;
  //  for (i=0; i<P.sec; i++)
  //    { AE.assign(x.at(i));
  //      if (!AE.to_type(0))
  //         { cout << "Error in making x[i]" << endl;
  //           cout << i << endl;
  //         }
  //    }

    bool ok=false;
    int cnt=0;
    while (!ok)
      { cnt++;
        Stage_1(P,ciphertexts,c,pk);
        P.set_challenge(ciphertexts);
        // Check check whether we are OK, or whether we should abort
        ok = Stage_2(P,cleartexts,x,r,pk);
      }
#ifdef VERBOSE
    if (cnt > 1)
        cout << "\t\tNumber iterations of prover = " << cnt << endl;
#endif
    return report_size(CAPACITY) + volatile_memory;
  }

  size_t report_size(ReportType type);
  void report_size(ReportType type, MemoryUsage& res);
};

#endif

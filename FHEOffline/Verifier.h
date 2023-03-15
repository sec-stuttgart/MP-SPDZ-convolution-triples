#ifndef _Verifier
#define _Verifier

#include "Proof.h"
#include "Math/Z2k.hpp"
#include "Math/modp.hpp"

/* Defines the Verifier */
template <class FD>
class Verifier
{
  AddableVector<typename Proof::bound_type> z;
  AddableMatrix<Int_Random_Coins::value_type::value_type> t;

  const FD& FieldD;

public:
  Verifier(Proof& proof, const FD& FieldD);

  template<typename ProofT>
  void Stage_2(ProofT& proof,
      AddableVector<Ciphertext>& c, octetStream& ciphertexts,
      octetStream& cleartexts,const FHE_PK& pk)
  {
    unsigned int i, V;

    c.unpack(ciphertexts, pk);
    if (c.size() != proof.U)
      throw length_error("number of received ciphertexts incorrect");

    // Now check the encryptions are correct
    Ciphertext d1(pk.get_params()), d2(pk.get_params());
    Random_Coins rc(pk.get_params());
    ciphertexts.get(V);
    if (V != proof.V)
      throw length_error("number of received commitments incorrect");
    cleartexts.get(V);
    if (V != proof.V)
      throw length_error("number of received cleartexts incorrect");
    for (i=0; i<V; i++)
      {
        z.unpack(cleartexts);
        t.unpack(cleartexts);
        if (!proof.check_bounds(z, t, i))
          throw runtime_error("preimage out of bounds");
        d1.unpack(ciphertexts);
        proof.apply_challenge(i, d1, c, pk);
        rc.assign(t[0], t[1], t[2]);
        pk.encrypt(d2,z,rc);
        if (!(d1 == d2))
          {
  #ifdef VERBOSE
            cout << "Fail Check 6 " << i << endl;
  #endif
            throw runtime_error("ciphertexts don't match");
          }
        if (!proof.check_sparse(z, check_decoding_mod2(FieldD)))
          {
  #ifdef VERBOSE
            cout << "\tCheck : " << i << endl;
  #endif
            throw runtime_error("cleartext isn't diagonal");
          }
      }
  }

  /* This is the non-interactive version using the ROM
      - Creates space for all output values
      - Diag flag mirrors that in Prover
  */
  template<typename ProofT>
  void NIZKPoK(ProofT& proof, AddableVector<Ciphertext>& c,octetStream& ciphertexts,octetStream& cleartexts,
               const FHE_PK& pk)
  {
    proof.set_challenge(ciphertexts);

    Stage_2(proof, c,ciphertexts,cleartexts,pk);

    if (proof.top_gear)
    {
      assert(not proof.get_diagonal());
      c += c;
    }
  }

  size_t report_size(ReportType type) { return z.report_size(type) + t.report_size(type); }
};

#endif

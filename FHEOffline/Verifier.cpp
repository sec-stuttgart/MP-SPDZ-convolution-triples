#include "Verifier.h"
#include "FHE/P2Data.h"
#include "Math/Z2k.hpp"
#include "Math/modp.hpp"

template <class FD>
Verifier<FD>::Verifier(Proof& proof, const FD& FieldD) :
    FieldD(FieldD)
{
#ifdef LESS_ALLOC_MORE_MEM
  z.resize(proof.phim);
  z.allocate_slots(bigint(1) << proof.B_plain_length);
  t.resize(3, proof.phim);
  t.allocate_slots(bigint(1) << proof.B_rand_length);
#endif
}


template class Verifier<FFT_Data>;
template class Verifier<P2Data>;

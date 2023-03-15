
#include "Prover.h"
#include "Verifier.h"

#include "FHE/P2Data.h"
#include "Tools/random.h"
#include "Math/Z2k.hpp"
#include "Math/modp.hpp"
#include "FHE/AddableVector.hpp"


template <class FD, class U>
Prover<FD,U>::Prover(Proof& proof, const FD& FieldD) :
  volatile_memory(0)
{
  s.resize(proof.V, proof.pk->get_params());
  y.resize(proof.V, FieldD);
#ifdef LESS_ALLOC_MORE_MEM
  t = s[0];
  z = y[0];
  // extra limb to prevent reallocation
  t.allocate_slots(bigint(1) << (proof.B_rand_length + 64));
  z.allocate_slots(bigint(1) << (proof.B_plain_length + 64));
  s.allocate_slots(bigint(1) << proof.B_rand_length);
  y.allocate_slots(bigint(1) << proof.B_plain_length);
#endif
}

template<class FD, class U>
size_t Prover<FD,U>::report_size(ReportType type)
{
  size_t res = 0;
  for (unsigned int i = 0; i < s.size(); i++)
    res += s[i].report_size(type);
  for (unsigned int i = 0; i < y.size(); i++)
    res += y[i].report_size(type);
#ifdef LESS_ALLOC_MORE_MEM
  res += z.report_size(type) + t.report_size(type);
#endif
  return res;
}


template<class FD, class U>
void Prover<FD, U>::report_size(ReportType type, MemoryUsage& res)
{
  res.update("prover s", s.report_size(type));
  res.update("prover y", y.report_size(type));
#ifdef LESS_ALLOC_MORE_MEM
  res.update("prover z", z.report_size(type));
  res.update("prover t", t.report_size(type));
#endif
  res.update("prover volatile", volatile_memory);
}


template class Prover<FFT_Data, Plaintext_<FFT_Data> >;
//template class Prover<FFT_Data, AddableVector<bigint> >;

template class Prover<P2Data, Plaintext_<P2Data> >;
//template class Prover<P2Data, AddableVector<bigint> >;

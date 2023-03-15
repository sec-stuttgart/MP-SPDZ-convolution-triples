#pragma once

#include <cassert>

#if defined(CONV2D_SUMMING_CIPHERTEXTS) and defined(CONV2D_MAX_IMAGE_DEPTH)
#define CONV2D_EXTRA_SLACK CONV2D_MAX_IMAGE_DEPTH
#undef CONV2D_SUMMING_CIPHERTEXTS
#define CONV2D_SUMMING_CIPHERTEXTS true
#else
#ifdef CONV2D_EXTRA_SLACK
  #undef CONV2D_EXTRA_SLACK
#endif
#define CONV2D_EXTRA_SLACK 1
#ifdef CONV2D_MAX_IMAGE_DEPTH
  #undef CONV2D_MAX_IMAGE_DEPTH
#endif
#define CONV2D_MAX_IMAGE_DEPTH 0
#ifdef CONV2D_SUMMING_CIPHERTEXTS
  #undef CONV2D_SUMMING_CIPHERTEXTS
#endif
#define CONV2D_SUMMING_CIPHERTEXTS false
#endif

#ifdef ASSERTIVE_CONV2D
#define CONV2D_ASSERT(...) assert(__VA_ARGS__)
#else
#define CONV2D_ASSERT(...) (void)0
#endif

#ifdef VERBOSE_CONV2D
#include <chrono>
#include <ctime>
#include <iomanip>
inline auto CONV2D_NOW_f()
{
  auto const n = std::chrono::system_clock::now();
  auto const t = std::chrono::system_clock::to_time_t(n);
  return std::put_time(std::localtime(&t), "%F %T");
}
#define CONV2D_NOW CONV2D_NOW_f()
#endif
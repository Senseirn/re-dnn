#include <iostream>
#include <vector>

#include "re-dnn/core/aligned_vector.h"

#define IUTEST_USE_MAIN 1
#include "iutest.hpp"

#define GTEST_COUT std::cerr << "[          ] [ INFO ] "

IUTEST(VectorAlignTest, isAligned) {
  std::size_t align = 0;

#ifdef __AVX__
  align = 32; // for avx
  GTEST_COUT << "SIMD Type: AVX" << std::endl;
#elif defined(__SSE__)
  align = 16; // for sse
  GTEST_COUT << "SIMD Type: SSE" << std::endl;
#else
  align = 1;
  GTEST_COUT << "SIMD Type: None" << std::endl;
#endif

  GTEST_COUT << "Test alignement: " << align << std::endl;

  using namespace rednn;
  aligned_vector<float> v(32);
  const auto is_aligned = ((reinterpret_cast<std::ptrdiff_t>(v.data())) % align) == 0;
  IUTEST_EXPECT_TRUE(is_aligned);
}
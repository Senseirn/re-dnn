/*
MIT License

Copyright (c) 2021 Senseirn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/**
 * @file aligned_allocator.h
 * @brief a header file for vector class and aligned allocator
 */

#pragma once

#include <cstdlib>
#include <immintrin.h>
#include <type_traits>

namespace rednn {

  namespace alignment {
    enum simd_type { sse = 16, avx = 32 };
  }

  static inline std::size_t default_align() {
#ifdef __AVX__
    return static_cast<std::size_t>(alignment::avx); // for avx
#elif defined(__SSE__)
    return static_cast<std::size_t>(alignment::sse); // for sse
#else
    return 1;
#endif
  }

  template <typename T>
  struct AlignedAllocator {
    using value_type = T;

    AlignedAllocator(){};
    AlignedAllocator(const AlignedAllocator&){};
    AlignedAllocator(AlignedAllocator&&){};

    /**
     * @brief allocation function
     *
     * @param n the number of elements of type T
     * @return T* a pointer of T which is aligned
     */
    T* allocate(std::size_t n) {
      // if C++17 or later
#if __cplusplus >= 201703L
      return reinterpret_cast<T*>(std::aligned_alloc(n * sizeof(T), default_align()));
#else
      return reinterpret_cast<T*>(_mm_malloc(n * sizeof(T), default_align()));
#endif
    }

    void deallocate(T* p, std::size_t n) {
      // if C++17 or later
#if __cplusplus >= 201703L
      std::free(p);
#else
      _mm_free(p);
#endif
    }
  };

  template <class T, class U>
  bool operator==(const AlignedAllocator<T>&, const AlignedAllocator<U>&) {
    return true;
  }

  template <class T, class U>
  bool operator!=(const AlignedAllocator<T>&, const AlignedAllocator<U>&) {
    return false;
  }

} // namespace rednn
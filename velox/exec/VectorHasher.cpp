/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/exec/VectorHasher.h"

#include <arm_neon.h>
// #include <immintrin.h> // This header defines AVX2 types and intrinsics>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Portability.h"
#include "velox/common/base/SimdUtil.h"
#include "velox/common/memory/HashStringAllocator.h"
#include "velox/type/FloatingPointUtil.h"

namespace facebook::velox::exec {

#define VALUE_ID_TYPE_DISPATCH(TEMPLATE_FUNC, typeKind, ...)                \
  [&]() {                                                                   \
    switch (typeKind) {                                                     \
      case TypeKind::BOOLEAN: {                                             \
        return TEMPLATE_FUNC<TypeKind::BOOLEAN>(__VA_ARGS__);               \
      }                                                                     \
      case TypeKind::TINYINT: {                                             \
        return TEMPLATE_FUNC<TypeKind::TINYINT>(__VA_ARGS__);               \
      }                                                                     \
      case TypeKind::SMALLINT: {                                            \
        return TEMPLATE_FUNC<TypeKind::SMALLINT>(__VA_ARGS__);              \
      }                                                                     \
      case TypeKind::INTEGER: {                                             \
        return TEMPLATE_FUNC<TypeKind::INTEGER>(__VA_ARGS__);               \
      }                                                                     \
      case TypeKind::BIGINT: {                                              \
        return TEMPLATE_FUNC<TypeKind::BIGINT>(__VA_ARGS__);                \
      }                                                                     \
      case TypeKind::VARCHAR:                                               \
      case TypeKind::VARBINARY: {                                           \
        return TEMPLATE_FUNC<TypeKind::VARCHAR>(__VA_ARGS__);               \
      }                                                                     \
      case TypeKind::TIMESTAMP: {                                           \
        return TEMPLATE_FUNC<TypeKind::TIMESTAMP>(__VA_ARGS__);             \
      }                                                                     \
      default:                                                              \
        VELOX_UNREACHABLE(                                                  \
            "Unsupported value ID type: ", TypeKindName::toName(typeKind)); \
    }                                                                       \
  }()

namespace {

constexpr uint64_t PRIME64_1 = 11400714785074694791ULL;
constexpr uint64_t PRIME64_2 = 14029467366897019727ULL;
constexpr uint64_t PRIME64_3 = 1609587929392839161ULL;
constexpr uint64_t PRIME64_4 = 9650029242287828579ULL;
constexpr uint64_t PRIME64_5 = 2870177450012600261ULL;

template <bool typeProvidesCustomComparison, TypeKind Kind>
uint64_t hashOne(DecodedVector& decoded, vector_size_t index) {
  if constexpr (
      Kind == TypeKind::ROW || Kind == TypeKind::ARRAY ||
      Kind == TypeKind::MAP) {
    // Virtual function call for complex type.
    return decoded.base()->hashValueAt(decoded.index(index));
  } else {
    // Inlined for scalars.
    using T = typename KindToFlatVector<Kind>::HashRowType;
    T value = decoded.valueAt<T>(index);

    if constexpr (typeProvidesCustomComparison) {
      return static_cast<const CanProvideCustomComparisonType<Kind>*>(
                 decoded.base()->type().get())
          ->hash(value);
    } else if constexpr (std::is_floating_point_v<T>) {
      return util::floating_point::NaNAwareHash<T>()(value);
    } else {
      return folly::hasher<T>()(value);
    }
  }
}

template <typename T>
uint64_t xxhash64(const T& value, uint64_t seed = 0) {
  static_assert(
      std::is_trivially_copyable_v<T>, "Type T must be trivially copyable");

  // Handle 8-byte types optimally
  if constexpr (sizeof(T) == 8) {
    uint64_t h64 =
        seed + PRIME64_5 + 8; // Initialize hash with seed and length (8 bytes)
    uint64_t k1 = *reinterpret_cast<const uint64_t*>(
        &value); // Read the value as a single 8-byte chunk
    k1 *= PRIME64_2; // Multiply by PRIME64_2
    k1 = (k1 << 31) | (k1 >> (64 - 31)); // Rotate left by 31 bits
    k1 *= PRIME64_1; // Multiply by PRIME64_1
    h64 ^= k1; // XOR the result into the running hash value
    h64 = ((h64 << 27) | (h64 >> (64 - 27))) * PRIME64_1 +
        PRIME64_4; // Final mixing

    // Final mixing to scramble the bits further
    h64 ^= h64 >> 33;
    h64 *= PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= PRIME64_3;
    h64 ^= h64 >> 32;

    return h64; // Return the final hash value
  } else {
    // Handle smaller or larger types in a general way
    const uint8_t* data = reinterpret_cast<const uint8_t*>(&value);
    size_t length = sizeof(T); // Use the size of T to handle non-8-byte types
    uint64_t h64 =
        seed + PRIME64_5 + length; // Initialize hash with seed and length

    // Process as many 8-byte chunks as possible
    while (length >= 8) {
      uint64_t k1 = *reinterpret_cast<const uint64_t*>(data);
      k1 *= PRIME64_2;
      k1 = (k1 << 31) | (k1 >> (64 - 31));
      k1 *= PRIME64_1;
      h64 ^= k1;
      h64 = ((h64 << 27) | (h64 >> (64 - 27))) * PRIME64_1 + PRIME64_4;
      data += 8;
      length -= 8;
    }

    // Process remaining bytes (if any)
    while (length > 0) {
      h64 ^= (*data) * PRIME64_5;
      h64 = ((h64 << 11) | (h64 >> (64 - 11))) * PRIME64_1;
      ++data;
      --length;
    }

    // Final mixing to scramble the bits further
    h64 ^= h64 >> 33;
    h64 *= PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= PRIME64_3;
    h64 ^= h64 >> 32;

    return h64; // Return the final hash value
  }
}
//
// void twang_unmix64_neon(
//    const uint64_t* inputArray,
//    size_t numElements,
//    uint64_t* outputArray) {
//  // NEON vectors for shifts and other operations
//  uint64x2_t multiplier1 = vdupq_n_u64(4611686016279904257U);
//  uint64x2_t multiplier2 = vdupq_n_u64(14933078535860113213U);
//  uint64x2_t multiplier3 = vdupq_n_u64(15244667743933553977U);
//  uint64x2_t multiplier4 = vdupq_n_u64(9223367638806167551U);
//  uint64x2_t oneVec = vdupq_n_u64(1);
//
//  size_t numChunks =
//      numElements / 2; // NEON processes 2x64-bit integers at a time
//  size_t remaining = numElements % 2; // Handle remaining elements
//
//  for (size_t i = 0; i < numChunks; ++i) {
//    // Load 2x uint64_t elements from the input array
//    uint64x2_t keyVec = vld1q_u64(&inputArray[i * 2]);
//
//    // Scalar multiplication for keyVec
//    uint64_t key0 = vgetq_lane_u64(keyVec, 0);
//    uint64_t key1 = vgetq_lane_u64(keyVec, 1);
//
//    key0 *= 4611686016279904257U; // First scalar multiplication
//    key1 *= 4611686016279904257U;
//
//    // Set the updated values back into the NEON vector
//    keyVec = vsetq_lane_u64(key0, keyVec, 0);
//    keyVec = vsetq_lane_u64(key1, keyVec, 1);
//
//    // XOR shifts: key ^= (key >> 28) ^ (key >> 56)
//    uint64x2_t shift28 = vshrq_n_u64(keyVec, 28);
//    uint64x2_t shift56 = vshrq_n_u64(keyVec, 56);
//    keyVec = veorq_u64(keyVec, veorq_u64(shift28, shift56));
//
//    // Second scalar multiplication
//    key0 = vgetq_lane_u64(keyVec, 0);
//    key1 = vgetq_lane_u64(keyVec, 1);
//    key0 *= 14933078535860113213U;
//    key1 *= 14933078535860113213U;
//
//    // Set the updated values back into the NEON vector
//    keyVec = vsetq_lane_u64(key0, keyVec, 0);
//    keyVec = vsetq_lane_u64(key1, keyVec, 1);
//
//    // More XOR shifts and subsequent scalar multiplications, similar to the
//    // steps above
//    uint64x2_t shift14 = vshrq_n_u64(keyVec, 14);
//    shift28 = vshrq_n_u64(keyVec, 28);
//    uint64x2_t shift42 = vshrq_n_u64(keyVec, 42);
//    shift56 = vshrq_n_u64(keyVec, 56);
//    keyVec = veorq_u64(
//        keyVec,
//        veorq_u64(veorq_u64(shift14, shift28), veorq_u64(shift42, shift56)));
//
//    // Third scalar multiplication
//    key0 = vgetq_lane_u64(keyVec, 0);
//    key1 = vgetq_lane_u64(keyVec, 1);
//    key0 *= 15244667743933553977U;
//    key1 *= 15244667743933553977U;
//
//    // Set the updated values back into the NEON vector
//    keyVec = vsetq_lane_u64(key0, keyVec, 0);
//    keyVec = vsetq_lane_u64(key1, keyVec, 1);
//
//    // XOR shifts
//    uint64x2_t shift24 = vshrq_n_u64(keyVec, 24);
//    uint64x2_t shift48 = vshrq_n_u64(keyVec, 48);
//    keyVec = veorq_u64(keyVec, veorq_u64(shift24, shift48));
//
//    // Final scalar multiplication
//    key0 = vgetq_lane_u64(keyVec, 0);
//    key1 = vgetq_lane_u64(keyVec, 1);
//    key0 = (key0 + 1) * 9223367638806167551U;
//    key1 = (key1 + 1) * 9223367638806167551U;
//
//    // Set the final values back into the NEON vector
//    keyVec = vsetq_lane_u64(key0, keyVec, 0);
//    keyVec = vsetq_lane_u64(key1, keyVec, 1);
//
//    // Store the result back into the output array
//    vst1q_u64(&outputArray[i * 2], keyVec);
//  }
//}
//
//// AVX2-optimized function to process four 64-bit integers at once
// void xxhash64_avx2(
//     const uint64_t* inputArray,
//     size_t numElements,
//     uint64_t* outputArray,
//     uint64_t seed = 0) {
//   // Calculate the number of full 256-bit chunks (each containing 4 x 64-bit
//   // integers)
//   size_t numChunks = numElements / 4;
//   size_t remaining = numElements % 4;
//
//   // Constants loaded into AVX2 registers
//   __m256i prime1 = _mm256_set1_epi64x(PRIME64_1);
//   __m256i prime2 = _mm256_set1_epi64x(PRIME64_2);
//   __m256i prime4 = _mm256_set1_epi64x(PRIME64_4);
//   __m256i seedVec = _mm256_set1_epi64x(seed + PRIME64_5 + 8);
//
//   // Process four elements at a time
//   for (size_t i = 0; i < numChunks; ++i) {
//     // Load 4 x uint64_t elements from the input array
//     __m256i dataVec = _mm256_loadu_si256((__m256i const*)&inputArray[i * 4]);
//
//     // First step: multiply by PRIME64_2
//     __m256i k1 = _mm256_mullo_epi64(dataVec, prime2);
//
//     // Rotate left by 31 bits (AVX2 doesn't have a native rotate instruction,
//     so
//     // we shift)
//     __m256i k1Shift1 = _mm256_slli_epi64(k1, 31); // Left shift by 31
//     __m256i k1Shift2 =
//         _mm256_srli_epi64(k1, 64 - 31); // Right shift by(64 - 31)
//     k1 = _mm256_or_si256(k1Shift1, k1Shift2);
//     // Combine the two to simulate a rotate left
//
//     // Multiply by PRIME64_1
//     k1 = _mm256_mullo_epi64(k1, prime1);
//
//     // XOR the result into the hash
//     __m256i h64 = _mm256_xor_si256(seedVec, k1);
//
//     // Final mixing: rotate left by 27 bits and mix
//     __m256i h64Shift1 = _mm256_slli_epi64(h64, 27);
//     __m256i h64Shift2 = _mm256_srli_epi64(h64, 64 - 27);
//     h64 = _mm256_or_si256(h64Shift1, h64Shift2);
//
//     h64 = _mm256_mullo_epi64(h64, prime1);
//     h64 = _mm256_add_epi64(h64, prime4);
//
//     // Store the result into the output array
//     _mm256_storeu_si256((__m256i*)&outputArray[i * 4], h64);
//   }
//
//   // Process any remaining elements that couldn't fit into a 256-bit chunk
//   for (size_t i = numChunks * 4; i < numElements; ++i) {
//     // Fallback to the scalar version for the remaining elements
//     uint64_t h64 =
//         seed + PRIME64_5 + 8; // Initialize hash with seed and length(8
//         bytes)
//     uint64_t k1 = inputArray[i];
//     k1 *= PRIME64_2;
//     k1 = (k1 << 31) | (k1 >> (64 - 31));
//     k1 *= PRIME64_1;
//     h64 ^= k1;
//     h64 = ((h64 << 27) | (h64 >> (64 - 27))) * PRIME64_1 + PRIME64_4;
//     h64 ^= h64 >> 33;
//     h64 *= PRIME64_2;
//     h64 ^= h64 >> 29;
//     h64 *= PRIME64_3;
//     h64 ^= h64 >> 32;
//     outputArray[i] = h64;
//   }
// }
//
// inline uint64_t
// xxhash64(const uint64_t* input, size_t length, uint64_t seed = 0) {
//   const uint8_t* data = reinterpret_cast<const uint8_t*>(input);
//   const uint8_t* end = data + length;
//   uint64_t h64;
//
//   if (length >= 32) {
//     uint64_t v1 = seed + PRIME64_1 + PRIME64_2;
//     uint64_t v2 = seed + PRIME64_2;
//     uint64_t v3 = seed + 0;
//     uint64_t v4 = seed - PRIME64_1;
//
//     const uint8_t* limit = end - 32;
//     do {
//       v1 += *reinterpret_cast<const uint64_t*>(data) * PRIME64_2;
//       v1 = (v1 << 31) | (v1 >> (64 - 31));
//       v1 *= PRIME64_1;
//       data += 8;
//
//       v2 += *reinterpret_cast<const uint64_t*>(data) * PRIME64_2;
//       v2 = (v2 << 31) | (v2 >> (64 - 31));
//       v2 *= PRIME64_1;
//       data += 8;
//
//       v3 += *reinterpret_cast<const uint64_t*>(data) * PRIME64_2;
//       v3 = (v3 << 31) | (v3 >> (64 - 31));
//       v3 *= PRIME64_1;
//       data += 8;
//
//       v4 += *reinterpret_cast<const uint64_t*>(data) * PRIME64_2;
//       v4 = (v4 << 31) | (v4 >> (64 - 31));
//       v4 *= PRIME64_1;
//       data += 8;
//     } while (data <= limit);
//
//     h64 = ((v1 << 1) | (v1 >> (64 - 1))) + ((v2 << 7) | (v2 >> (64 - 7))) +
//         ((v3 << 12) | (v3 >> (64 - 12))) + ((v4 << 18) | (v4 >> (64 - 18)));
//
//     v1 *= PRIME64_2;
//     v1 = (v1 << 31) | (v1 >> (64 - 31));
//     v1 *= PRIME64_1;
//     h64 ^= v1;
//     h64 = h64 * PRIME64_1 + PRIME64_4;
//
//     v2 *= PRIME64_2;
//     v2 = (v2 << 31) | (v2 >> (64 - 31));
//     v2 *= PRIME64_1;
//     h64 ^= v2;
//     h64 = h64 * PRIME64_1 + PRIME64_4;
//
//     v3 *= PRIME64_2;
//     v3 = (v3 << 31) | (v3 >> (64 - 31));
//     v3 *= PRIME64_1;
//     h64 ^= v3;
//     h64 = h64 * PRIME64_1 + PRIME64_4;
//
//     v4 *= PRIME64_2;
//     v4 = (v4 << 31) | (v4 >> (64 - 31));
//     v4 *= PRIME64_1;
//     h64 ^= v4;
//     h64 = h64 * PRIME64_1 + PRIME64_4;
//   } else {
//     h64 = seed + PRIME64_5;
//   }
//
//   h64 += length;
//
//   while (data + 8 <= end) {
//     uint64_t k1 = *reinterpret_cast<const uint64_t*>(data);
//     k1 *= PRIME64_2;
//     k1 = (k1 << 31) | (k1 >> (64 - 31));
//     k1 *= PRIME64_1;
//     h64 ^= k1;
//     h64 = ((h64 << 27) | (h64 >> (64 - 27))) * PRIME64_1 + PRIME64_4;
//     data += 8;
//   }
//
//   while (data < end) {
//     h64 ^= (*data) * PRIME64_5;
//     h64 = ((h64 << 11) | (h64 >> (64 - 11))) * PRIME64_1;
//     data++;
//   }
//
//   h64 ^= h64 >> 33;
//   h64 *= PRIME64_2;
//   h64 ^= h64 >> 29;
//   h64 *= PRIME64_3;
//   h64 ^= h64 >> 32;
//
//   return h64;
// }
//
// constexpr uint64_t twang_unmix64(uint64_t key) noexcept {
//   // See the comments in jenkins_rev_unmix32 for an explanation as to how
//   this
//   // was generated
//   key *= 4611686016279904257U;
//   key ^= (key >> 28) ^ (key >> 56);
//   key *= 14933078535860113213U;
//   key ^= (key >> 14) ^ (key >> 28) ^ (key >> 42) ^ (key >> 56);
//   key *= 15244667743933553977U;
//   key ^= (key >> 24) ^ (key >> 48);
//   key = (key + 1) * 9223367638806167551U;
//   return key;
// }
//
//// AVX2 optimized version to process four 64 - bit integers at once
// void twang_unmix64_avx2(
//     const uint64_t* inputArray,
//     size_t numElements,
//     uint64_t* outputArray) {
//   // Constants used in twang_unmix64, packed into AVX2 registers
//   __m256i multiplier1 = _mm256_set1_epi64x(4611686016279904257U);
//   __m256i multiplier2 = _mm256_set1_epi64x(14933078535860113213U);
//   __m256i multiplier3 = _mm256_set1_epi64x(15244667743933553977U);
//   __m256i multiplier4 = _mm256_set1_epi64x(9223367638806167551U);
//   __m256i oneVec = _mm256_set1_epi64x(1);
//
//   size_t numChunks =
//       numElements / 4; // Number of full 256-bit chunks (4 x 64 - bit
//       integers)
//   size_t remaining =
//       numElements % 4; // Remaining elements that don’t fit in a chunk
//
//   for (size_t i = 0; i < numChunks; ++i) {
//     // Load 4 x uint64_t elements from the input array
//     __m256i keyVec = _mm256_loadu_si256((__m256i const*)&inputArray[i * 4]);
//
//     // Apply the twang_unmix64 transformation step-by-step
//
//     // First multiplication: key *= 4611686016279904257U
//     keyVec = _mm256_mullo_epi64(keyVec, multiplier1);
//
//     // First XOR shifts: key ^= (key >> 28) ^ (key >> 56)
//     __m256i shift28 = _mm256_srli_epi64(keyVec, 28);
//     __m256i shift56 = _mm256_srli_epi64(keyVec, 56);
//     keyVec = _mm256_xor_si256(keyVec, _mm256_xor_si256(shift28, shift56));
//
//     // Second multiplication: key *= 14933078535860113213U
//     keyVec = _mm256_mullo_epi64(keyVec, multiplier2);
//
//     // Second XOR shifts: key ^= (key >> 14) ^ (key >> 28) ^ (key >> 42) ^
//     (key
//     // >> 56)
//     __m256i shift14 = _mm256_srli_epi64(keyVec, 14);
//     shift28 = _mm256_srli_epi64(keyVec, 28);
//     __m256i shift42 = _mm256_srli_epi64(keyVec, 42);
//     shift56 = _mm256_srli_epi64(keyVec, 56);
//     keyVec = _mm256_xor_si256(
//         keyVec,
//         _mm256_xor_si256(
//             _mm256_xor_si256(shift14, shift28),
//             _mm256_xor_si256(shift42, shift56)));
//
//     // Third multiplication: key *= 15244667743933553977U
//     keyVec = _mm256_mullo_epi64(keyVec, multiplier3);
//
//     // Third XOR shifts: key ^= (key >> 24) ^ (key >> 48)
//     __m256i shift24 = _mm256_srli_epi64(keyVec, 24);
//     __m256i shift48 = _mm256_srli_epi64(keyVec, 48);
//     keyVec = _mm256_xor_si256(keyVec, _mm256_xor_si256(shift24, shift48));
//
//     // Final addition and multiplication: key = (key + 1)
//     *9223367638806167551U keyVec = _mm256_add_epi64(keyVec, oneVec); keyVec =
//     _mm256_mullo_epi64(keyVec, multiplier4);
//
//     // Store the result into the output array
//     _mm256_storeu_si256((__m256i*)&outputArray[i * 4], keyVec);
//   }
//
//   // Handle remaining elements that don't fit into a full 256-bit chunk
//   for (size_t i = numChunks * 4; i < numElements; ++i) {
//     outputArray[i] = twang_unmix64(
//         inputArray[i]); // Use scalar version for the remaining elements
//   }
// }

} // namespace

void VectorHasher::xx64hash(bool mix, raw_vector<uint64_t>& result) {
  auto numValues = decoded_.size();
  if (typeKind_ == TypeKind::UNKNOWN) {
    if (mix) {
      for (int i = 0; i < numValues; i++) {
        result[i] = bits::hashMix(result[i], kNullHash);
      }
    } else {
      std::fill(result.begin(), result.end(), kNullHash);
    }
  } else {
    VELOX_DYNAMIC_TYPE_DISPATCH(xx64hashValues, typeKind_, mix, result.data());
  }
}

template <TypeKind Kind>
void VectorHasher::xx64hashValues(bool mix, uint64_t* result) {
  using T = typename TypeTraits<Kind>::NativeType;
  if (decoded_.isConstantMapping()) {
    VELOX_NYI();
  } else if (!decoded_.isIdentityMapping()) {
    VELOX_NYI();
  } else {
    xx64hashFlatValues<T>(mix, result);
  }
}

template <typename T>
void VectorHasher::xx64hashFlatValues(bool mix, uint64_t* result) {
  // Check if T is void or an unsupported type and skip or handle differently
  if constexpr (std::is_void_v<T>) {
    VELOX_NYI(); // Handle or throw error for unsupported types
  } else {
    const T* values = decoded_.data<T>();
    auto numValues = decoded_.size();

    if (mix) {
      for (int i = 0; i < numValues; i++) {
        auto hash = xxhash64<T>(values[i]);
        result[i] = bits::hashMix(result[i], hash);
      }
    } else {
      for (int i = 0; i < numValues; i++) {
        result[i] = xxhash64<T>(values[i]);
      }
    }

    if (decoded_.mayHaveNulls()) {
      if (mix) {
        for (int i = 0; i < numValues; i++) {
          if (decoded_.isNullAt(i)) {
            result[i] = bits::hashMix(result[i], kNullHash);
          }
        }
      } else {
        for (int i = 0; i < numValues; i++) {
          if (decoded_.isNullAt(i)) {
            result[i] = kNullHash;
          }
        }
      }
    }
  }
}

void VectorHasher::hash(bool mix, raw_vector<uint64_t>& result) {
  auto numValues = decoded_.size();
  if (typeKind_ == TypeKind::UNKNOWN) {
    if (mix) {
      for (int i = 0; i < numValues; i++) {
        result[i] = bits::hashMix(result[i], kNullHash);
      }
    } else {
      std::fill(result.begin(), result.end(), kNullHash);
    }
  } else {
    VELOX_DYNAMIC_TYPE_DISPATCH(hashValues, typeKind_, mix, result.data());
  }
}

template <TypeKind Kind>
void VectorHasher::hashValues(bool mix, uint64_t* result) {
  using T = typename TypeTraits<Kind>::NativeType;
  if (decoded_.isConstantMapping()) {
    VELOX_NYI();
  } else if (!decoded_.isIdentityMapping()) {
    VELOX_NYI();
  } else {
    hashFlatValues<T>(mix, result);
  }
}

template <typename T>
void VectorHasher::hashFlatValues(bool mix, uint64_t* result) {
  // Check if T is void or an unsupported type and skip or handle differently
  if constexpr (std::is_void_v<T>) {
    VELOX_NYI(); // Handle or throw error for unsupported types
  } else {
    const T* values = decoded_.data<T>();
    auto numValues = decoded_.size();
    folly::hasher<T> hasher; // Ensure T is not void or unsupported

    if (mix) {
      for (int i = 0; i < numValues; i++) {
        auto hash = hasher(values[i]);
        result[i] = bits::hashMix(result[i], hash);
      }
    } else {
      for (int i = 0; i < numValues; i++) {
        result[i] = hasher(values[i]);
      }
    }

    if (decoded_.mayHaveNulls()) {
      if (mix) {
        for (int i = 0; i < numValues; i++) {
          if (decoded_.isNullAt(i)) {
            result[i] = bits::hashMix(result[i], kNullHash);
          }
        }
      } else {
        for (int i = 0; i < numValues; i++) {
          if (decoded_.isNullAt(i)) {
            result[i] = kNullHash;
          }
        }
      }
    }
  }
}

void VectorHasher::hashPrecomputed(bool mix, raw_vector<uint64_t>& result)
    const {
  VELOX_NYI();
}
//
//
// void VectorHasher::hashNeon(bool mix, raw_vector<uint64_t>& result) {
//  auto numValues = decoded_.size();
//  if (typeKind_ == TypeKind::UNKNOWN) {
//    if (mix) {
//      for (int i = 0; i < numValues; i++) {
//        result[i] = bits::hashMix(result[i], kNullHash);
//      }
//    } else {
//      std::fill(result.begin(), result.end(), kNullHash);
//    }
//  } else {
//    VELOX_DYNAMIC_TYPE_DISPATCH(hashValuesNeon, typeKind_, mix,
//    result.data());
//  }
//}
//
// template <TypeKind Kind>
// void VectorHasher::hashValuesNeon(bool mix, uint64_t* result) {
//  using T = typename TypeTraits<Kind>::NativeType;
//  if (decoded_.isConstantMapping()) {
//    VELOX_NYI();
//  } else if (!decoded_.isIdentityMapping()) {
//    VELOX_NYI();
//  } else {
//    hashFlatValuesNeon<T>(mix, result);
//  }
//}

// template <typename T>
// void VectorHasher::hashFlatValuesNeon(bool mix, uint64_t* result) {
//   // Check if T is void or an unsupported type and skip or handle differently
//   if constexpr (std::is_void_v<T>) {
//     VELOX_NYI(); // Handle or throw error for unsupported types
//   } else {
//     const T* values = decoded_.data<T>();
//     auto numValues = decoded_.size();
//
//     twang_unmix64_neon((const uint64_t*)values, numValues, result);
//     if (mix) {
//       for (int i = 0; i < numValues; i++) {
//         result[i] = bits::hashMix(result[i], result[i]);
//       }
//     }
//
//     if (decoded_.mayHaveNulls()) {
//       if (mix) {
//         for (int i = 0; i < numValues; i++) {
//           if (decoded_.isNullAt(i)) {
//             result[i] = bits::hashMix(result[i], kNullHash);
//           }
//         }
//       } else {
//         for (int i = 0; i < numValues; i++) {
//           if (decoded_.isNullAt(i)) {
//             result[i] = kNullHash;
//           }
//         }
//       }
//     }
//   }
// }

template <bool typeProvidesCustomComparison, TypeKind Kind>
void VectorHasher::hashValues(
    const SelectivityVector& rows,
    bool mix,
    uint64_t* result) {
  if (decoded_.isConstantMapping()) {
    auto hash = decoded_.isNullAt(rows.begin())
        ? kNullHash
        : hashOne<typeProvidesCustomComparison, Kind>(decoded_, rows.begin());
    rows.applyToSelected([&](vector_size_t row) {
      result[row] = mix ? bits::hashMix(result[row], hash) : hash;
    });
  } else if (
      !decoded_.isIdentityMapping() &&
      rows.countSelected() > decoded_.base()->size()) {
    cachedHashes_.resize(decoded_.base()->size());
    std::fill(cachedHashes_.begin(), cachedHashes_.end(), kNullHash);
    rows.applyToSelected([&](vector_size_t row) {
      if (decoded_.isNullAt(row)) {
        result[row] = mix ? bits::hashMix(result[row], kNullHash) : kNullHash;
        return;
      }
      auto baseIndex = decoded_.index(row);
      uint64_t hash = cachedHashes_[baseIndex];
      if (hash == kNullHash) {
        hash = hashOne<typeProvidesCustomComparison, Kind>(decoded_, row);
        cachedHashes_[baseIndex] = hash;
      }
      result[row] = mix ? bits::hashMix(result[row], hash) : hash;
    });
  } else {
    rows.applyToSelected([&](vector_size_t row) {
      if (decoded_.isNullAt(row)) {
        result[row] = mix ? bits::hashMix(result[row], kNullHash) : kNullHash;
        return;
      }
      auto hash = hashOne<typeProvidesCustomComparison, Kind>(decoded_, row);
      result[row] = mix ? bits::hashMix(result[row], hash) : hash;
    });
  }
}

template <TypeKind Kind>
bool VectorHasher::makeValueIds(
    const SelectivityVector& rows,
    uint64_t* result) {
  using T = typename TypeTraits<Kind>::NativeType;

  if (decoded_.isConstantMapping()) {
    uint64_t id = decoded_.isNullAt(rows.begin())
        ? 0
        : valueId(decoded_.valueAt<T>(rows.begin()));
    if (id == kUnmappable) {
      analyzeValue(decoded_.valueAt<T>(rows.begin()));
      return false;
    }
    rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
      result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
    });
    return true;
  }

  if (decoded_.isIdentityMapping()) {
    if (decoded_.mayHaveNulls()) {
      return makeValueIdsFlatWithNulls<T>(rows, result);
    } else {
      return makeValueIdsFlatNoNulls<T>(rows, result);
    }
  }

  if (decoded_.mayHaveNulls()) {
    return makeValueIdsDecoded<T, true>(rows, result);
  } else {
    return makeValueIdsDecoded<T, false>(rows, result);
  }
}

template <>
bool VectorHasher::makeValueIdsFlatNoNulls<bool>(
    const SelectivityVector& rows,
    uint64_t* result) {
  const auto* values = decoded_.data<uint64_t>();
  rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
    bool value = bits::isBitSet(values, row);
    uint64_t id = valueId(value);
    result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
  });
  return true;
}

template <>
bool VectorHasher::makeValueIdsFlatWithNulls<bool>(
    const SelectivityVector& rows,
    uint64_t* result) {
  const auto* values = decoded_.data<uint64_t>();
  const auto* nulls = decoded_.nulls(&rows);
  rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
    if (bits::isBitNull(nulls, row)) {
      if (multiplier_ == 1) {
        result[row] = 0;
      }
      return;
    }
    bool value = bits::isBitSet(values, row);
    uint64_t id = valueId(value);
    result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
  });
  return true;
}

template <typename T, bool mayHaveNulls>
void VectorHasher::makeValueIdForOneRow(
    const uint64_t* nulls,
    vector_size_t row,
    const T* values,
    vector_size_t valueRow,
    uint64_t* result,
    bool& success) {
  if constexpr (mayHaveNulls) {
    if (bits::isBitNull(nulls, row)) {
      if (multiplier_ == 1) {
        result[row] = 0;
      }
      return;
    }
  }
  T value = values[valueRow];
  if (!success) {
    analyzeValue(value);
    return;
  }
  auto id = valueId(value);
  if (id == kUnmappable) {
    success = false;
    analyzeValue(value);
  } else {
    result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
  }
}

template <typename T>
bool VectorHasher::makeValueIdsFlatNoNulls(
    const SelectivityVector& rows,
    uint64_t* result) {
  const auto* values = decoded_.data<T>();
  if (isRange_ && tryMapToRange(values, rows, result)) {
    return true;
  }

  bool success = true;
  rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
    makeValueIdForOneRow<T, false>(nullptr, row, values, row, result, success);
  });

  return success;
}

template <typename T>
bool VectorHasher::makeValueIdsFlatWithNulls(
    const SelectivityVector& rows,
    uint64_t* result) {
  const auto* values = decoded_.data<T>();
  const auto* nulls = decoded_.nulls(&rows);

  bool success = true;
  rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
    makeValueIdForOneRow<T, true>(nulls, row, values, row, result, success);
  });
  return success;
}

template <typename T, bool mayHaveNulls>
bool VectorHasher::makeValueIdsDecoded(
    const SelectivityVector& rows,
    uint64_t* result) {
  auto indices = decoded_.indices();
  auto values = decoded_.data<T>();
  bool success = true;

  if (rows.countSelected() <= decoded_.base()->size()) {
    // Cache is not beneficial in this case and we don't use them.
    auto* nulls = decoded_.nulls(&rows);
    rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
      makeValueIdForOneRow<T, mayHaveNulls>(
          nulls, row, values, indices[row], result, success);
    });
    return success;
  }

  cachedHashes_.resize(decoded_.base()->size());
  std::fill(cachedHashes_.begin(), cachedHashes_.end(), 0);

  int numCachedHashes = 0;
  rows.testSelected([&](vector_size_t row) INLINE_LAMBDA {
    if constexpr (mayHaveNulls) {
      if (decoded_.isNullAt(row)) {
        if (multiplier_ == 1) {
          result[row] = 0;
        }
        return true;
      }
    }

    auto baseIndex = indices[row];
    uint64_t& id = cachedHashes_[baseIndex];

    if (success) {
      if (id == 0) {
        T value = values[baseIndex];
        id = valueId(value);
        numCachedHashes++;
        if (id == kUnmappable) {
          analyzeValue(value);
          success = false;
        }
      }
      result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
    } else {
      if (id == 0) {
        id = kUnmappable;
        numCachedHashes++;
        analyzeValue(values[baseIndex]);
      }
    }

    return success || numCachedHashes < cachedHashes_.size();
  });

  return success;
}

template <>
bool VectorHasher::makeValueIdsDecoded<bool, true>(
    const SelectivityVector& rows,
    uint64_t* result) {
  auto indices = decoded_.indices();
  auto values = decoded_.data<uint64_t>();

  rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
    if (decoded_.isNullAt(row)) {
      if (multiplier_ == 1) {
        result[row] = 0;
      }
      return;
    }

    bool value = bits::isBitSet(values, indices[row]);
    auto id = valueId(value);
    result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
  });
  return true;
}

template <>
bool VectorHasher::makeValueIdsDecoded<bool, false>(
    const SelectivityVector& rows,
    uint64_t* result) {
  auto indices = decoded_.indices();
  auto values = decoded_.data<uint64_t>();

  rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
    bool value = bits::isBitSet(values, indices[row]);
    auto id = valueId(value);
    result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
  });
  return true;
}

bool VectorHasher::computeValueIds(
    const SelectivityVector& rows,
    raw_vector<uint64_t>& result) {
  checkTypeSupportsValueIds();

  return VALUE_ID_TYPE_DISPATCH(makeValueIds, typeKind_, rows, result.data());
}

bool VectorHasher::computeValueIdsForRows(
    char** groups,
    int32_t numGroups,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask,
    raw_vector<uint64_t>& result) {
  checkTypeSupportsValueIds();

  return VALUE_ID_TYPE_DISPATCH(
      makeValueIdsForRows,
      typeKind_,
      groups,
      numGroups,
      offset,
      nullByte,
      nullMask,
      result.data());
}

template <>
bool VectorHasher::makeValueIdsForRows<TypeKind::VARCHAR>(
    char** groups,
    int32_t numGroups,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask,
    uint64_t* result) {
  for (int32_t i = 0; i < numGroups; ++i) {
    if (isNullAt(groups[i], nullByte, nullMask)) {
      if (multiplier_ == 1) {
        result[i] = 0;
      }
    } else {
      std::string storage;
      auto id = valueId<StringView>(HashStringAllocator::contiguousString(
          valueAt<StringView>(groups[i], offset), storage));
      if (id == kUnmappable) {
        return false;
      }
      result[i] = multiplier_ == 1 ? id : result[i] + multiplier_ * id;
    }
  }
  return true;
}

template <TypeKind Kind>
void VectorHasher::lookupValueIdsTyped(
    const DecodedVector& decoded,
    SelectivityVector& rows,
    raw_vector<uint64_t>& hashes,
    uint64_t* result) const {
  using T = typename TypeTraits<Kind>::NativeType;
  if (decoded.isConstantMapping()) {
    if (decoded.isNullAt(rows.begin())) {
      if (multiplier_ == 1) {
        rows.applyToSelected([&](vector_size_t row)
                                 INLINE_LAMBDA { result[row] = 0; });
      }
      return;
    }

    uint64_t id = lookupValueId(decoded.valueAt<T>(rows.begin()));
    if (id == kUnmappable) {
      rows.clearAll();
    } else {
      rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
        result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
      });
    }
  } else if (decoded.isIdentityMapping()) {
    if (Kind == TypeKind::BIGINT && isRange_) {
      lookupIdsRangeSimd<int64_t>(decoded, rows, result);
    } else if (Kind == TypeKind::INTEGER && isRange_) {
      lookupIdsRangeSimd<int32_t>(decoded, rows, result);
    } else {
      rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
        if (decoded.isNullAt(row)) {
          if (multiplier_ == 1) {
            result[row] = 0;
          }
          return;
        }
        T value = decoded.valueAt<T>(row);
        uint64_t id = lookupValueId(value);
        if (id == kUnmappable) {
          rows.setValid(row, false);
          return;
        }
        result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
      });
    }
    rows.updateBounds();
  } else {
    hashes.resize(decoded.base()->size());
    std::fill(hashes.begin(), hashes.end(), 0);
    rows.applyToSelected([&](vector_size_t row) INLINE_LAMBDA {
      if (decoded.isNullAt(row)) {
        if (multiplier_ == 1) {
          result[row] = 0;
        }
        return;
      }
      auto baseIndex = decoded.index(row);
      uint64_t id = hashes[baseIndex];
      if (id == 0) {
        T value = decoded.valueAt<T>(row);
        id = lookupValueId(value);
        if (id == kUnmappable) {
          rows.setValid(row, false);
          return;
        }
        hashes[baseIndex] = id;
      }
      result[row] = multiplier_ == 1 ? id : result[row] + multiplier_ * id;
    });
    rows.updateBounds();
  }
}

template <typename T>
void VectorHasher::lookupIdsRangeSimd(
    const DecodedVector& decoded,
    SelectivityVector& rows,
    uint64_t* result) const {
  static_assert(sizeof(T) == 8 || sizeof(T) == 4);
  auto lower = xsimd::batch<T>::broadcast(min_);
  auto upper = xsimd::batch<T>::broadcast(max_);
  auto data = decoded.data<T>();
  uint64_t offset = min_ - 1;
  auto bits = rows.asMutableRange().bits();
  bits::forBatches<xsimd::batch<T>::size>(
      bits, rows.begin(), rows.end(), [&](auto index, auto /*mask*/) {
        auto values = xsimd::batch<T>::load_unaligned(data + index);
        uint64_t outOfRange =
            simd::toBitMask(lower > values) | simd::toBitMask(values > upper);
        if (outOfRange) {
          bits[index / 64] &= ~(outOfRange << (index & 63));
        }
        if (outOfRange != bits::lowMask(xsimd::batch<T>::size)) {
          if constexpr (sizeof(T) == 8) {
            auto unsignedValues =
                simd::reinterpretBatch<typename std::make_unsigned<T>::type>(
                    values);
            if (multiplier_ == 1) {
              (unsignedValues - offset).store_unaligned(result + index);
            } else {
              (xsimd::batch<uint64_t>::load_unaligned(result + index) +
               ((unsignedValues - offset) * multiplier_))
                  .store_unaligned(result + index);
            }
          } else {
            // Widen 8 to 2 x 4 since result is always 64 bits wide.
            auto first4 = simd::reinterpretBatch<uint64_t>(
                              simd::getHalf<int64_t, 0>(values)) -
                offset;
            auto next4 = simd::reinterpretBatch<uint64_t>(
                             simd::getHalf<int64_t, 1>(values)) -
                offset;
            if (multiplier_ == 1) {
              first4.store_unaligned(result + index);
              next4.store_unaligned(result + index + first4.size);
            } else {
              (xsimd::batch<uint64_t>::load_unaligned(result + index) +
               (first4 * multiplier_))
                  .store_unaligned(result + index);
              (xsimd::batch<uint64_t>::load_unaligned(
                   result + index + first4.size) +
               (next4 * multiplier_))
                  .store_unaligned(result + index + first4.size);
            }
          }
        }
      });
}

void VectorHasher::lookupValueIds(
    const BaseVector& values,
    SelectivityVector& rows,
    ScratchMemory& scratchMemory,
    raw_vector<uint64_t>& result) const {
  checkTypeSupportsValueIds();

  scratchMemory.decoded.decode(values, rows);
  VALUE_ID_TYPE_DISPATCH(
      lookupValueIdsTyped,
      typeKind_,
      scratchMemory.decoded,
      rows,
      scratchMemory.hashes,
      result.data());
}

void VectorHasher::hash(
    const SelectivityVector& rows,
    bool mix,
    raw_vector<uint64_t>& result) {
  if (typeKind_ == TypeKind::UNKNOWN) {
    rows.applyToSelected([&](auto row) {
      result[row] = mix ? bits::hashMix(result[row], kNullHash) : kNullHash;
    });
  } else {
    if (typeProvidesCustomComparison_) {
      VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH(
          hashValues, true, typeKind_, rows, mix, result.data());
    } else {
      VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH(
          hashValues, false, typeKind_, rows, mix, result.data());
    }
  }
}

void VectorHasher::hashPrecomputed(
    const SelectivityVector& rows,
    bool mix,
    raw_vector<uint64_t>& result) const {
  rows.applyToSelected([&](vector_size_t row) {
    result[row] =
        mix ? bits::hashMix(result[row], precomputedHash_) : precomputedHash_;
  });
}

void VectorHasher::precompute(const BaseVector& value) {
  if (value.isNullAt(0)) {
    precomputedHash_ = kNullHash;
    return;
  }

  const SelectivityVector rows(1, true);
  decoded_.decode(value, rows);

  if (typeProvidesCustomComparison_) {
    precomputedHash_ = VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH(
        hashOne, true, typeKind_, decoded_, 0);
  } else {
    precomputedHash_ = VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH(
        hashOne, false, typeKind_, decoded_, 0);
  }
}

void VectorHasher::analyze(
    char** groups,
    int32_t numGroups,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask) {
  checkTypeSupportsValueIds();

  VALUE_ID_TYPE_DISPATCH(
      analyzeTyped, typeKind_, groups, numGroups, offset, nullByte, nullMask);
}

template <>
void VectorHasher::analyzeValue(StringView value) {
  int size = value.size();
  auto data = value.data();
  if (!rangeOverflow_) {
    if (size > kStringASRangeMaxSize) {
      setRangeOverflow();
    } else {
      int64_t number = stringAsNumber(data, size);
      updateRange(number);
    }
  }
  if (!distinctOverflow_) {
    UniqueValue unique(data, size);
    unique.setId(uniqueValues_.size() + 1);
    auto pair = uniqueValues_.insert(unique);
    if (pair.second) {
      if (uniqueValues_.size() > kMaxDistinct) {
        setDistinctOverflow();
        return;
      }
      copyStringToLocal(&*pair.first);
    }
  }
}

void VectorHasher::copyStringToLocal(const UniqueValue* unique) {
  auto size = unique->size();
  if (size <= sizeof(int64_t)) {
    return;
  }
  if (distinctStringsBytes_ > kMaxDistinctStringsBytes) {
    setDistinctOverflow();
    return;
  }
  if (uniqueValuesStorage_.empty()) {
    uniqueValuesStorage_.emplace_back();
    uniqueValuesStorage_.back().reserve(std::max(kStringBufferUnitSize, size));
    distinctStringsBytes_ += uniqueValuesStorage_.back().capacity();
  }
  auto str = &uniqueValuesStorage_.back();
  if (str->size() + size > str->capacity()) {
    uniqueValuesStorage_.emplace_back();
    uniqueValuesStorage_.back().reserve(std::max(kStringBufferUnitSize, size));
    distinctStringsBytes_ += uniqueValuesStorage_.back().capacity();
    str = &uniqueValuesStorage_.back();
  }
  auto start = str->size();
  str->resize(start + size);
  memcpy(str->data() + start, reinterpret_cast<char*>(unique->data()), size);
  const_cast<UniqueValue*>(unique)->setData(
      reinterpret_cast<int64_t>(str->data() + start));
}

void VectorHasher::setDistinctOverflow() {
  distinctOverflow_ = true;
  uniqueValues_.clear();
  uniqueValuesStorage_.clear();
  distinctStringsBytes_ = 0;
}

void VectorHasher::setRangeOverflow() {
  rangeOverflow_ = true;
  hasRange_ = false;
}

std::unique_ptr<common::Filter> VectorHasher::getFilter(
    bool nullAllowed) const {
  if (typeProvidesCustomComparison_) {
    return nullptr;
  }

  switch (typeKind_) {
    case TypeKind::TINYINT:
      [[fallthrough]];
    case TypeKind::SMALLINT:
      [[fallthrough]];
    case TypeKind::INTEGER:
      [[fallthrough]];
    case TypeKind::BIGINT:
      if (!distinctOverflow_) {
        std::vector<int64_t> values;
        values.reserve(uniqueValues_.size());
        for (const auto& value : uniqueValues_) {
          values.emplace_back(value.data());
        }

        return common::createBigintValues(values, nullAllowed);
      }
      [[fallthrough]];
    default:
      // TODO Add support for strings.
      return nullptr;
  }
}

namespace {
template <typename T>
// Adds 'reserve' to either end of the range between 'min' and 'max' while
// staying in the range of T.
void extendRange(int64_t reserve, int64_t& min, int64_t& max) {
  int64_t kMin = std::numeric_limits<T>::min();
  int64_t kMax = std::numeric_limits<T>::max();
  if (kMin + reserve + 1 > min) {
    min = kMin;
  } else {
    min -= reserve;
  }
  if (kMax - reserve < max) {
    max = kMax;
  } else {
    max += reserve;
  }
}

// Adds 'reservePct' % to either end of the range between 'min' and 'max'
// while staying in the range of 'kind'.
void extendRange(
    TypeKind kind,
    int32_t reservePct,
    int64_t& min,
    int64_t& max) {
  // The reserve is 2 + reservePct % of the range. Add 2 to make sure
  // that a non-0 peercentage actually adds something for a small
  // range.
  int64_t reserve =
      reservePct == 0 ? 0 : 2 + (max - min) * (reservePct / 100.0);
  switch (kind) {
    case TypeKind::BOOLEAN:
      break;
    case TypeKind::TINYINT:
      extendRange<int8_t>(reserve, min, max);
      break;
    case TypeKind::SMALLINT:
      extendRange<int16_t>(reserve, min, max);
      break;
    case TypeKind::INTEGER:
      extendRange<int32_t>(reserve, min, max);
      break;
    case TypeKind::BIGINT:
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
    case TypeKind::TIMESTAMP:
      extendRange<int64_t>(reserve, min, max);
      break;

    default:
      VELOX_FAIL("Unsupported VectorHasher typeKind {}", kind);
  }
}

int64_t addIdReserve(size_t numDistinct, int32_t reservePct) {
  // A merge of hashers in a hash join build may end up over the limit, so
  // return that.
  if (numDistinct > VectorHasher::kMaxDistinct) {
    return numDistinct;
  }
  if (reservePct == VectorHasher::kNoLimit) {
    return VectorHasher::kMaxDistinct;
  }
  // NOTE: 'kMaxDistinct' is a small value so no need to check overflow for
  // reservation here.
  return std::min<int64_t>(
      VectorHasher::kMaxDistinct, numDistinct * (1 + (reservePct / 100.0)));
}
} // namespace

void VectorHasher::cardinality(
    int32_t reservePct,
    uint64_t& asRange,
    uint64_t& asDistincts) {
  if (!typeSupportsValueIds()) {
    asRange = kRangeTooLarge;
    asDistincts = kRangeTooLarge;
    return;
  }

  if (typeKind_ == TypeKind::BOOLEAN) {
    hasRange_ = true;
    asRange = 3;
    asDistincts = 3;
    return;
  }
  int64_t signedRange;
  if (!hasRange_ || rangeOverflow_) {
    asRange = kRangeTooLarge;
  } else if (__builtin_sub_overflow(max_, min_, &signedRange)) {
    setRangeOverflow();
    asRange = kRangeTooLarge;
  } else if (signedRange < kMaxRange) {
    // We check that after the extension by reservePct the range of max - min
    // will still be in int64_t bounds.
    VELOX_CHECK_GE(100, reservePct);
    static_assert(kMaxRange < std::numeric_limits<uint64_t>::max() / 4);
    // We pad the range by 'reservePct'%, half below and half above,
    // while staying within bounds of the type. We do not pad the
    // limits yet, this is done only when enabling range mode.
    int64_t min = min_;
    int64_t max = max_;
    extendRange(type_->kind(), reservePct, min, max);
    asRange = (max - min) + 2;
  } else {
    setRangeOverflow();
    asRange = kRangeTooLarge;
  }
  if (distinctOverflow_) {
    asDistincts = kRangeTooLarge;
    return;
  }
  // Padded count of values + 1 for null.
  asDistincts = addIdReserve(uniqueValues_.size(), reservePct) + 1;
}

uint64_t VectorHasher::enableValueIds(uint64_t multiplier, int32_t reservePct) {
  VELOX_CHECK_NE(
      typeKind_,
      TypeKind::BOOLEAN,
      "A boolean VectorHasher should  always be by range");
  checkTypeSupportsValueIds();

  multiplier_ = multiplier;
  rangeSize_ = addIdReserve(uniqueValues_.size(), reservePct) + 1;
  isRange_ = false;
  uint64_t result;
  if (__builtin_mul_overflow(multiplier_, rangeSize_, &result)) {
    return kRangeTooLarge;
  }
  return result;
}

uint64_t VectorHasher::enableValueRange(
    uint64_t multiplier,
    int32_t reservePct) {
  checkTypeSupportsValueIds();

  multiplier_ = multiplier;
  VELOX_CHECK_LE(0, reservePct);
  VELOX_CHECK(hasRange_);
  extendRange(type_->kind(), reservePct, min_, max_);
  isRange_ = true;
  // No overflow because max range is under 63 bits.
  if (typeKind_ == TypeKind::BOOLEAN) {
    rangeSize_ = 3;
  } else {
    rangeSize_ = (max_ - min_) + 2;
  }
  uint64_t result;
  if (__builtin_mul_overflow(multiplier_, rangeSize_, &result)) {
    return kRangeTooLarge;
  }
  return result;
}

void VectorHasher::copyStatsFrom(const VectorHasher& other) {
  hasRange_ = other.hasRange_;
  rangeOverflow_ = other.rangeOverflow_;
  distinctOverflow_ = other.distinctOverflow_;

  min_ = other.min_;
  max_ = other.max_;
  uniqueValues_ = other.uniqueValues_;
}

void VectorHasher::merge(const VectorHasher& other, size_t maxNumDistinct) {
  if (typeKind_ == TypeKind::BOOLEAN) {
    return;
  }
  if (other.empty()) {
    return;
  }
  if (empty()) {
    copyStatsFrom(other);
    return;
  }
  if (hasRange_ && other.hasRange_ && !rangeOverflow_ &&
      !other.rangeOverflow_) {
    min_ = std::min(min_, other.min_);
    max_ = std::max(max_, other.max_);
  } else {
    setRangeOverflow();
  }
  if (distinctOverflow_) {
    return;
  }
  if (other.distinctOverflow_) {
    setDistinctOverflow();
    return;
  }
  // Unique values can be merged without dispatch on type. All the
  // merged hashers must stay live for string type columns.
  for (UniqueValue value : other.uniqueValues_) {
    // Assign a new id at end of range for the case 'value' is not
    // in 'uniqueValues_'. We do not set overflow here because the
    // memory is already allocated and there is a known cap on size.
    value.setId(uniqueValues_.size() + 1);
    if (uniqueValues_.insert(value).second &&
        uniqueValues_.size() > maxNumDistinct) {
      setDistinctOverflow();
      break;
    }
  }
}

std::string VectorHasher::toString() const {
  std::stringstream out;
  out << "VectorHasher channel: " << channel_ << " " << type_->toString()
      << " multiplier: " << multiplier_;
  if (isRange_) {
    out << " range size " << rangeSize_ << ": [" << min_ << ", " << max_ << "]";
  }
  if (!distinctOverflow_) {
    out << " numDistinct: " << uniqueValues_.size();
  }
  return out.str();
}

std::vector<std::unique_ptr<VectorHasher>> createVectorHashers(
    const RowTypePtr& rowType,
    const std::vector<core::FieldAccessTypedExprPtr>& keys) {
  const auto numKeys = keys.size();
  std::vector<column_index_t> keyChannels;
  keyChannels.reserve(numKeys);
  for (const auto& key : keys) {
    const auto channel = exprToChannel(key.get(), rowType);
    keyChannels.push_back(channel);
  }
  return createVectorHashers(rowType, keyChannels);
}

std::vector<std::unique_ptr<VectorHasher>> createVectorHashers(
    const RowTypePtr& rowType,
    const std::vector<column_index_t>& keyChannels) {
  const auto numKeys = keyChannels.size();
  std::vector<std::unique_ptr<VectorHasher>> hashers;
  hashers.reserve(numKeys);
  for (const auto& keyChannel : keyChannels) {
    hashers.push_back(
        VectorHasher::create(rowType->childAt(keyChannel), keyChannel));
  }
  return hashers;
}

} // namespace facebook::velox::exec

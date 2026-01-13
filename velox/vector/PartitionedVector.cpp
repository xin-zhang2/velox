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
#include "velox/vector/PartitionedVector.h"

#include "velox/vector/DictionaryVector.h"
#include "velox/vector/VectorTypeUtils.h"

namespace facebook::velox {

namespace {

inline void countPartitionSizes(
    const std::vector<uint32_t>& partitions,
    vector_size_t*& counts) {
  for (auto i = 0; i < partitions.size(); i++) {
    counts[partitions[i]]++;
  }
}

inline void countPartitionSizes(
    const std::vector<uint32_t>& topRowPartitions,
    vector_size_t*& topRowOffsets,
    vector_size_t lastTopRowOffset,
    vector_size_t numTopRows,
    vector_size_t*& counts) {
  for (auto topRow = 0; topRow < numTopRows - 1; topRow++) {
    auto p = topRowPartitions[topRow];
    counts[p] += topRowOffsets[topRow + 1] - topRowOffsets[topRow];
  }

  // The last topRow
  auto p = topRowPartitions[numTopRows - 1];
  counts[p] += lastTopRowOffset - topRowOffsets[numTopRows - 1];
}

template <typename T>
void partitionFixedWidthValuesInPlace(
    T*& values,
    const std::vector<uint32_t>& partitions,
    uint32_t numPartitions,
    vector_size_t*& beginPartitionOffsets,
    vector_size_t*& endPartitionOffsets) {
  for (auto partition = 0; partition < numPartitions; partition++) {
    auto& offset = beginPartitionOffsets[partition];
    auto endOffset = endPartitionOffsets[partition];
    while (offset < endOffset) {
      uint32_t p = partitions[offset];
      while (p != partition) {
        auto destinationOffset = beginPartitionOffsets[p]++;
        std::swap(values[destinationOffset], values[offset]);
        p = partitions[destinationOffset];
      }
      offset = ++beginPartitionOffsets[partition];
    }
  }
}

// This has to be called after endPartitionOffsetsBuffer is populated
void initializeBeginPartitionOffsets(
    BufferPtr& beginPartitionOffsetsBuffer,
    BufferPtr& endPartitionOffsetsBuffer,
    int32_t numPartitions,
    velox::memory::MemoryPool* pool) {
  ensureCapacity<vector_size_t>(
      beginPartitionOffsetsBuffer, numPartitions, pool);

  beginPartitionOffsetsBuffer->asMutable<vector_size_t>()[0] = 0;
  std::memcpy(
      &beginPartitionOffsetsBuffer->asMutable<vector_size_t>()[1],
      endPartitionOffsetsBuffer->asMutable<vector_size_t>(),
      sizeof(uint32_t) * (numPartitions - 1));
  beginPartitionOffsetsBuffer->setSize(numPartitions * sizeof(vector_size_t));
}

void swapBit(char& byte1, int8_t bit1, char& byte2, int8_t bit2) {
  // Calculate the difference between the bits
  char bitDiff = ((byte1 >> bit1) & 1) ^ ((byte2 >> bit2) & 1);

  // Apply the difference to toggle the bits
  byte1 ^= (bitDiff << bit1);
  byte2 ^= (bitDiff << bit2);
}

void partitionBitsInPlace(
    char*& bits,
    const std::vector<uint32_t>& partitions,
    uint32_t numPartitions,
    BufferPtr& beginPartitionOffsetsBuffer,
    BufferPtr& endPartitionOffsetsBuffer,
    velox::memory::MemoryPool* pool) {
  initializeBeginPartitionOffsets(
      beginPartitionOffsetsBuffer,
      endPartitionOffsetsBuffer,
      numPartitions,
      pool);

  auto beginPartitionOffsets =
      beginPartitionOffsetsBuffer->asMutable<vector_size_t>();
  auto endPartitionOffsets =
      endPartitionOffsetsBuffer->asMutable<vector_size_t>();

  for (auto partition = 0; partition < numPartitions; partition++) {
    auto& offset = beginPartitionOffsets[partition];
    auto endOffset = endPartitionOffsets[partition];
    while (offset < endOffset) {
      uint32_t p = partitions[offset];
      while (p != partition) {
        vector_size_t destinationOffset = beginPartitionOffsets[p]++;

        vector_size_t destinationAddr = destinationOffset / 8;
        int8_t destinationBitInByte = destinationOffset - destinationAddr * 8;
        vector_size_t fromAddr = offset / 8;
        int8_t fromBitInByte = offset - fromAddr * 8;

        swapBit(
            bits[destinationAddr],
            destinationBitInByte,
            bits[fromAddr],
            fromBitInByte);
        p = partitions[destinationOffset];
      }
      offset = ++beginPartitionOffsets[partition];
    }
  }
}

template <typename T>
void partitionFixedWidthValuesToOutput(
    T*& input,
    const std::vector<uint32_t>& topRowPartitions,
    vector_size_t*& topRowOffsets,
    vector_size_t lastTopRowOffset,
    vector_size_t*& beginPartitionOffsets,
    T*& output) {
  VELOX_CHECK(input != output);

  auto numTopRows = topRowPartitions.size();
  for (auto topRow = 0; topRow < numTopRows - 1; topRow++) {
    uint32_t partition = topRowPartitions[topRow];

    auto toBegin = beginPartitionOffsets[partition];
    auto fromBegin = topRowOffsets[topRow];
    auto fromEnd = topRowOffsets[topRow + 1];
    auto length = fromEnd - fromBegin;
    std::memcpy(output + toBegin, input + fromBegin, sizeof(T) * length);

    beginPartitionOffsets[partition] += length;
  }

  uint32_t partition = topRowPartitions[numTopRows - 1];
  auto toBegin = beginPartitionOffsets[partition];
  auto lastRowBegin = topRowOffsets[numTopRows - 1];
  auto lastRowEnd = lastTopRowOffset;
  std::memcpy(
      output + toBegin,
      input + lastRowBegin,
      (lastRowEnd - lastRowBegin) * sizeof(T));
}

template <typename T>
void partitionFixedWidthValues(
    BufferPtr& inputBuffer,
    int32_t numRows,
    const std::vector<uint32_t>& topRowPartitions,
    BufferPtr& topRowOffsetsBuffer,
    vector_size_t lastTopRowOffset,
    int32_t numPartitions,
    BufferPtr& beginPartitionOffsetsBuffer,
    BufferPtr& endPartitionOffsetsBuffer,
    BufferPtr& swappingBuffer,
    int32_t nestLevel,
    velox::memory::MemoryPool* pool,
    bool useSwapping = false) {
  auto input = inputBuffer->asMutable<T>();

  initializeBeginPartitionOffsets(
      beginPartitionOffsetsBuffer,
      endPartitionOffsetsBuffer,
      numPartitions,
      pool);

  auto beginPartitionOffsets =
      beginPartitionOffsetsBuffer->asMutable<vector_size_t>();
  auto endPartitionOffsets =
      endPartitionOffsetsBuffer->asMutable<vector_size_t>();

  if (nestLevel == 0) {
    // It's the top level so just partition in place
    partitionFixedWidthValuesInPlace<T>(
        input,
        topRowPartitions,
        numPartitions,
        beginPartitionOffsets,
        endPartitionOffsets);
  } else {
    ensureCapacity<T>(swappingBuffer, numRows, pool);
    auto buffer = swappingBuffer->asMutable<T>();
    auto topRowOffsets = topRowOffsetsBuffer->asMutable<vector_size_t>();
    partitionFixedWidthValuesToOutput<T>(
        input,
        topRowPartitions,
        topRowOffsets,
        lastTopRowOffset,
        beginPartitionOffsets,
        buffer);
    if (useSwapping) {
      std::swap(inputBuffer, swappingBuffer);
    } else {
      std::memcpy(input, buffer, numRows * sizeof(T));
    }
  }
}

template <TypeKind typeKind>
PartitionedVectorPtr createPartitionedFlatVector(
    VectorPtr vector,
    const std::vector<uint32_t>& topRowPartitions,
    BufferPtr& topRowOffsets,
    vector_size_t lastTopRowOffset,
    int32_t numPartitions,
    BufferPtr& beginPartitionOffsetsBuffer,
    BufferPtr& endPartitionOffsets,
    BufferPtr& swappingBuffer,
    int32_t nestLevel,
    velox::memory::MemoryPool* pool) {
  using T = typename TypeTraits<typeKind>::NativeType;
  auto flatVector = std::dynamic_pointer_cast<FlatVector<T>>(vector);
  if (!flatVector) {
    throw std::invalid_argument("Vector is not a FlatVector<T>");
  }

  auto partitionedFlatVector = std::make_shared<PartitionedFlatVector<T>>(
      flatVector, numPartitions, endPartitionOffsets, nullptr, pool);

  if (numPartitions > 1) {
    partitionedFlatVector->partition(
        topRowPartitions,
        topRowOffsets,
        lastTopRowOffset,
        beginPartitionOffsetsBuffer,
        swappingBuffer,
        nestLevel);
  }

  return partitionedFlatVector;
}

template <TypeKind typeKind>
PartitionedVectorPtr createWrappedPartitionedFlatVector(
    VectorPtr vector,
    int32_t numPartitions,
    BufferPtr& partitionOffsets,
    BufferPtr& partitionedIndices,
    velox::memory::MemoryPool* pool) {
  using T = typename TypeTraits<typeKind>::NativeType;
  auto flatVector = std::dynamic_pointer_cast<FlatVector<T>>(vector);
  if (!flatVector) {
    throw std::invalid_argument("Vector is not a FlatVector<T>");
  }

  auto partitionedFlatVector = std::make_shared<PartitionedFlatVector<T>>(
      flatVector, numPartitions, partitionOffsets, partitionedIndices, pool);

  return partitionedFlatVector;
}

template <TypeKind typeKind>
PartitionedVectorPtr createPartitionedVectorFromDictionary(
    VectorPtr vector,
    const std::vector<uint32_t>& topRowPartitions,
    BufferPtr& topRowOffsetsBuffer,
    vector_size_t lastTopRowOffset,
    int32_t numPartitions,
    BufferPtr& beginPartitionOffsetsBuffer,
    BufferPtr& endPartitionOffsetsBuffer,
    BufferPtr& swappingBuffer,
    int32_t nestLevel,
    velox::memory::MemoryPool* pool) {
  using T = typename KindToFlatVector<typeKind>::WrapperType;
  auto dictionaryVector = vector->as<DictionaryVector<T>>();
  if (!dictionaryVector) {
    throw std::invalid_argument("Vector is not a DictionaryVector<T>");
  }

  auto indicesBuffer = dictionaryVector->indices();
  auto& dictionaryValues = dictionaryVector->valueVector();

  if (numPartitions > 1) {
    partitionFixedWidthValues<vector_size_t>(
        indicesBuffer,
        vector->size(),
        topRowPartitions,
        topRowOffsetsBuffer,
        lastTopRowOffset,
        numPartitions,
        beginPartitionOffsetsBuffer,
        endPartitionOffsetsBuffer,
        swappingBuffer,
        nestLevel,
        pool);
  }

  auto partitionedVector = PartitionedVector::createWrapped(
      const_cast<VectorPtr&>(dictionaryValues),
      topRowPartitions,
      topRowOffsetsBuffer,
      lastTopRowOffset,
      numPartitions,
      beginPartitionOffsetsBuffer,
      endPartitionOffsetsBuffer,
      swappingBuffer,
      indicesBuffer,
      nestLevel + 1,
      pool);

  return partitionedVector;
}

template <TypeKind typeKind>
PartitionedVectorPtr remapIdsAndCreatePartitionedVector(
    VectorPtr vector,
    const std::vector<uint32_t>& topRowPartitions,
    BufferPtr& upperLevelOffsetsBuffer,
    vector_size_t upperLevelLastOffset,
    int32_t numPartitions,
    BufferPtr& beginPartitionOffsetsBuffer,
    BufferPtr& endPartitionOffsetsBuffer,
    BufferPtr& swappingBuffer,
    BufferPtr& upperLevelIndicesBuffer,
    int32_t nestLevel,
    velox::memory::MemoryPool* pool) {
  // upperLevelIndicesBuffer should have been allocated and calculated from the
  // upper level
  using T = typename KindToFlatVector<typeKind>::WrapperType;
  auto dictionaryVector = vector->as<DictionaryVector<T>>();
  if (!dictionaryVector) {
    throw std::invalid_argument("Vector is not a DictionaryVector<T>");
  }

  // Map the indices. The partition offsets won't change.
  // upperLevelIndicesBuffer is the upper level indices: {1,0,2,3,3,1,2,0}
  // upperLevelOffsetsBuffer
  // vector is a DictionaryVector and has 4 distinct rows.
  // indicesBuffer: {0,1,0,0}, dictionaryValues: {"cat","dog"}
  // New indicesBuffer: {1,0,0,0,0,1,0,0}, dictionaryValues: {"cat","dog"}
  auto outerRows = upperLevelIndicesBuffer->asMutable<vector_size_t>();
  auto innerRows =
      dictionaryVector->indices()->template asMutable<vector_size_t>();
  auto numIndicesMapped =
      upperLevelIndicesBuffer->size() / sizeof(vector_size_t);
  for (auto i = 0; i < numIndicesMapped; ++i) {
    outerRows[i] = innerRows[outerRows[i]];
  }

  auto& dictionaryValues = dictionaryVector->valueVector();
  auto partitioningVector = PartitionedVector::createWrapped(
      const_cast<VectorPtr&>(dictionaryValues),
      topRowPartitions,
      upperLevelOffsetsBuffer,
      upperLevelLastOffset,
      numPartitions,
      beginPartitionOffsetsBuffer,
      endPartitionOffsetsBuffer,
      swappingBuffer,
      upperLevelIndicesBuffer,
      nestLevel + 1,
      pool);

  return partitioningVector;
}

void preparePartitionOffsetsForNextLevel(
    int32_t numPartitions,
    memory::MemoryPool* pool,
    BufferPtr& partitionOffsetsForNextLevelBuffer) {
  ensureCapacity<vector_size_t>(
      partitionOffsetsForNextLevelBuffer, numPartitions, pool);

  vector_size_t* partitionOffsetsForNextLevel =
      partitionOffsetsForNextLevelBuffer->asMutable<vector_size_t>();
  memset(
      partitionOffsetsForNextLevel, 0, numPartitions * sizeof(vector_size_t));
}

} // namespace

PartitionedVector::~PartitionedVector() = default;

// The detection of (numPartitions == 1) is done when the
// endPartitionOffsetsBuffer is populated and will be passed to the next level.
// There are 2 cases of this: 1) RowVector 2) ArrayVector. So we don't need to
// test it when the partitioning is actually happening.
PartitionedVectorPtr PartitionedVector::create(
    VectorPtr& vector,
    const std::vector<uint32_t>& topRowPartitions,
    BufferPtr& topRowOffsetsForCurrentLevel,
    BufferPtr& topRowOffsetsForNextLevelBuffer,
    vector_size_t lastTopRowOffset,
    int32_t numPartitions,
    BufferPtr& beginPartitionOffsetsBuffer,
    BufferPtr& endPartitionOffsetsBuffer,
    BufferPtr& swappingBuffer,
    int32_t nestLevel,
    velox::memory::MemoryPool* pool) {
  auto numRows = vector->size();
  auto encoding = vector->encoding();
  auto typeKind = vector->typeKind();

  switch (encoding) {
    case VectorEncoding::Simple::FLAT: {
      // partitionOffsets should be passed from upper level and already
      // calculated
      VELOX_CHECK(endPartitionOffsetsBuffer);
      if (endPartitionOffsetsBuffer->size() <
          numPartitions * sizeof(vector_size_t)) {
        break;
      }
      VELOX_CHECK_GE(
          endPartitionOffsetsBuffer->size(),
          numPartitions * sizeof(vector_size_t));

      auto partitionedFlatVector = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
          createPartitionedFlatVector,
          typeKind,
          vector,
          topRowPartitions,
          topRowOffsetsForCurrentLevel,
          lastTopRowOffset,
          numPartitions,
          beginPartitionOffsetsBuffer,
          endPartitionOffsetsBuffer,
          swappingBuffer,
          nestLevel,
          pool);
      return partitionedFlatVector;
    }

    case VectorEncoding::Simple::ROW: {
      // if partitionOffsets is not passed from upper level, create it and
      // calculate partition offsets
      ensureCapacity<vector_size_t>(
          endPartitionOffsetsBuffer, numPartitions, pool);
      auto* partitionOffsets =
          endPartitionOffsetsBuffer->asMutable<vector_size_t>();

      if (numPartitions > 1) {
        std::fill(&partitionOffsets[0], &partitionOffsets[numPartitions], 0);
        countPartitionSizes(topRowPartitions, partitionOffsets);
        prefixSum(partitionOffsets, numPartitions);
      } else {
        partitionOffsets[0] = numRows;
      }

      auto rowVectorPtr = vector->as<RowVector>();
      std::vector<PartitionedVectorPtr> children;
      children.reserve(rowVectorPtr->childrenSize());
      for (auto i = 0; i < rowVectorPtr->childrenSize(); i++) {
        // The children will share the partition offsets of the RowVector

        children.push_back(
            PartitionedVector::create(
                rowVectorPtr->childAt(i),
                topRowPartitions,
                topRowOffsetsForCurrentLevel,
                topRowOffsetsForNextLevelBuffer,
                lastTopRowOffset,
                numPartitions,
                beginPartitionOffsetsBuffer,
                endPartitionOffsetsBuffer,
                swappingBuffer,
                nestLevel,
                pool));
      }

      auto partitionedRowVector = std::make_shared<PartitionedRowVector>(
          vector,
          numPartitions,

          endPartitionOffsetsBuffer,
          nullptr,
          pool,
          children);
      return partitionedRowVector;
    }

    case VectorEncoding::Simple::ARRAY: {
      // partitionOffsets should be passed from upper level and already
      // calculated
      VELOX_CHECK(endPartitionOffsetsBuffer);
      VELOX_CHECK_GE(
          endPartitionOffsetsBuffer->size(), numPartitions * sizeof(uint32_t));
      auto* endPartitionOffsets =
          endPartitionOffsetsBuffer->asMutable<vector_size_t>();

      // Create without child elements first
      auto partitionedArrayVector = std::make_shared<PartitionedArrayVector>(
          vector,
          numPartitions,
          endPartitionOffsetsBuffer,
          nullptr,
          pool,
          nullptr);

      auto arrayVector = vector->as<ArrayVector>();
      auto* arrayOffsets = arrayVector->rawOffsets();
      auto* arraySizes = arrayVector->rawSizes();

      BufferPtr partitionOffsetsForNextLevelBuffer;
      preparePartitionOffsetsForNextLevel(
          numPartitions, pool, partitionOffsetsForNextLevelBuffer);
      auto* partitionOffsetsForNextLevel =
          partitionOffsetsForNextLevelBuffer->asMutable<vector_size_t>();

      // TODO: If numPartitions == 1, we can skip the partitioning and counting
      // topRowOffsetsForNextLevel
      if (numPartitions == 1) {
        VELOX_CHECK_EQ(endPartitionOffsets[0], vector->size());
        partitionOffsetsForNextLevel[0] = arrayOffsets[numRows - 1];
        lastTopRowOffset = arrayOffsets[numRows - 1] + arraySizes[numRows - 1];
      } else {
        // Partition the array arraySizes. This will populate
        // topRowOffsetsForNextLevelBuffer
        partitionedArrayVector->partition(
            topRowPartitions,
            topRowOffsetsForCurrentLevel,
            topRowOffsetsForNextLevelBuffer,
            beginPartitionOffsetsBuffer,
            nestLevel);

        // Calculate the partitionOffsets for the next level. This will populate
        // the topRowOffsetsForNextLevelBuffer from the current level
        auto* topRowOffsetsForNextLevel =
            topRowOffsetsForNextLevelBuffer->asMutable<vector_size_t>();
        lastTopRowOffset = arrayOffsets[numRows - 1] + arraySizes[numRows - 1];

        countPartitionSizes(
            topRowPartitions,
            topRowOffsetsForNextLevel,
            lastTopRowOffset,
            topRowPartitions.size(),
            partitionOffsetsForNextLevel);
        prefixSum(partitionOffsetsForNextLevel, numPartitions);

        // swap topRowOffsetsForNextLevelBuffer and topRowOffsetsForCurrentLevel
        std::swap(
            topRowOffsetsForNextLevelBuffer, topRowOffsetsForCurrentLevel);
      }

      auto elements = PartitionedVector::create(
          vector->as<ArrayVector>()->elements(),
          topRowPartitions,
          topRowOffsetsForCurrentLevel,
          topRowOffsetsForNextLevelBuffer,
          lastTopRowOffset,
          numPartitions,
          partitionOffsetsForNextLevelBuffer,
          beginPartitionOffsetsBuffer,
          swappingBuffer,
          nestLevel + 1,
          pool);

      partitionedArrayVector->setElements(elements);
      return partitionedArrayVector;
    }

    case VectorEncoding::Simple::DICTIONARY: {
      initializeBeginPartitionOffsets(
          beginPartitionOffsetsBuffer,
          endPartitionOffsetsBuffer,
          numPartitions,
          pool);
      return VELOX_DYNAMIC_TYPE_DISPATCH(
          createPartitionedVectorFromDictionary,
          typeKind,
          vector,
          topRowPartitions,
          topRowOffsetsForCurrentLevel,
          lastTopRowOffset,
          numPartitions,
          beginPartitionOffsetsBuffer,
          endPartitionOffsetsBuffer,
          swappingBuffer,
          nestLevel,
          pool);
    }

    case VectorEncoding::Simple::BIASED:
    case VectorEncoding::Simple::SEQUENCE:
    case VectorEncoding::Simple::MAP:
    case VectorEncoding::Simple::LAZY:
      VELOX_UNSUPPORTED(
          "Unsupported vector encoding for OptimizedPartitionedOutput: ",
          encoding);
      break;
    default:
      VELOX_UNREACHABLE(
          "Invalid vector encoding for OptimizedPartitionedOutput: ", encoding);
  }

  return nullptr;
}

PartitionedVectorPtr PartitionedVector::createWrapped(
    VectorPtr& vector,
    const std::vector<uint32_t>& topRowPartitions,
    BufferPtr& upperLevelOffsets,
    vector_size_t upperLevelLastOffset,
    int32_t numPartitions,
    BufferPtr& beginPartitionOffsetsBuffer,
    BufferPtr& endPartitionOffsetsBuffer,
    BufferPtr& swappingBuffer,
    BufferPtr& indicesBuffer,
    int32_t nestLevel,
    velox::memory::MemoryPool* pool) {
  // If the vector is a dictionary, we need to populate an indicesBuffer with
  // the remapped ids. All nested vectors below the dictionary will be wrapped
  // in this indicesBuffer. No actual partitioning will be done now. The real
  // data will be mapped by the indices and copied to the output streams at the
  // flush stage. The indicesBuffer passed in should be already partitioned.
  auto numRows = vector->size();
  auto encoding = vector->encoding();
  auto typeKind = vector->typeKind();

  switch (encoding) {
    case VectorEncoding::Simple::FLAT: {
      auto partitionedFlatVector = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
          createWrappedPartitionedFlatVector,
          typeKind,
          vector,
          numPartitions,
          endPartitionOffsetsBuffer,
          indicesBuffer,
          pool);
      return partitionedFlatVector;
    }

    case VectorEncoding::Simple::ROW: {
      VELOX_UNSUPPORTED(
          "Unsupported vector encoding for OptimizedPartitionedOutput: ",
          encoding);
    }

    case VectorEncoding::Simple::ARRAY: {
      // partitionOffsets should be passed from upper level and already
      // calculated
      VELOX_CHECK(endPartitionOffsetsBuffer);
      VELOX_CHECK_GE(
          endPartitionOffsetsBuffer->size(),
          numPartitions * sizeof(vector_size_t));
      auto partitionOffsets =
          endPartitionOffsetsBuffer->asMutable<vector_size_t>();

      auto* arrayVector = vector->as<ArrayVector>();
      auto* arrayOffsets = arrayVector->rawOffsets();
      auto* arraySizes = arrayVector->rawSizes();

      // Calculate the indices and partition offsets for the next level
      BufferPtr partitionOffsetsForNextLevelBuffer;
      preparePartitionOffsetsForNextLevel(
          numPartitions, pool, partitionOffsetsForNextLevelBuffer);

      // Calculate the partition offsets, and the total number of elements for
      // the next level. No special numPartitions=1 case here, as the operations
      // are the same
      auto partitionOffsetsForNextLevel =
          partitionOffsetsForNextLevelBuffer->asMutable<vector_size_t>();
      auto numIndicesForNextLevel = 0;
      auto lastOffset = 0;

      for (auto i = 0; i < numPartitions; ++i) {
        auto numIdsInPartition = 0;
        auto offset = partitionOffsets[i];
        for (auto j = lastOffset; j < offset; ++j) {
          numIdsInPartition += arraySizes[j];
        }
        lastOffset = offset;
        partitionOffsetsForNextLevel[i] = numIdsInPartition;
        numIndicesForNextLevel += numIdsInPartition;
      }
      prefixSum(partitionOffsetsForNextLevel, numPartitions);

      // Calculate the indices for the next level. This is not related to
      // partition offsets
      BufferPtr indicesForNextLevelBuffer;
      ensureCapacity<vector_size_t>(
          indicesForNextLevelBuffer, numIndicesForNextLevel, pool);
      auto* indicesForNextLevel =
          indicesForNextLevelBuffer->asMutable<vector_size_t>();
      auto* indices = indicesBuffer->asMutable<vector_size_t>();
      auto indicesForNextLevelOffset = 0;
      for (auto i = 0; i < numRows; i++) {
        auto id = indices[i];
        auto offset = arrayOffsets[id];
        auto size = arraySizes[id];
        std::iota(
            indicesForNextLevel + indicesForNextLevelOffset,
            indicesForNextLevel + indicesForNextLevelOffset + size,
            offset);
        indicesForNextLevelOffset += size;
      }

      auto elements = PartitionedVector::createWrapped(
          arrayVector->elements(),
          topRowPartitions,
          const_cast<BufferPtr&>(arrayVector->offsets()),
          arrayOffsets[numRows - 1] + arraySizes[numRows - 1],
          numPartitions,
          beginPartitionOffsetsBuffer,
          partitionOffsetsForNextLevelBuffer,
          swappingBuffer,
          indicesForNextLevelBuffer,
          nestLevel + 1,
          pool);
      auto partitioningArrayVector = std::make_shared<PartitionedArrayVector>(
          vector,
          numPartitions,
          beginPartitionOffsetsBuffer,
          indicesBuffer,
          pool,
          elements);
      return partitioningArrayVector;
    }

    case VectorEncoding::Simple::DICTIONARY: {
      // Flatten to no more than one level of indicesBuffer.
      return VELOX_DYNAMIC_TYPE_DISPATCH(
          remapIdsAndCreatePartitionedVector,
          typeKind,
          vector,
          topRowPartitions,
          upperLevelOffsets,
          upperLevelLastOffset,
          numPartitions,
          beginPartitionOffsetsBuffer,
          endPartitionOffsetsBuffer,
          swappingBuffer,
          indicesBuffer,
          nestLevel,
          pool);

      default:
        VELOX_UNSUPPORTED(
            "Unsupported vector encoding for OptimizedPartitionedOutput: ",
            encoding);
    }
  }
}

VectorPtr PartitionedVector::baseVector() {
  return vector_;
}

std::string PartitionedVector::toString() const {
  std::string offsets;
  for (auto i = 0; i < numPartitions_; i++) {
    offsets += rawPartitionOffsets_[i];
    offsets += ",";
  }

  return fmt::format(
      "PartitionedVector[numPartitions: {}, offsets: {}]",
      numPartitions_,
      offsets);
}

template <typename T>
void PartitionedFlatVector<T>::partition(
    const std::vector<uint32_t>& topRowPartitions,
    BufferPtr& topRowOffsetsBuffer,
    vector_size_t lastTopRowOffset,
    BufferPtr& beginPartitionOffsetsBuffer,
    BufferPtr& swappingBuffer,
    int32_t nestLevel) {
  auto valuesBuffer = vector_->as<FlatVector<T>>()->values();

  // TODO: partition nulls
  char* rawNulls = (char*)vector_->rawNulls();
  if (rawNulls) {
    partitionBitsInPlace(
        rawNulls,
        topRowPartitions,
        numPartitions_,
        beginPartitionOffsetsBuffer,
        partitionOffsets_,
        pool_);
  }

  partitionFixedWidthValues<T>(
      valuesBuffer,
      vector_->size(),
      topRowPartitions,
      topRowOffsetsBuffer,
      lastTopRowOffset,
      numPartitions_,
      beginPartitionOffsetsBuffer,
      partitionOffsets_,
      swappingBuffer,
      nestLevel,
      pool_);
}

template <typename T>
VectorPtr PartitionedFlatVector<T>::partitionAt(int32_t partition) {
  VELOX_CHECK(partition < numPartitions_);

  vector_size_t beginOffset =
      partition == 0 ? 0 : rawPartitionOffsets_[partition - 1];
  vector_size_t numRowsInPartition =
      rawPartitionOffsets_[partition] - beginOffset;

  if (indices_) {
    auto indicesOfPartition = Buffer::slice<vector_size_t>(
        indices_, beginOffset, numRowsInPartition, pool_);

    return BaseVector::wrapInDictionary(
        nullptr, indicesOfPartition, numRowsInPartition, vector_);
  } else {
    return vector_->slice(beginOffset, numRowsInPartition);
  }
}

void PartitionedArrayVector::partition(
    const std::vector<uint32_t>& topRowPartitions,
    BufferPtr& topRowOffsetsForCurrentLevel,
    BufferPtr& topRowOffsetsForNextLevel,
    BufferPtr& beginPartitionOffsetsBuffer,
    int32_t nestLevel) {
  auto numRows = vector_->size();
  auto numTopRows = topRowPartitions.size();
  auto arrayVector = vector_->as<ArrayVector>();
  auto arraySizesBuffer = arrayVector->mutableSizes(numRows);
  auto* arraySizes = arraySizesBuffer->asMutable<vector_size_t>();
  auto arrayOffsetsBuffer = arrayVector->mutableOffsets(numRows);
  auto* arrayOffsets = arrayOffsetsBuffer->asMutable<vector_size_t>();

  if (nestLevel == 0) {
    // top level array. Use its offsets array as topRowOffsets.
    ensureCapacity<vector_size_t>(topRowOffsetsForNextLevel, numTopRows, pool_);
    auto* topRowOffsetsForNext =
        topRowOffsetsForNextLevel->asMutable<vector_size_t>();
    topRowOffsetsForNext[0] = 0;
    std::memcpy(
        topRowOffsetsForNext, arrayOffsets, numTopRows * sizeof(vector_size_t));
    topRowOffsetsForCurrentLevel = arrayOffsetsBuffer;
  } else {
    // This is not the top level. Calculate the topRowOffsetsForNextLevel
    // from arrayOffsets first, because partitioning arraySizes may need to
    // overwrite the arrayOffsets
    VELOX_CHECK(topRowOffsetsForNextLevel);
    VELOX_CHECK(topRowOffsetsForCurrentLevel);

    auto* topRowOffsetsForCurrent =
        topRowOffsetsForCurrentLevel->asMutable<vector_size_t>();
    auto* topRowOffsetsForNext =
        topRowOffsetsForNextLevel->asMutable<vector_size_t>();

    for (auto topRow = 0; topRow < numTopRows - 1; topRow++) {
      topRowOffsetsForNext[topRow] =
          arrayOffsets[topRowOffsetsForCurrent[topRow]];
    }
  }

  auto lastTopRowOffset = arrayOffsets[numRows - 1] + arraySizes[numRows - 1];
  partitionFixedWidthValues<vector_size_t>(
      arraySizesBuffer,
      vector_->size(),
      topRowPartitions,
      topRowOffsetsForCurrentLevel,
      lastTopRowOffset,
      numPartitions_,
      beginPartitionOffsetsBuffer,
      partitionOffsets_,
      arrayOffsetsBuffer,
      nestLevel,
      pool_,
      true);
}

void PartitionedArrayVector::setElements(PartitionedVectorPtr elements) {
  elements_ = elements;
}

PartitionedVectorPtr PartitionedArrayVector::elements() {
  return elements_;
}

} // namespace facebook::velox

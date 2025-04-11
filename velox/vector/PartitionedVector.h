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
#pragma once

#include <vector>

#include "velox/vector/BaseVector.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox {

class PartitionedVector;
using PartitionedVectorPtr = std::shared_ptr<PartitionedVector>;

namespace {

// TODO: This was copied from dwio::common::BufferUtil.h. However the vector
// module should not depend on dwio. Move this to a common place
template <typename T>
void ensureCapacity(
    BufferPtr& data,
    size_t capacity,
    velox::memory::MemoryPool* pool,
    bool preserveOldData = false,
    bool clearBits = false) {
  size_t oldSize = 0;
  if (!data) {
    data = AlignedBuffer::allocate<T>(capacity, pool);
  } else {
    oldSize = data->size();
    if (!data->isMutable() ||
        data->capacity() < BaseVector::byteSize<T>(capacity)) {
      auto newData = AlignedBuffer::allocate<T>(capacity, pool);
      if (preserveOldData) {
        std::memcpy(
            newData->template asMutable<uint8_t>(),
            data->as<uint8_t>(),
            oldSize);
      }
      data = newData;
    }
  }

  if (clearBits) {
    std::memset(
        (void*)(data->asMutable<int8_t>() + oldSize), 0L, capacity - oldSize);
  }
}

inline void prefixSum(vector_size_t*& offsets, uint32_t numPartitions) {
  for (uint32_t i = 1; i < numPartitions; i++) {
    offsets[i] += offsets[i - 1];
  }
}

} // namespace

class PartitionedVector {
 public:
  /// Disable default constructor.
  PartitionedVector() = delete;

  /// Disable copy constructor and assignment.
  PartitionedVector(const PartitionedVector& other) = delete;
  PartitionedVector& operator=(const PartitionedVector& other) = delete;

  //  /// Allow move constructor and move assignment operator.
  //  PartitionedVector(PartitionedVector&& other) = default;
  //  PartitionedVector& operator=(PartitionedVector&& other) = default;

  /// Virtual destructor.
  virtual ~PartitionedVector();

  static PartitionedVectorPtr create(
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
      velox::memory::MemoryPool* pool);

  static PartitionedVectorPtr createWrapped(
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
      velox::memory::MemoryPool* pool);

  PartitionedVector(
      VectorPtr vector,
      const int32_t numPartitions,
      BufferPtr partitionOffsets,
      BufferPtr indices,
      velox::memory::MemoryPool* pool)
      : vector_(vector),
        numPartitions_(numPartitions),
        partitionOffsets_(partitionOffsets),
        indices_(indices),
        pool_(pool) {
    if (!partitionOffsets_) {
      partitionOffsets_ =
          AlignedBuffer::allocate<vector_size_t>(numPartitions, pool);
    }
    rawPartitionOffsets_ = partitionOffsets_->asMutable<vector_size_t>();
  }

  /// Returns the underlying vector.
  VectorPtr baseVector();

  /// Returns the partitioned vector at partition p. If the number of rows in
  /// that partition is 0, returns an empty vector.
  virtual VectorPtr partitionAt(int32_t partition) = 0;

  template <typename T>
  T* as() {
    static_assert(std::is_base_of_v<PartitionedVector, T>);
    return dynamic_cast<T*>(this);
  }

  TypeKind typeKind() const {
    return vector_->typeKind();
  }

  //  virtual void partition(
  //      const std::vector<uint32_t>& topRowPartitions,
  //      vector_size_t*& topRowOffsets,
  //      BufferPtr& beginOffsetsBuffer,
  //      BufferPtr& swappingBuffer,
  //      vector_size_t numTopRows) = 0;

  vector_size_t* rawPartitionOffsets() {
    return rawPartitionOffsets_;
  }

  BufferPtr indices() {
    return indices_;
  }

  virtual const vector_size_t* rawSizes() = 0;

  /// Returns string representation of the value in the specified row.
  virtual std::string toString() const;

 protected:
  //  void initializeBeginPartitionOffsets(BufferPtr&
  //  beginPartitionOffsetsBuffer);

  VectorPtr vector_;
  const uint32_t numPartitions_;
  BufferPtr partitionOffsets_;
  vector_size_t* rawPartitionOffsets_;
  // We want to defer the DictionaryVector values copy-out to the fluhing time
  BufferPtr indices_;
  velox::memory::MemoryPool* pool_;
};

using PartitionedVectorPtr = std::shared_ptr<PartitionedVector>;

template <typename T>
class PartitionedFlatVector : public PartitionedVector {
 public:
  PartitionedFlatVector(
      VectorPtr flatVector,
      const int32_t numPartitions,
      BufferPtr partitionOffsets,
      BufferPtr indices,
      velox::memory::MemoryPool* pool)
      : PartitionedVector(
            flatVector,
            numPartitions,
            partitionOffsets,
            indices,
            pool) {}

  void partition(
      const std::vector<uint32_t>& topRowPartitions,
      BufferPtr& topRowOffsets,
      vector_size_t lastTopRowOffset,
      BufferPtr& beginOffsetsBuffer,
      BufferPtr& swappingBuffer,
      int32_t nestLevel);

  const vector_size_t* rawSizes() override {
    VELOX_UNREACHABLE("PartitionedFlatVector does not implement rawSizes()");
  }

  VectorPtr partitionAt(int32_t partition) override;
};

class PartitionedRowVector : public PartitionedVector {
 public:
  PartitionedRowVector(
      VectorPtr vector,
      //      const std::vector<uint32_t>& partitions,
      const int32_t numPartitions,
      BufferPtr partitionOffsets,
      BufferPtr indices,
      velox::memory::MemoryPool* pool,
      std::vector<PartitionedVectorPtr>& children)
      : PartitionedVector(
            vector,
            //            partitions,
            numPartitions,
            partitionOffsets,
            indices,
            pool),
        children_(children) {}

  ~PartitionedRowVector() {
    children_.clear();
  }

  /// Get the child vector at a given offset.
  std::shared_ptr<PartitionedVector> childAt(column_index_t index) {
    VELOX_CHECK_LT(
        index,
        static_cast<column_index_t>(children_.size()),
        "Trying to access non-existing child in RowVector: {}",
        toString());
    return children_[index];
  }

  const vector_size_t* rawSizes() override {
    VELOX_UNREACHABLE("PartitionedFlatVector does not implement rawSizes()");
  }

  VectorPtr partitionAt(int32_t partition) override {
    VELOX_NYI("PartitionedRowVector::partitionAt not implemented");
  }

 private:
  std::vector<PartitionedVectorPtr> children_;
};

class PartitionedArrayVector : public PartitionedVector {
 public:
  PartitionedArrayVector(
      VectorPtr vector,
      int32_t numPartitions,
      BufferPtr partitionOffsets,
      BufferPtr indices,
      velox::memory::MemoryPool* pool,
      PartitionedVectorPtr elements)
      : PartitionedVector(
            vector,
            numPartitions,
            partitionOffsets,
            indices,
            pool),
        elements_(elements) {}

  void partition(
      const std::vector<uint32_t>& topRowPartitions,
      BufferPtr& topRowOffsetsForCurrentLevel,
      BufferPtr& topRowOffsetsForNextLevel,
      BufferPtr& beginPartitionOffsetsBuffer,
      int32_t nestLevel);

  VectorPtr partitionAt(int32_t partition) override {
    VELOX_NYI("PartitionedRowVector::partitionAt not implemented");
  }

  void setElements(PartitionedVectorPtr elements);

  PartitionedVectorPtr elements();

  const vector_size_t* rawSizes() override {
    return vector_->as<ArrayVector>()->rawSizes();
  }

 private:
  PartitionedVectorPtr elements_;
};

} // namespace facebook::velox

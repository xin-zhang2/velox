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
#include <algorithm>
#include <random>

#include <gtest/gtest.h>

#include "vector/tests/utils/VectorTestBase.h"
#include "velox/vector/PartitionedVector.h"
#include "velox/vector/tests/utils/PartitionedVectorTestBase.h"

using namespace facebook::velox;

class PartitioningVectorTest : public testing::Test,
                               public test::PartitionedVectorTestBase {
 protected:
  std::mt19937 gen_ = std::mt19937(std::random_device{}());
  BufferPtr topRowOffsets_;
  BufferPtr beginPartitionOffsets_;
  BufferPtr endPartitionOffsets_;
  BufferPtr swappingBuffer_;

  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance({});
  }

  void testPartitionedVector(
      VectorPtr vector,
      const std::vector<uint32_t>& partitions,
      uint32_t numPartitions) {
    auto numValues = partitions.size();

    // Back up the vector before calling PartitionedVector::create()
    VectorPtr vectorCopy = BaseVector::copy(*vector);
    // Build the expected vector
    std::vector<VectorPtr> expectedVectors =
        partitionVectorByWrapping(vectorCopy, partitions, numPartitions);

    ensureCapacity<vector_size_t>(topRowOffsets_, numValues, pool_.get());
    ensureCapacity<vector_size_t>(
        beginPartitionOffsets_, numPartitions, pool_.get());
    ensureCapacity<vector_size_t>(
        endPartitionOffsets_, numPartitions, pool_.get());
    ensureCapacity<vector_size_t>(swappingBuffer_, numValues, pool_.get());

    // Calculate the number of values for each partition
    std::vector<vector_size_t> partitionRowCounts(numPartitions, 0);
    for (auto partition : partitions) {
      partitionRowCounts[partition]++;
    }

    // Initialize endPartitionOffsets_
    auto rawEndPartitionOffsets =
        endPartitionOffsets_->asMutable<vector_size_t>();
    vector_size_t offset = 0;
    for (uint32_t i = 0; i < numPartitions; ++i) {
      offset += partitionRowCounts[i];
      rawEndPartitionOffsets[i] = offset;
    }
    endPartitionOffsets_->setSize(numPartitions * sizeof(vector_size_t));

    auto partitionedVector = PartitionedVector::create(
        vector,
        partitions,
        topRowOffsets_,
        topRowOffsets_,
        numValues,
        numPartitions,
        beginPartitionOffsets_,
        endPartitionOffsets_,
        swappingBuffer_,
        0,
        pool_.get());
    ASSERT_NE(partitionedVector, nullptr);

//    std::vector<VectorPtr> partitionedVectors(numPartitions, nullptr);
    std::vector<VectorPtr> partitionedVectors;
    vector_size_t lastOffset = 0;
    for (uint32_t i = 0; i < numPartitions; ++i) {
      auto partition = partitionedVector->partitionAt(i);
      partitionedVectors.push_back(partition);
//      partitionedVectors[i] = partitionedVector->partitionAt(i);
    }

    for (uint32_t i = 0; i < numPartitions; ++i) {
      test::assertEqualVectors(
          expectedVectors[i], canonicalize(partitionedVectors[i]));
    }
  }

  void testVectorPartitioning(VectorPtr vector) {
    // 100 random values between 1 and 100
    //  auto randomValues = generateRandomValues(100, 1, 100);
    auto numRows = vector->size();
    std::vector<uint32_t> partitions(numRows);

    // two partitions
    for (uint32_t i = 0; i < partitions.size(); ++i) {
      partitions[i] = i % 2;
    }
    auto vectorCopy = BaseVector::copy(*vector, pool_.get());
    testPartitionedVector(vectorCopy, partitions, 2);

    // three partitions
    for (uint32_t i = 0; i < partitions.size(); ++i) {
      partitions[i] = i % 3;
    }
    vectorCopy = BaseVector::copy(*vector, pool_.get());
    testPartitionedVector(vectorCopy, partitions, 3);

    // one partitions
    std::fill(partitions.begin(), partitions.end(), 0);
    vectorCopy = BaseVector::copy(*vector, pool_.get());
    testPartitionedVector(vectorCopy, partitions, 1);

    // One value per partition
    std::iota(partitions.begin(), partitions.end(), 0);
    vectorCopy = BaseVector::copy(*vector, pool_.get());
    testPartitionedVector(vectorCopy, partitions, numRows);

    // Random number of partitions less than the number of values
    std::uniform_int_distribution<> dis(0, numRows);
    uint32_t maxPartition = 0;
    for (uint32_t i = 0; i < numRows; ++i) {
      partitions[i] = dis(gen_);
      maxPartition = std::max(maxPartition, partitions[i]);
    }
    vectorCopy = BaseVector::copy(*vector, pool_.get());
    testPartitionedVector(vectorCopy, partitions, maxPartition + 1);

    // Four partitions, where the first partition is empty
    for (uint32_t i = 0; i < partitions.size(); ++i) {
      partitions[i] = i % 3 + 1;
    }
    vectorCopy = BaseVector::copy(*vector, pool_.get());
    testPartitionedVector(vectorCopy, partitions, 4);
  }
};

TEST_F(PartitioningVectorTest, testFlatVector) {
  std::vector<int> numValuesVector({100, 5});

  for (auto numValues : numValuesVector) {
    // random values, no nulls
    testVectorPartitioning(
        makeFlatVector<int>(numValues, [](auto row) { return row; }));

    // random values, with half number of nulls
    testVectorPartitioning(
        makeFlatVector<int>(
            numValues, [](auto row) { return row; }, nullEvery(2, 1)));

    // All nulls
    testVectorPartitioning(makeAllNullFlatVector<int>(numValues));
  }
}

TEST_F(PartitioningVectorTest, testDictionaryOfFlatVector) {
  auto testCase = [&](int numValues, VectorPtr flatVector) {
    auto indices = makeIndices(numValues, [](auto row) { return row; });
    auto dict =
        BaseVector::wrapInDictionary(nullptr, indices, numValues, flatVector);
    testVectorPartitioning(dict);
  };

  std::vector<int> numValuesVector({100, 5});
  for (auto numValues : numValuesVector) {
    testCase(numValues, makeFlatVector<int>(numValues, [](auto row) {
               return row;
             }));
    testCase(
        numValues,
        makeFlatVector<int>(
            numValues, [](auto row) { return row; }, nullEvery(2, 1)));

    testCase(numValues, makeAllNullFlatVector<int>(numValues));
  }
}

TEST_F(PartitioningVectorTest, testDictOverDictOfFlatVector) {
  auto testCase = [&](int numValues, VectorPtr flatVector) {
    auto baseIndices = makeIndices(numValues, [](auto row) { return row; });
    auto dict1 = BaseVector::wrapInDictionary(
        nullptr, baseIndices, numValues, flatVector);

    auto secondIndices =
        makeIndices(numValues * 2, [](auto row) { return row / 2; });
    auto dict2 = BaseVector::wrapInDictionary(
        nullptr, secondIndices, numValues * 2, dict1);

    testVectorPartitioning(dict2);
  };

  std::vector<int> numValuesVector({100, 5});
  for (auto numValues : numValuesVector) {
    // random values, no nulls
    testCase(numValues, makeFlatVector<int>(numValues, [](auto row) {
               return row;
             }));

    // random values, with half number of nulls
    testCase(
        numValues,
        makeFlatVector<int>(
            numValues, [](auto row) { return row; }, nullEvery(2, 1)));
    testCase(numValues, makeAllNullFlatVector<int>(numValues));
  }
}


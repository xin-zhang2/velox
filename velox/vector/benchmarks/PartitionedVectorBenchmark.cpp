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

#include <absl/random/uniform_int_distribution.h>
#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "dwio/common/tests/utils/BatchMaker.h"
#include "vector/VectorPrinter.h"
#include "vector/tests/utils/PartitionedVectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;

namespace facebook::velox::test {

namespace {

auto gen_ = std::mt19937(std::random_device{}());

auto noNulls = [](vector_size_t) { return false; };

auto allNulls = [](vector_size_t) { return true; };

auto halfNulls = [](vector_size_t row) { return row % 2 == 0; };

template <TypeKind T>
RowTypePtr scalarTypeGenerator(int32_t numColumns) {
  return ROW(std::vector<TypePtr>(numColumns, createScalarType<T>()));
}

RowTypePtr mixedFlatTypeGenerator(int32_t numColumns) {
  const std::vector<TypePtr> typeSelection = {
      BOOLEAN(),
      TINYINT(),
      SMALLINT(),
      INTEGER(),
      BIGINT(),
      HUGEINT(),
      REAL(),
      DOUBLE(),
      TIMESTAMP(),
      DATE(),
      DECIMAL(10, 2),
      DECIMAL(20, 3),
  };

  std::vector<TypePtr> types;
  types.reserve(numColumns);

  for (int i = 0; i < numColumns; ++i) {
    types.push_back(typeSelection[i % typeSelection.size()]);
  }

  return ROW(std::move(types));
}

auto roundRobinPartitionFunction = [](const RowVectorPtr& vector,
                                      const int32_t numPartitions,
                                      std::vector<uint32_t>& partitions) {
  partitions.resize(vector->size());
  for (auto i = 0; i < vector->size(); ++i) {
    partitions[i] = i % numPartitions;
  }
};

auto randomPartitionFunction = [](const RowVectorPtr& vector,
                                  const int32_t numPartitions,
                                  std::vector<uint32_t>& partitions) {
  partitions.resize(vector->size());
  std::uniform_int_distribution<int> dis(0, numPartitions - 1);
  for (int i = 0; i < vector->size(); ++i) {
    partitions[i] = dis(gen_);
  }
};
} // namespace

class PartitionedVectorBenchmark : public VectorTestBase {
 protected:
  std::vector<uint32_t> partitions_;
  BufferPtr topRowOffsets_;
  BufferPtr beginPartitionOffsets_;
  BufferPtr endPartitionOffsets_;
  BufferPtr swappingBuffer_;

 public:
  RowVectorPtr createTestVector(
      const std::function<RowTypePtr(int32_t)>& rowTypeGenerator,
      const int32_t numRows,
      const int32_t numColumns,
      const std::function<bool(vector_size_t)>& isNullAt) {
    auto rowType = rowTypeGenerator(numColumns);
    const auto batch =
        BatchMaker::createBatch(rowType, numRows, *pool_, isNullAt);
    return std::dynamic_pointer_cast<RowVector>(batch);
  }

  void prepareBuffers(
      const vector_size_t numRows,
      const int32_t numPartitions) {
    ensureCapacity<vector_size_t>(topRowOffsets_, numRows, pool());
    ensureCapacity<vector_size_t>(
        beginPartitionOffsets_, numPartitions, pool());
    ensureCapacity<vector_size_t>(endPartitionOffsets_, numPartitions, pool());
  }

  void calculatePartitionOffsets(int32_t numPartitions) {
    std::vector partitionRowCounts(numPartitions, 0);
    for (auto partition : partitions_) {
      partitionRowCounts[partition]++;
    }
    auto rawEndPartitionOffsets =
        endPartitionOffsets_->asMutable<vector_size_t>();
    vector_size_t offset = 0;
    for (int32_t i = 0; i < numPartitions; ++i) {
      offset += partitionRowCounts[i];
      rawEndPartitionOffsets[i] = offset;
    }
    endPartitionOffsets_->setSize(numPartitions * sizeof(vector_size_t));
  }

  void createPartitionedVector(
      const RowVectorPtr& vector,
      const int32_t numPartitions) {
    auto vectorPtr = std::dynamic_pointer_cast<BaseVector>(vector);
    PartitionedVector::create(
        vectorPtr,
        partitions_,
        topRowOffsets_,
        topRowOffsets_,
        vector->size(),
        numPartitions,
        beginPartitionOffsets_,
        endPartitionOffsets_,
        swappingBuffer_,
        0,
        pool());
  }

  memory::MemoryPool* getPool() const {
    return pool_.get();
  }

  void run(const RowVectorPtr& vector, const int32_t numPartitions) {
    folly::BenchmarkSuspender suspender;
    // roundRobinPartitionFunction(vector, numPartitions, partitions_);
    randomPartitionFunction(vector, numPartitions, partitions_);
    prepareBuffers(vector->size(), numPartitions);
    calculatePartitionOffsets(numPartitions);
    suspender.dismiss();
    createPartitionedVector(vector, numPartitions);
    suspender.rehire();
  }
};
} // namespace facebook::velox::test

std::unique_ptr<PartitionedVectorBenchmark> bm;

void runBM(
    uint32_t iterations,
    const std::function<RowTypePtr(int32_t)>& rowTypeGenerator,
    const int32_t numColumns,
    const std::function<bool(vector_size_t)>& isNullAt = noNulls,
    const int32_t numRows = 10000) {
  folly::BenchmarkSuspender suspender;
  auto vector =
      bm->createTestVector(rowTypeGenerator, numRows, numColumns, isNullAt);
  for (uint32_t i = 0; i < iterations; ++i) {
    const auto vectorCopy = std::static_pointer_cast<RowVector>(
        BaseVector::copy(*vector, bm->getPool()));
    suspender.dismiss();
    bm->run(vectorCopy, 10000);
    suspender.rehire();
  }
}

#define BENCHMARK_SCALAR_GEN(kind, numCols, nulls) \
  BENCHMARK_NAMED_PARAM(                           \
      runBM,                                       \
      kind##_##numCols##Cols_##nulls,              \
      scalarTypeGenerator<TypeKind::kind>,         \
      numCols,                                     \
      nulls);

#define BENCHMARK_SCALAR_SIZES(kind, nulls) \
  BENCHMARK_SCALAR_GEN(kind, 1, nulls)      \
  BENCHMARK_SCALAR_GEN(kind, 10, nulls)     \
  BENCHMARK_SCALAR_GEN(kind, 100, nulls)    \
  BENCHMARK_SCALAR_GEN(kind, 1000, nulls)

#define BENCHMARK_SCALAR(kind)           \
  BENCHMARK_SCALAR_SIZES(kind, noNulls)  \
  BENCHMARK_SCALAR_SIZES(kind, allNulls) \
  BENCHMARK_SCALAR_SIZES(kind, halfNulls)

BENCHMARK_SCALAR(BOOLEAN);
BENCHMARK_SCALAR(SMALLINT);
BENCHMARK_SCALAR(INTEGER);
BENCHMARK_SCALAR(BIGINT);
BENCHMARK_SCALAR(HUGEINT);
BENCHMARK_SCALAR(REAL);
BENCHMARK_SCALAR(DOUBLE);
BENCHMARK_SCALAR(TIMESTAMP);

#define BENCHMARK_MIXED_GEN(numCols, nulls) \
  BENCHMARK_NAMED_PARAM(                    \
      runBM,                                \
      Mixed_##numCols##Cols_##nulls,        \
      mixedFlatTypeGenerator,               \
      numCols,                              \
      nulls);

#define BENCHMARK_MIXED(nulls)    \
  BENCHMARK_MIXED_GEN(1, nulls)   \
  BENCHMARK_MIXED_GEN(10, nulls)  \
  BENCHMARK_MIXED_GEN(100, nulls) \
  BENCHMARK_MIXED_GEN(1000, nulls)

BENCHMARK_MIXED(noNulls);
BENCHMARK_MIXED(allNulls);
BENCHMARK_MIXED(halfNulls);

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  bm = std::make_unique<PartitionedVectorBenchmark>();
  folly::runBenchmarks();
  bm.reset();
  return 0;
}

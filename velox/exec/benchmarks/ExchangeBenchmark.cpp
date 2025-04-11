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
#include <folly/Benchmark.h>
#include <folly/Random.h>
#include <folly/init/Init.h>

#include "velox/core/QueryConfig.h"
#include "velox/dwio/common/tests/utils/BatchMaker.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/TypeResolver.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/vector/tests/utils/VectorTestBase.h"

static const int32_t entries_per_row = 4;

DEFINE_int32(width, 8, "Number of parties in shuffle");
DEFINE_int32(task_width, 4, "Number of threads in each task in shuffle");

DEFINE_int32(num_local_tasks, 1, "Number of concurrent local shuffles");
DEFINE_int32(num_local_repeat, 1, "Number of repeats of local exchange query");
DEFINE_int32(flat_batch_mb, 1, "MB in a 10k row flat batch.");
DEFINE_int64(local_exchange_buffer_mb, 1, "task-wide buffer in local exchange");
DEFINE_int64(exchange_buffer_mb, 32, "task-wide buffer in remote exchange");
DEFINE_int32(dict_pct, 0, "Percentage of columns wrapped in dictionary");

DEFINE_bool(gtest_color, false, "");
DEFINE_string(gtest_filter, "*", "");

/// Benchmarks repartition/exchange with different batch sizes,
/// numbers of destinations and data type mixes.  Generates a plan
/// that 1. shuffles a constant input in each of n workers, sending
/// each partition to n consumers in the next stage. The consumers
/// count the rows and send the count to a final single task stage
/// that returns the sum of the counts. The sum is expected to be n *
/// number of rows in constant input.

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::test;

namespace {
class ExchangeBenchmark : public VectorTestBase {
 public:
  std::vector<RowVectorPtr> makeRows(
      RowTypePtr type,
      int32_t numVectors,
      int32_t rowsPerVector,
      int32_t dictPct = 0) {
    std::vector<RowVectorPtr> vectors;
    BufferPtr indices;
    for (int32_t i = 0; i < numVectors; ++i) {
      //      // This can't control the nullness of the nested columns
      //      auto vector =
      //          std::dynamic_pointer_cast<RowVector>(BatchMaker::createBatch(
      //              type, rowsPerVector, *pool_, [](vector_size_t i) {
      //                return false;
      //              }));

      std::vector<VectorPtr> children;

      //
      //      for (int j = 0; j < type->children().size(); j++) {
      //      auto c0 = makeFlatVector<int64_t>(
      //          rowsPerVector, [&](auto row) { return folly::Random::rand64();
      //          });
      //
      //      children.push_back(c0);

      // make c0
      std::vector<int64_t> values;
      for (auto i = 0; i < rowsPerVector; i++) {
        values.push_back(i);
      }
      auto c0 = makeFlatVector<int64_t>(values);

      children.push_back(c0);

      BufferPtr indices = makeIndices(rowsPerVector, [&](auto i) { return i; });
      auto d0 =
          BaseVector::wrapInDictionary(nullptr, indices, rowsPerVector, c0);
      children.push_back(d0);

      //      // make c0
      //      std::vector<vector_size_t> values;
      //      for (auto i = 0; i < rowsPerVector; i++) {
      //        values.push_back(i);
      //      }
      //      auto c0 = makeFlatVector<int32_t>(values);
      //      children.push_back(c0);

      // Make random array sizes in [10, 100]
      //      std::vector<vector_size_t> offsets;
      //
      //      vector_size_t lastOffset = 0;
      //
      //      for (auto i = 0; i < rowsPerVector; i++) {
      //        offsets.push_back(lastOffset);
      //        auto size = folly::Random::rand32(100, 1000);
      //        lastOffset += size;
      //      }
      //      auto arrayValues =
      //          makeFlatVector<int64_t>(lastOffset, [&](auto row) { return
      //          row; });
      //      auto arrayVector = makeArrayVector(offsets, arrayValues);
      //
      //      //     // Make array sizes = row number
      //      //     auto arrayValues = makeFlatVector<int32_t>(
      //      //         (1 + rowsPerVector) * rowsPerVector / 2,
      //      //         [&](auto row) { return row; });
      //      //     std::vector<vector_size_t> offsets;
      //      //     offsets.push_back(0);
      //      //     for (auto i = 1; i < rowsPerVector; i++) {
      //      //       offsets.push_back(i + offsets[i - 1]);
      //      //     }
      //      //     auto arrayVector = makeArrayVector(offsets, arrayValues);
      //
      //
      //      children.push_back(arrayVector);

      auto vector = makeRowVector(children);

      vectors.push_back(vector);
    }
    return vectors;
  }

  void run(
      std::vector<RowVectorPtr>& vectors,
      int32_t width,
      int32_t taskWidth,
      PlanNodeStats& leafRepartitionStats,
      PlanNodeStats& finalRepartitionStats,
      PlanNodeStats& exchangeStats) {
    core::PlanNodePtr plan;
    core::PlanNodeId exchangeId;
    core::PlanNodeId leafRartitionedOutputId;
    core::PlanNodeId finalAggReartitionedOutputId;

    //    std::vector<std::shared_ptr<Task>> tasks;
    std::vector<std::shared_ptr<Task>> leafTasks;
    std::vector<std::shared_ptr<Task>> finalAggTasks;
    std::vector<exec::Split> finalAggSplits;

    RowVectorPtr expected;

    auto startMicros = getCurrentTimeMicro();
    BENCHMARK_SUSPEND {
      assert(!vectors.empty());
      configSettings_[core::QueryConfig::kMaxPartitionedOutputBufferSize] =
          fmt::format("{}", FLAGS_exchange_buffer_mb << 20);
      auto iteration = ++iteration_;

      // leafPlan: PartitionedOutput/kPartitioned(1) <-- Values(0)
      std::vector<std::string> leafTaskIds;
      auto leafPlan = exec::test::PlanBuilder()
                          .values(vectors, true)
                          .partitionedOutput({"c0"}, width)
                          .capturePlanNodeId(leafRartitionedOutputId)
                          .planNode();

      for (int32_t counter = 0; counter < width; ++counter) {
        auto leafTaskId = makeTaskId(iteration, "leaf", counter);
        leafTaskIds.push_back(leafTaskId);
        auto leafTask = makeTask(leafTaskId, leafPlan, counter);
        leafTasks.push_back(leafTask);
        leafTask->start(taskWidth);
      }

      // finalAggPlan: PartitionedOutput/kPartitioned(2) <-- Agg/kSingle(1) <--
      // Exchange(0)
      std::vector<std::string> finalAggTaskIds;
      core::PlanNodePtr finalAggPlan =
          exec::test::PlanBuilder()
              .exchange(leafPlan->outputType(), VectorSerde::Kind::kPresto)
              .capturePlanNodeId(exchangeId)
              .singleAggregation({}, {"count(1)"})
              .partitionedOutput({}, 1)
              .capturePlanNodeId(finalAggReartitionedOutputId)
              .planNode();

      for (int i = 0; i < width; i++) {
        auto taskId = makeTaskId(iteration, "final-agg", i);
        finalAggSplits.push_back(
            exec::Split(std::make_shared<exec::RemoteConnectorSplit>(taskId)));
        auto finalAggTask = makeTask(taskId, finalAggPlan, i);
        finalAggTasks.push_back(finalAggTask);
        finalAggTask->start(taskWidth);
        addRemoteSplits(finalAggTask, leafTaskIds);
      }

      expected = makeRowVector({makeFlatVector<int64_t>(1, [&](auto /*row*/) {
        return vectors.size() * vectors[0]->size() * width * taskWidth;
      })});

      plan =
          exec::test::PlanBuilder()
              .exchange(finalAggPlan->outputType(), VectorSerde::Kind::kPresto)
              .singleAggregation({}, {"sum(a0)"})
              .planNode();
    };

    exec::test::AssertQueryBuilder(plan)
        .splits(finalAggSplits)
        .assertResults(expected);

    BENCHMARK_SUSPEND {
      auto elapsed = getCurrentTimeMicro() - startMicros;
      //      counters.usec += elapsed;
      std::vector<int64_t> taskWallMs;

      //      PlanNodeStats repartitionStats;
      //      PlanNodeStats exchangeStats;
      for (auto& task : leafTasks) {
        auto taskStats = task->taskStats();
        taskWallMs.push_back(
            taskStats.executionEndTimeMs - taskStats.executionStartTimeMs);
        auto planStats = toPlanStats(taskStats);
        auto& taskRepartitionStats = planStats.at(leafRartitionedOutputId);
        leafRepartitionStats += taskRepartitionStats;
        auto repartitionRuntimeStats = taskRepartitionStats.customStats;
      }

      for (auto& task : finalAggTasks) {
        auto taskStats = task->taskStats();
        taskWallMs.push_back(
            taskStats.executionEndTimeMs - taskStats.executionStartTimeMs);
        auto planStats = toPlanStats(taskStats);

        auto& taskRepartitionStats = planStats.at(finalAggReartitionedOutputId);
        finalRepartitionStats += taskRepartitionStats;
        auto repartitionRuntimeStats = taskRepartitionStats.customStats;

        auto& taskExchangeStats = planStats.at(exchangeId);
        exchangeStats += taskExchangeStats;
        auto exchangeRuntimeStats = taskExchangeStats.customStats;
      }
    };
  }

  void runLocal(
      std::vector<RowVectorPtr>& vectors,
      int32_t numPartitions,
      int32_t numLocalTasks,
      PlanNodeStats& leafRepartitionStats,
      PlanNodeStats& finalRepartitionStats,
      PlanNodeStats& exchangeStats)
  //      (
  //      std::vector<RowVectorPtr>& vectors,
  //      int32_t numPartitions,
  //      int32_t numLocalTasks,
  //      Counters& counters)
  {
    assert(!vectors.empty());
    std::vector<std::shared_ptr<Task>> tasks;
    //    counters.inputBytes = vectors[0]->retainedSize() * vectors.size() *
    //        numLocalTasks * FLAGS_num_local_repeat;
    std::vector<std::string> aggregates = {"count(1)"};
    auto& rowType = vectors[0]->type()->as<TypeKind::ROW>();
    for (auto i = 1; i < rowType.size(); ++i) {
      aggregates.push_back(fmt::format("checksum({})", rowType.nameOf(i)));
    }

    // plan: Agg/kSingle(4) <-- LocalPartition/Gather(3) <--
    //    Agg / kGather(2) < --
    // LocalPartition/kRepartition(1) <-- Values(0)
    core::PlanNodeId exchangeId;
    auto plan = exec::test::PlanBuilder()
                    .values(vectors, true)
                    .localPartition({"c0"})
                    .capturePlanNodeId(exchangeId)
                    .singleAggregation({}, aggregates)
                    .localPartition(std::vector<std::string>{})
                    .singleAggregation({}, {"sum(a0)"})
                    .planNode();
    auto startMicros = getCurrentTimeMicro();

    std::vector<std::thread> threads;
    threads.reserve(numLocalTasks);
    auto expected = makeRowVector({makeFlatVector<int64_t>(
        1,
        [&](auto
            /*row*/) {
          return vectors.size() * vectors[0]->size() * numPartitions;
        })});

    std::mutex mutex;
    for (int32_t i = 0; i < numLocalTasks; ++i) {
      threads.push_back(std::thread([&]() {
        for (auto repeat = 0; repeat < FLAGS_num_local_repeat; ++repeat) {
          auto task =
              exec::test::AssertQueryBuilder(plan)
                  .config(
                      core::QueryConfig::kMaxLocalExchangeBufferSize,
                      fmt::format("{}", FLAGS_local_exchange_buffer_mb << 20))
                  .maxDrivers(numPartitions)
                  .assertResults(expected);
          {
            std::lock_guard<std::mutex> l(mutex);
            tasks.push_back(task);
          }
        }
      }));
    }
    for (auto& thread : threads) {
      thread.join();
    }
    //    counters.usec = getCurrentTimeMicro() - startMicros;
    int64_t totalProducer = 0;
    int64_t totalConsumer = 0;
    std::vector<RuntimeMetric> waitConsumer;
    std::vector<RuntimeMetric> waitProducer;
    std::vector<int64_t> wallMs;
    for (auto& task : tasks) {
      auto taskStats = task->taskStats();
      wallMs.push_back(
          taskStats.executionEndTimeMs - taskStats.executionStartTimeMs);
      auto planStats = toPlanStats(taskStats);
      auto runtimeStats = planStats.at(exchangeId).customStats;
      waitProducer.push_back(runtimeStats["blockedWaitForProducerWallNanos"]);
      waitConsumer.push_back(runtimeStats["blockedWaitForConsumerWallNanos"]);
      totalConsumer += waitConsumer.back().sum;
      totalProducer += waitProducer.back().sum;
    }
    printMax("Producer", totalProducer, waitProducer);
    printMax("Consumer", totalConsumer, waitConsumer);
    std::sort(wallMs.begin(), wallMs.end());
    assert(!wallMs.empty());
    std::cout << "Wall ms: " << wallMs.back() << " / "
              << wallMs[wallMs.size() / 2] << " / " << wallMs.front()
              << std::endl;
  }

 private:
  static constexpr int64_t kMaxMemory = 6UL << 30; // 6GB

  static std::string
  makeTaskId(int32_t iteration, const std::string& prefix, int num) {
    return fmt::format("local://{}-{}-{}", iteration, prefix, num);
  }

  std::shared_ptr<Task> makeTask(
      const std::string& taskId,
      std::shared_ptr<const core::PlanNode> planNode,
      int destination,
      Consumer consumer = nullptr,
      int64_t maxMemory = kMaxMemory) {
    auto configCopy = configSettings_;
    auto queryCtx = core::QueryCtx::create(
        executor_.get(), core::QueryConfig(std::move(configCopy)));
    queryCtx->testingOverrideMemoryPool(
        memory::memoryManager()->addRootPool(queryCtx->queryId(), maxMemory));
    core::PlanFragment planFragment{planNode};
    return Task::create(
        taskId,
        std::move(planFragment),
        destination,
        std::move(queryCtx),
        Task::ExecutionMode::kParallel,
        std::move(consumer));
  }

  void addRemoteSplits(
      std::shared_ptr<Task> task,
      const std::vector<std::string>& remoteTaskIds) {
    for (auto& taskId : remoteTaskIds) {
      auto split =
          exec::Split(std::make_shared<RemoteConnectorSplit>(taskId), -1);
      task->addSplit("0", std::move(split));
    }
    task->noMoreSplits("0");
  }

  void sortByMax(std::vector<RuntimeMetric>& metrics) {
    std::sort(
        metrics.begin(),
        metrics.end(),
        [](const RuntimeMetric& left, const RuntimeMetric& right) {
          return left.max > right.max;
        });
  }

  void printMax(
      const char* title,
      int64_t total,
      std::vector<RuntimeMetric>& metrics) {
    sortByMax(metrics);
    assert(!metrics.empty());
    std::cout << title << " Total " << succinctNanos(total)
              << " Max: " << metrics.front().toString()
              << " Median: " << metrics[metrics.size() / 2].toString()
              << " Min: " << metrics.back().toString() << std::endl;
  }

  std::unordered_map<std::string, std::string> configSettings_;
  // Serial number to differentiate consecutive benchmark repeats.
  static int32_t iteration_;
};

int32_t ExchangeBenchmark::iteration_;

std::unique_ptr<ExchangeBenchmark> bm;

void runBenchmarks() {
  std::vector<RowVectorPtr> flat10k;
  //        std::vector<RowVectorPtr> deep10k;
  //        std::vector<RowVectorPtr> flat50;
  //        std::vector<RowVectorPtr> deep50;
  //        std::vector<RowVectorPtr> struct1k;
  //
  //        Counters flat10kCounters;
  //        Counters deep10kCounters;
  //        Counters flat50Counters;
  //        Counters deep50Counters;
  //        Counters localFlat10kCounters;
  //        Counters struct1kCounters;

  std::vector<std::string> flatNames = {"c0"};
  std::vector<TypePtr> flatTypes = {BIGINT()};

  // partitioning column cannot be dictionary
  std::vector<std::string> dictionaryNames = {"c0", "d0"};
  std::vector<TypePtr> dictionaryTypes = {BIGINT(), BIGINT()};
  //        std::vector<TypePtr> typeSelection = {
  //                BOOLEAN(),
  //                TINYINT(),
  //                DECIMAL(20, 3),
  //                INTEGER(),
  //                BIGINT(),
  //                REAL(),
  //                DECIMAL(10, 2),
  //                DOUBLE(),
  //                VARCHAR()};
  //
  //        int64_t flatSize = 0;
  //        // Add enough columns of different types to make a 10K row batch
  //        be
  //        // flat_batch_mb in flat size.
  //        while (flatSize * 10000 <
  //        static_cast<int64_t>(FLAGS_flat_batch_mb) << 20) {
  //            flatNames.push_back(fmt::format("c{}", flatNames.size()));
  //            assert(!flatNames.empty());
  //            flatTypes.push_back(typeSelection[flatTypes.size() %
  //            typeSelection.size()]); if
  //            (flatTypes.back()->isFixedWidth()) {
  //                flatSize += flatTypes.back()->cppSizeInBytes();
  //            } else {
  //                flatSize += 20;
  //            }
  //        }
  auto flatType = ROW(std::move(flatNames), std::move(flatTypes));
  auto dictionaryType =
      ROW(std::move(dictionaryNames), std::move(dictionaryTypes));
  //        auto structType = ROW(
  //                {{"c0", BIGINT()},
  //                 {"r1",
  //                        ROW(
  //                                {{"k2", BIGINT()},
  //                                 {"r2",
  //                                        ROW(
  //                                                {{"i1", BIGINT()},
  //                                                 {"i2", BIGINT()},
  //                                                 {"r3}, ROW({{s3",
  //                                                 VARCHAR()},
  //                                                 {"i5", INTEGER()},
  //                                                 {"d5", DOUBLE()},
  //                                                 {"b5", BOOLEAN()},
  //                                                 {"a5",
  //                                                 ARRAY(TINYINT())}})}})}});
  //
  //        auto deepType = ROW(
  //                {{"c0", BIGINT()},
  //                 {"long_array_val", ARRAY(ARRAY(BIGINT()))},
  //                 {"array_val", ARRAY(VARCHAR())},
  //                 {"struct_val", ROW({{"s_int", INTEGER()}, {"s_array",
  //                 ARRAY(REAL())}})},
  //                 {"map_val",
  //                         MAP(VARCHAR(),
  //                             MAP(BIGINT(),
  //                                 ROW({{"s2_int", INTEGER()},
  //                                 {"s2_string", VARCHAR()}})))}});

  std::vector<std::string> arrayNames = {"c0", "array"};
  std::vector<TypePtr> arrayTypes = {BIGINT(), ARRAY(BIGINT())};
  auto arrayType = ROW(std::move(arrayNames), std::move(arrayTypes));

  //  std::vector<RowVectorPtr>  flat10k = bm->makeRows(flatType, 100, 1000,
  //  FLAGS_dict_pct);
  std::vector<RowVectorPtr> dictionary10k =
      bm->makeRows(dictionaryType, 1000, 10000, FLAGS_dict_pct);
  //  deep10k = bm->makeRows(deepType, 10, 10000, FLAGS_dict_pct);
  //  flat50 = bm->makeRows(flatType, 2000, 50, FLAGS_dict_pct);
  //  deep50 = bm->makeRows(deepType, 2000, 50, FLAGS_dict_pct);
  //  struct1k = bm->makeRows(structType, 100, 1000, FLAGS_dict_pct);
  // auto array10k = bm->makeRows(arrayType, 100, 1000, FLAGS_dict_pct);

  PlanNodeStats leafRepartitionStats;
  PlanNodeStats finalRepartitionStats;
  PlanNodeStats exchangeStats;

  //  folly::addBenchmark(__FILE__, "exchangeFlat10k", [&]() {
  //    bm->run(
  //        flat10k,
  //        FLAGS_width,
  //        FLAGS_task_width,
  //        leafRepartitionStats,
  //        finalRepartitionStats,
  //        exchangeStats);
  //    return 1;
  //  });

  folly::addBenchmark(__FILE__, "exchangeFlat10k", [&]() {
    bm->run(
        dictionary10k,
        FLAGS_width,
        FLAGS_task_width,
        leafRepartitionStats,
        finalRepartitionStats,
        exchangeStats);
    return 1;
  });
  //
  //  folly::addBenchmark(__FILE__, "exchangeFlat50", [&]() {
  //    bm->run(flat50, FLAGS_width, FLAGS_task_width, flat50Counters);
  //    return 1;
  //  });
  //
  //  folly::addBenchmark(__FILE__, "exchangeDeep10k", [&]() {
  //    bm->run(deep10k, FLAGS_width, FLAGS_task_width, deep10kCounters);
  //    return 1;
  //  });
  //
  //  folly::addBenchmark(__FILE__, "exchangeDeep50", [&]() {
  //    bm->run(deep50, FLAGS_width, FLAGS_task_width, deep50Counters);
  //    return 1;
  //  });
  //
  //  folly::addBenchmark(__FILE__, "exchangeStruct1K", [&]() {
  //    bm->run(struct1k, FLAGS_width, FLAGS_task_width, struct1kCounters);
  //    return 1;
  //  });
  //
  //  folly::addBenchmark(__FILE__, "localFlat10k", [&]() {
  //    bm->runLocal(
  //        flat10k, FLAGS_width, FLAGS_num_local_tasks, localFlat10kCounters);
  //    return 1;
  //  });
  //
  // folly::addBenchmark(__FILE__, "exchangeArray10k", [&]() {
  //   bm->run(
  //       array10k,
  //       FLAGS_width,
  //       FLAGS_task_width,
  //       leafRepartitionStats,
  //       finalRepartitionStats,
  //       exchangeStats);
  //   return 1;
  // });

  folly::runBenchmarks();

  //        std::cout << "flat10k: " << flat10kCounters.toString() <<
  //        std::endl;
  //            << "flat50: " << flat50Counters.toString() << std::endl
  //            << "deep10k: " << deep10kCounters.toString() << std::endl
  //            << "deep50: " << deep50Counters.toString() << std::endl
  //            << "struct1k: " << struct1kCounters.toString() << std::endl;

  PlanNodeStats partitionedOutputStats;
  partitionedOutputStats += leafRepartitionStats;
  partitionedOutputStats += finalRepartitionStats;
  std::cout << "LeafPartitionOutput: " << leafRepartitionStats.toString()
            << std::endl
            << std::endl;
  std::cout << "FinalPartitionOutput: " << finalRepartitionStats.toString()
            << std::endl
            << std::endl;
  std::cout << "partitionedOutputStats: " << partitionedOutputStats.toString()
            << std::endl
            << std::endl;
  std::cout << "Exchange: " << exchangeStats.toString() << std::endl
            << std::endl;
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  parse::registerTypeResolver();
  if (!isRegisteredNamedVectorSerde(VectorSerde::Kind::kPresto)) {
    serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();
  }
  exec::ExchangeSource::registerFactory(exec::test::createLocalExchangeSource);

  bm = std::make_unique<ExchangeBenchmark>();
  runBenchmarks();
  bm.reset();

  return 0;
}

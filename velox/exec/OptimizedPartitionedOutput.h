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

#include "velox/exec/Operator.h"
#include "velox/exec/OutputBufferManager.h"
#include "velox/serializers/PartitioningSerializer.h"

#include <iostream>

namespace facebook::velox::exec {

using RowSet = folly::Range<const vector_size_t*>;

class OptimizedPartitionedOutput : public Operator {
 public:
  // Minimum flush size for non-final flush. 60KB + overhead fits a
  // network MTU of 64K.
  static constexpr uint64_t kMinDestinationSize = 60 * 1024;

  OptimizedPartitionedOutput(
      int32_t operatorId,
      DriverCtx* ctx,
      const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
      bool eagerFlush);

  //  ~OptimizedPartitionedOutput() {
  //    std::cout << "numSerializedPages_=" << numSerializedPages_;
  //
  //    int64_t totalSizes = 0;
  //    for (auto size : serializedPageSizes_) {
  //      totalSizes += size;
  //      //      std::cout << size << " ";
  //    }
  //
  //    int64_t avgPageSize =
  //        numSerializedPages_ == 0 ? -1 : totalSizes / numSerializedPages_;
  //    std::cout << " totalPageSizes=" << totalSizes
  //              << " avgPageSize=" << avgPageSize << std::endl;
  //  }

  void addInput(RowVectorPtr input) override;

  // Always returns nullptr. The action is to further process
  // unprocessed input. If all input has been processed, 'this' is in
  // a non-blocked state, otherwise blocked.
  RowVectorPtr getOutput() override;

  // always true but the caller will check isBlocked before adding input, hence
  // the blocked state does not accumulate input.
  bool needsInput() const override {
    return true;
  }

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  void close() override {
    //    destinations_.clear();
  }

 private:
  void flush();

  const std::string taskId_;
  // TODO: For now assume input is the same as output
  const RowTypePtr inputType_;
  const std::vector<column_index_t> keyChannels_;
  // Empty if column order in the output is exactly the same as in input.
  const std::vector<column_index_t> outputChannels_;
  const int32_t numDestinations_;

  const bool replicateNullsAndAny_;
  const std::weak_ptr<exec::OutputBufferManager> bufferManager_;
  const std::function<void()> bufferReleaseFn_;
  const int64_t maxBufferedBytes_;

  velox::memory::MemoryPool* pool_;
  std::unique_ptr<serializer::presto::IterativePartitioningSerializer>
      serializer_;

  BlockingReason blockingReason_{BlockingReason::kNotBlocked};
  ContinueFuture future_;
  bool finished_{false};

  std::vector<IOBufOutputStream> outputStreams_;

  std::vector<int64_t> serializedPageSizes_;
  int64_t numSerializedPages_ = 0;
  int64_t numNonZeroSerializedPages_ = 0;
};

} // namespace facebook::velox::exec

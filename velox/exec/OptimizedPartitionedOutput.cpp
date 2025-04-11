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

#include "velox/exec/OptimizedPartitionedOutput.h"

#include <iostream>

#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {

OptimizedPartitionedOutput::OptimizedPartitionedOutput(
    int32_t operatorId,
    DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
    bool eagerFlush)
    : Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "OptimizedPartitionedOutput"),
      taskId_(operatorCtx_->taskId()),
      inputType_(planNode->inputType()),
      keyChannels_(toChannels(planNode->inputType(), planNode->keys())),
      outputChannels_(calculateOutputChannels(
          planNode->inputType(),
          planNode->outputType(),
          planNode->outputType())),
      numDestinations_(planNode->numPartitions()),
      replicateNullsAndAny_(planNode->isReplicateNullsAndAny()),
      bufferManager_(OutputBufferManager::getInstanceRef()),
      // NOTE: 'bufferReleaseFn_' holds a reference on the associated task to
      // prevent it from deleting while there are output buffers being accessed
      // out of the partitioned output buffer manager such as in Prestissimo,
      // the http server holds the buffers while sending the data response.
      bufferReleaseFn_([task = operatorCtx_->task()]() {}),
      maxBufferedBytes_(ctx->task->queryCtx()
                            ->queryConfig()
                            .maxPartitionedOutputBufferSize()),
      //      maxSerializedPageBytes_(std::max<uint64_t>(
      //          kMinDestinationSize, // 60KB
      //          std::min<uint64_t>(
      //              1 << 20,
      //              maxBufferedBytes_ /*32MB*/ / numPartitions_))),
      pool_(pool()) {
  if (!planNode->isPartitioned()) {
    VELOX_USER_CHECK_EQ(numDestinations_, 1);
  }
  if (numDestinations_ == 1) {
    VELOX_USER_CHECK(keyChannels_.empty());
  }

  serializer::presto::PrestoVectorSerde::PrestoOptions options;
  options.compressionKind = common::stringToCompressionKind(
      operatorCtx_->driverCtx()->queryConfig().shuffleCompressionKind());
  options.minCompressionRatio = 0.8;

  std::unique_ptr<core::PartitionFunction> partitionFunction =
      numDestinations_ == 1
      ? nullptr
      : planNode->partitionFunctionSpec().create(numDestinations_);
  serializer_ =
      std::make_unique<serializer::presto::IterativePartitioningSerializer>(
          inputType_,
          numDestinations_,
          bufferReleaseFn_,
          options,
          std::move(partitionFunction),
          pool_);
}

BlockingReason OptimizedPartitionedOutput::isBlocked(ContinueFuture* future) {
  if (blockingReason_ != BlockingReason::kNotBlocked) {
    *future = std::move(future_);
    blockingReason_ = BlockingReason::kNotBlocked;
    return BlockingReason::kWaitForConsumer;
  }
  return BlockingReason::kNotBlocked;
}

void OptimizedPartitionedOutput::addInput(RowVectorPtr input) {
  // VLOG(0) << "OptimizedPartitionedOutput::addInput Adding input " <<
  // input->size() << " rows.";
  // TODO: replicateNullsAndAny_

  if (serializer_->bytesBuffered() + input->inMemoryBytes() >=
      maxBufferedBytes_) {
    //      >=   maxSerializedPageBytes_ * numPartitions_) {
    flush();
  }

  serializer_->append(input);
}

RowVectorPtr OptimizedPartitionedOutput::getOutput() {
  // VLOG(0) << "OptimizedPartitionedOutput::getOutput begin finished_: " <<
  // finished_;
  if (finished_) {
    return nullptr;
  }

  blockingReason_ = BlockingReason::kNotBlocked;

  if (noMoreInput_ || serializer_->bytesBuffered() >= maxBufferedBytes_) {
    //      serializer_->bytesBuffered() >=   maxSerializedPageBytes_ *
    //      numPartitions_) {
    // VLOG(0) << "OptimizedPartitionedOutput::getOutput. noMoreInput_: " <<
    // noMoreInput_ << " flushing";
    flush();
  }

  if (noMoreInput_) {
    const auto serializerStats = serializer_->runtimeStats();
    auto lockedStats = stats_.wlock();
    for (auto& pair : serializerStats) {
      lockedStats->addRuntimeStat(pair.first, pair.second);
    }

    bufferManager_.lock()->noMoreData(operatorCtx_->task()->taskId());
    finished_ = true;
  }
  // The input is fully processed, drop the reference to allow reuse.
  input_ = nullptr;
  return nullptr;
}

bool OptimizedPartitionedOutput::isFinished() {
  return finished_;
}

void OptimizedPartitionedOutput::flush() {
  auto flushedBytes = serializer_->bytesBuffered();
  auto flushedRows = serializer_->rowsBuffered();
  // VLOG(0) << "OptimizedPartitionedOutput::flush begin flushedBytes" <<
  // flushedBytes << " flushedRows:"
  //                << flushedRows;
  auto lockedStats = stats_.wlock();
  lockedStats->addOutputVector(flushedBytes, flushedRows);

  auto serializedPages = serializer_->flushUncompressed();

  for (uint32_t destination = 0; destination < numDestinations_;
       destination++) {
    auto& serializedPage = serializedPages[destination];
    // VLOG(0) << "OptimizedPartitionedOutput::flush Got serializedPage " <<
    // serializedPage->toString()
    //                    << "for destination " << destination << " taskId_:" <<
    //                    taskId_;
    numSerializedPages_++;
    if (serializedPage && serializedPage->numRows() > 0) {
      serializedPageSizes_.push_back(serializedPage->size());
      numNonZeroSerializedPages_++;

      bool blocked = bufferManager_.lock()->enqueue(
          taskId_, destination, std::move(serializedPage), &future_);
      if (blocked) {
        blockingReason_ = BlockingReason::kWaitForConsumer;
      }
    }
  }
}

} // namespace facebook::velox::exec

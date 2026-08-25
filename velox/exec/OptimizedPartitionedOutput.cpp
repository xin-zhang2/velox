/*
 * Copyright (c) International Business Machines Corporation
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

#include <limits>
#include <unordered_map>

#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/SerializedPage.h"
#include "velox/exec/Task.h"

namespace facebook::velox::exec {

OptimizedPartitionedOutput::OptimizedPartitionedOutput(
    int32_t operatorId,
    DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode)
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
      bufferManager_(DefaultOutputBufferManager::getInstanceRef()),
      // NOTE: 'bufferReleaseFn_' holds a reference on the associated task to
      // prevent it from deleting while there are output buffers being accessed
      // out of the partitioned output buffer manager such as in Prestissimo,
      // the http server holds the buffers while sending the data response.
      bufferReleaseFn_([task = operatorCtx_->task()]() {}),
      maxOutputBufferBytes_(ctx->task->queryCtx()
                                ->queryConfig()
                                .maxPartitionedOutputBufferSize()),
      pool_(pool()),
      partitionFunction_(
          numDestinations_ == 1 ? nullptr
                                : planNode->partitionFunctionSpec().create(
                                      numDestinations_,
                                      /*localExchange=*/false,
                                      true)) {
  if (!planNode->isPartitioned()) {
    VELOX_USER_CHECK_EQ(numDestinations_, 1);
  }
  if (numDestinations_ == 1) {
    VELOX_USER_CHECK(keyChannels_.empty());
  }

  serializer::presto::SerdeOpts options;
  options.compressionKind = common::stringToCompressionKind(
      operatorCtx_->driverCtx()->queryConfig().shuffleCompressionKind());
  options.minCompressionRatio = 0.8;

  initializeSerializerLayout();

  std::function<std::unique_ptr<OutputStreamListener>()> listenerFactory =
      nullptr;
  if (operatorCtx_->driverCtx()->queryConfig().isExchangeChecksumEnabled()) {
    listenerFactory = [bufferManager = bufferManager_]()
        -> std::unique_ptr<OutputStreamListener> {
      auto lockedBufferManager = bufferManager.lock();
      VELOX_CHECK_NOT_NULL(
          lockedBufferManager, "OutputBufferManager was already destructed");
      return lockedBufferManager->newListener();
    };
  }

  serializer_ = std::make_unique<
      serializer::presto::PrestoIterativePartitioningSerializer>(
      outputType_,
      numDestinations_,
      options,
      pool_,
      serializerInputByOutput_,
      std::move(listenerFactory));
}

void OptimizedPartitionedOutput::addInput(RowVectorPtr input) {
  auto serializerInput = prepareSerializerInput(input);

  if (serializer_->maxRowsBufferedPerPartition() >= kMaxRowsPerDestinationBeforeFlush ||
    serializer_->estimateBytesAfterAppend(serializerInput) >
      maxOutputBufferBytes_) {
    flush();
  }

  auto singlePartition = numDestinations_ == 1
      ? std::optional<uint32_t>{0u}
      : partitionFunction_->partition(*input, partitions_);

  // Replication preparation must complete before the main append: the
  // multi-partition append permutes the input's buffers in place, and the
  // serializer input shares child vectors with 'input', so neither the
  // null-key decode nor the deep copy may run after it.
  RowVectorPtr compactCopy;
  uint32_t replicaHome = 0;
  if (replicateNullsAndAny_ && numDestinations_ > 1 && input->size() > 0) {
    const auto& nullRows = nullKeyRows_.collect(*input, keyChannels_);
    replicatedRowIds_.clear();
    if (!replicatedAny_) {
      replicatedRowIds_.push_back(0);
    }
    nullRows.applyToSelected([&](vector_size_t row) {
      // Row 0 may already be in the set as the replicated "any" row.
      if (replicatedAny_ || row != 0) {
        replicatedRowIds_.push_back(row);
      }
    });

    if (!replicatedRowIds_.empty()) {
      if (singlePartition.has_value()) {
        // 'partitions_' is stale when the partition function reported a
        // single partition; the whole batch already lands on it.
        replicaHome = singlePartition.value();
      } else {
        // Force a uniform home so the unmodified main append delivers the
        // replica-home copy; a replicated row's hash partition is irrelevant
        // because the row reaches every destination anyway.
        replicaHome = 0;
        for (const auto row : replicatedRowIds_) {
          partitions_[row] = 0;
        }
      }
      compactCopy = copyRows(serializerInput, replicatedRowIds_);
    }
  }

  if (singlePartition.has_value()) {
    serializer_->append(serializerInput, *singlePartition);
  } else {
    serializer_->append(serializerInput, partitions_);
  }

  if (compactCopy != nullptr) {
    replicatedAny_ = true;
    {
      auto lockedStats = stats_.wlock();
      // The shuffle amplification: additional emitted copies. Both operands
      // originate as 32-bit values, so multiply in 64-bit.
      lockedStats->addRuntimeStat(
          "numReplicatedRows",
          RuntimeCounter(static_cast<int64_t>(
              uint64_t(compactCopy->size()) * uint64_t(numDestinations_ - 1))));
    }
    appendReplicatedRows(std::move(compactCopy), replicaHome, 0);
  }

  auto lockedStats = stats_.wlock();
  ++numAppends_;
  lockedStats->addRuntimeStat("numAppends", RuntimeCounter(1));
}

bool OptimizedPartitionedOutput::needsInput() const {
  // No new input while replica delivery is suspended: the pending compact
  // copy must reach its remaining destinations first.
  return blockingReason_ == BlockingReason::kNotBlocked &&
      !pendingReplication_.has_value();
}

RowVectorPtr OptimizedPartitionedOutput::getOutput() {
  if (finished_) {
    return nullptr;
  }

  blockingReason_ = BlockingReason::kNotBlocked;

  // Resume suspended replica delivery before any finalization: the remaining
  // destinations must receive the replicated rows even after noMoreInput().
  if (pendingReplication_.has_value()) {
    auto pending = std::move(pendingReplication_.value());
    pendingReplication_.reset();
    appendReplicatedRows(
        std::move(pending.compactCopy),
        pending.replicaHome,
        pending.nextDestination);
    // Re-suspended on a still-full output buffer; try again after it drains.
    if (blockingReason_ != BlockingReason::kNotBlocked) {
      return nullptr;
    }
  }

  if (noMoreInput_ || serializer_->bytesBuffered() >= maxOutputBufferBytes_ ||
    serializer_->maxRowsBufferedPerPartition() >= kMaxRowsPerDestinationBeforeFlush) {
    flush();
  }

  // If blocked, stop here. We avoid advancing operator state while blocked,
  // even if noMoreInput_ may already be true. The driver will resume and call
  // getOutput() again once the OutputBuffer has space.
  if (blockingReason_ != BlockingReason::kNotBlocked) {
    return nullptr;
  }

  if (noMoreInput_ && serializer_->bytesBuffered() == 0) {
    // TODO: merge serializer runtime stats into operator stats once
    // PrestoIterativePartitioningSerializer exposes runtimeStats().
    bufferManager_.lock()->noMoreData(operatorCtx_->task()->taskId());
    finished_ = true;
  }

  return nullptr;
}

BlockingReason OptimizedPartitionedOutput::isBlocked(ContinueFuture* future) {
  if (blockingReason_ != BlockingReason::kNotBlocked) {
    *future = std::move(future_);
    blockingReason_ = BlockingReason::kNotBlocked;
    return BlockingReason::kWaitForConsumer;
  }
  return BlockingReason::kNotBlocked;
}

bool OptimizedPartitionedOutput::isFinished() {
  return finished_;
}

void OptimizedPartitionedOutput::initializeSerializerLayout() {
  if (outputType_->size() == 0 || outputChannels_.empty()) {
    serializerInputType_ = outputType_;
    return;
  }

  std::unordered_map<column_index_t, column_index_t> outputToSerializerInput;
  outputToSerializerInput.reserve(outputChannels_.size());

  std::vector<std::string> names;
  std::vector<TypePtr> types;
  names.reserve(outputChannels_.size());
  types.reserve(outputChannels_.size());
  serializerInputByOutput_.reserve(outputChannels_.size());

  for (const auto outputChannel : outputChannels_) {
    auto it = outputToSerializerInput.find(outputChannel);
    if (it == outputToSerializerInput.end()) {
      const auto serializerInputChannel =
          static_cast<column_index_t>(serializerInputChannels_.size());
      serializerInputChannels_.push_back(outputChannel);
      names.push_back(inputType_->nameOf(outputChannel));
      types.push_back(inputType_->childAt(outputChannel));
      it =
          outputToSerializerInput.emplace(outputChannel, serializerInputChannel)
              .first;
    }
    serializerInputByOutput_.push_back(it->second);
  }

  serializerInputType_ = ROW(std::move(names), std::move(types));
}

RowVectorPtr OptimizedPartitionedOutput::prepareSerializerInput(
    const RowVectorPtr& input) const {
  VELOX_CHECK_NOT_NULL(input);

  if (serializerInputType_->size() == 0) {
    return std::make_shared<RowVector>(
        input->pool(),
        serializerInputType_,
        nullptr /*nulls*/,
        input->size(),
        std::vector<VectorPtr>{});
  }

  if (serializerInputChannels_.empty()) {
    input->loadedVector();
    return input;
  }

  std::vector<VectorPtr> serializerInputColumns;
  serializerInputColumns.reserve(serializerInputChannels_.size());
  for (auto channel : serializerInputChannels_) {
    auto loadedChild = BaseVector::loadedVectorShared(input->childAt(channel));
    serializerInputColumns.push_back(loadedChild);
  }

  return std::make_shared<RowVector>(
      input->pool(),
      serializerInputType_,
      nullptr /*nulls*/,
      input->size(),
      std::move(serializerInputColumns));
}

RowVectorPtr OptimizedPartitionedOutput::copyRows(
    const RowVectorPtr& source,
    const std::vector<vector_size_t>& rows) {
  const auto numRows = static_cast<vector_size_t>(rows.size());
  auto copy =
      BaseVector::create<RowVector>(serializerInputType_, numRows, pool_);

  // Coalesce runs of adjacent row indices ('rows' is sorted ascending).
  std::vector<BaseVector::CopyRange> ranges;
  vector_size_t targetIndex = 0;
  for (vector_size_t i = 0; i < numRows;) {
    vector_size_t runEnd = i + 1;
    while (runEnd < numRows && rows[runEnd] == rows[runEnd - 1] + 1) {
      ++runEnd;
    }
    ranges.push_back({rows[i], targetIndex, runEnd - i});
    targetIndex += runEnd - i;
    i = runEnd;
  }
  copy->copyRanges(source.get(), folly::Range(ranges.data(), ranges.size()));
  return copy;
}

RowVectorPtr OptimizedPartitionedOutput::makeReplicaChunk(
    const RowVectorPtr& compactCopy,
    const std::vector<uint32_t>& chunkDestinations) {
  const auto numRows = compactCopy->size();
  const auto chunkSize = static_cast<vector_size_t>(
      numRows * static_cast<vector_size_t>(chunkDestinations.size()));

  replicaPartitions_.resize(chunkSize);

  if (serializerInputType_->size() == 0) {
    // A dictionary wrap over a zero-child RowVector is degenerate; row counts
    // come from the partitions vector alone.
    vector_size_t offset = 0;
    for (const auto destination : chunkDestinations) {
      std::fill_n(replicaPartitions_.begin() + offset, numRows, destination);
      offset += numRows;
    }
    return std::make_shared<RowVector>(
        pool_,
        serializerInputType_,
        nullptr /*nulls*/,
        chunkSize,
        std::vector<VectorPtr>{});
  }

  auto indices = allocateIndices(chunkSize, pool_);
  auto* rawIndices = indices->asMutable<vector_size_t>();
  vector_size_t offset = 0;
  for (const auto destination : chunkDestinations) {
    for (vector_size_t row = 0; row < numRows; ++row) {
      rawIndices[offset] = row;
      replicaPartitions_[offset] = destination;
      ++offset;
    }
  }

  // Wrapping the compact copy is safe only because the copy is private to
  // this operator and no append ever mutates its buffers; the multi-partition
  // append scatters only the freshly allocated indices buffer. Wrapping the
  // live input instead would alias buffers the main append permutes in place.
  // The wrap must carry no wrapper nulls: the dictionary partition path
  // scatters wrapper nulls in place, and the replicated rows' nulls live in
  // the compact copy's children.
  std::vector<VectorPtr> children;
  children.reserve(compactCopy->childrenSize());
  for (const auto& child : compactCopy->children()) {
    children.push_back(
        BaseVector::wrapInDictionary(nullptr, indices, chunkSize, child));
  }
  return std::make_shared<RowVector>(
      pool_,
      serializerInputType_,
      nullptr /*nulls*/,
      chunkSize,
      std::move(children));
}

bool OptimizedPartitionedOutput::ensureReplicationAppendCapacity(
    const RowVectorPtr& appendInput) {
  if (blockingReason_ != BlockingReason::kNotBlocked) {
    return false;
  }
  // rowsBuffered() is an int32_t aggregate that byte thresholds do not bound
  // for zero- or narrow-column layouts; check its headroom in 64-bit.
  const int64_t rowsAfterAppend =
      static_cast<int64_t>(serializer_->rowsBuffered()) + appendInput->size();
  const bool needsFlush =
      serializer_->maxRowsBufferedPerPartition() >=
          kMaxRowsPerDestinationBeforeFlush ||
      serializer_->estimateBytesAfterAppend(appendInput) >
          maxOutputBufferBytes_ ||
      rowsAfterAppend > std::numeric_limits<vector_size_t>::max() ||
      replicationAppendsSinceLastFlush_ >= kMaxReplicationAppendsPerFlush;
  if (needsFlush) {
    flush();
    return blockingReason_ == BlockingReason::kNotBlocked;
  }
  return true;
}

void OptimizedPartitionedOutput::appendReplicatedRows(
    RowVectorPtr compactCopy,
    uint32_t replicaHome,
    uint32_t nextDestination) {
  const int64_t numRows = compactCopy->size();
  VELOX_CHECK_GT(numRows, 0);

  // Fall back to per-destination single-partition appends when a chunk would
  // cover at most one destination: a one-destination chunk is an identity
  // wrap, strictly worse than the single-partition append, which moves no
  // data at all.
  const int64_t destinationsPerChunk = kMaxReplicaRowsPerAppend / numRows;
  const bool useDictionaryChunks = destinationsPerChunk >= 2;

  std::vector<uint32_t> chunkDestinations;
  uint32_t destination = nextDestination;
  while (destination < static_cast<uint32_t>(numDestinations_)) {
    if (destination == replicaHome) {
      ++destination;
      continue;
    }

    if (useDictionaryChunks) {
      chunkDestinations.clear();
      uint32_t next = destination;
      while (next < static_cast<uint32_t>(numDestinations_) &&
             static_cast<int64_t>(chunkDestinations.size()) <
                 destinationsPerChunk) {
        if (next != replicaHome) {
          chunkDestinations.push_back(next);
        }
        ++next;
      }
      auto chunk = makeReplicaChunk(compactCopy, chunkDestinations);
      if (!ensureReplicationAppendCapacity(chunk)) {
        pendingReplication_ = PendingReplication{
            std::move(compactCopy), destination, replicaHome};
        return;
      }
      serializer_->append(chunk, replicaPartitions_);
      ++replicationAppendsSinceLastFlush_;
      destination = next;
    } else {
      if (!ensureReplicationAppendCapacity(compactCopy)) {
        pendingReplication_ = PendingReplication{
            std::move(compactCopy), destination, replicaHome};
        return;
      }
      // Single-partition appends never mutate the shared compact copy.
      serializer_->append(compactCopy, destination);
      ++replicationAppendsSinceLastFlush_;
      ++destination;
    }
  }
}

void OptimizedPartitionedOutput::flush() {
  replicationAppendsSinceLastFlush_ = 0;

  const auto flushedBytes = serializer_->bytesBuffered();
  const auto flushedRows = serializer_->rowsBuffered();

  // This will serialize all destinations and reset serializer_->bytesBuffered()
  // to 0.
  auto serializedIOBufs = serializer_->flush();
  auto bufferManager = bufferManager_.lock();
  VELOX_CHECK_NOT_NULL(
      bufferManager, "OutputBufferManager was already destructed");

  bool shouldBlock = false;
  ContinueFuture future = ContinueFuture::makeEmpty();
  for (auto& [destination, pageData] : serializedIOBufs) {
    // We will only pass the future to bufferManager->enqueue() for the first
    // blocked destination. This is to avoid unnecessary creation of
    // ContinueFuture objects for the remaining destinations.
    ContinueFuture* futurePtr = shouldBlock ? nullptr : &future;

    // Enqueue the data for each non-empty partition. Since the pageData is
    // already serialized, enqueueing them would not cause new memory
    // allocations. This will always move the pageData to the OutputBuffers no
    // matter if the OutputBuffer is blocked.
    bool blocked = bufferManager->enqueue(
        taskId_,
        static_cast<int>(destination),
        std::make_unique<PrestoSerializedPage>(
            std::move(pageData.first),
            [fn = bufferReleaseFn_](folly::IOBuf&) { fn(); },
            pageData.second),
        futurePtr);

    if (blocked && !shouldBlock) {
      blockingReason_ = BlockingReason::kWaitForConsumer;
      shouldBlock = true;
      future_ = std::move(future);
    }
  }

  auto lockedStats = stats_.wlock();
  if (flushedRows > 0) {
    lockedStats->addOutputVector(flushedBytes, flushedRows);
    ++numFlushes_;
    lockedStats->addRuntimeStat("numFlushes", RuntimeCounter(1));
  }
  if (shouldBlock) {
    ++numBlockedTimes_;
    lockedStats->addRuntimeStat("numBlockedTimes", RuntimeCounter(1));
  }
}

} // namespace facebook::velox::exec

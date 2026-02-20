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
#include "velox/serializers/PartitioningSerializer.h"

#include <map>

namespace facebook::velox::serializer::presto {

namespace {
constexpr int8_t kCompressedBitMask = 1;
constexpr int8_t kEncryptedBitMask = 2;
constexpr int8_t kCheckSumBitMask = 4;
// uncompressed size comes after the number of rows and the codec
constexpr int32_t kSizeInBytesOffset{4 + 1};
constexpr int32_t kHeaderSize{kSizeInBytesOffset + 4 + 4 + 8};

static inline const std::string_view kByteArray{"BYTE_ARRAY"};
static inline const std::string_view kShortArray{"SHORT_ARRAY"};
static inline const std::string_view kIntArray{"INT_ARRAY"};
static inline const std::string_view kLongArray{"LONG_ARRAY"};
static inline const std::string_view kInt128Array{"INT128_ARRAY"};
static inline const std::string_view kVariableWidth{"VARIABLE_WIDTH"};
static inline const std::string_view kArray{"ARRAY"};
static inline const std::string_view kMap{"MAP"};
static inline const std::string_view kRow{"ROW"};
static inline const std::string_view kRLE{"RLE"};
static inline const std::string_view kDictionary{"DICTIONARY"};

inline void
prefixSum(vector_size_t* offsets, uint32_t numPartitions, vector_size_t base) {
  offsets[0] += base;
  for (uint32_t i = 1; i < numPartitions; i++) {
    offsets[i] += offsets[i - 1];
  }
}

inline void writeInt32(OutputStream* out, int32_t value) {
  out->write(reinterpret_cast<char*>(&value), sizeof(value));
}

inline void writeInt64(OutputStream* out, int64_t value) {
  out->write(reinterpret_cast<char*>(&value), sizeof(value));
}

char getCodecMarker() {
  char marker = 0;
  marker |= kCheckSumBitMask;
  return marker;
}

std::string_view typeToEncodingName(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
      return kByteArray;
    case TypeKind::TINYINT:
      return kByteArray;
    case TypeKind::SMALLINT:
      return kShortArray;
    case TypeKind::INTEGER:
      return kIntArray;
    case TypeKind::BIGINT:
      return kLongArray;
    case TypeKind::HUGEINT:
      return kInt128Array;
    case TypeKind::REAL:
      return kIntArray;
    case TypeKind::DOUBLE:
      return kLongArray;
    case TypeKind::VARCHAR:
      return kVariableWidth;
    case TypeKind::VARBINARY:
      return kVariableWidth;
    case TypeKind::TIMESTAMP:
      return kLongArray;
    case TypeKind::ARRAY:
      return kArray;
    case TypeKind::MAP:
      return kMap;
    case TypeKind::ROW:
      return kRow;
    case TypeKind::UNKNOWN:
      return kByteArray;
    default:
      VELOX_FAIL("Unknown type kind: {}", static_cast<int>(type->kind()));
  }
  return "";
}

int64_t computeChecksum(
    serializer::presto::PrestoOutputStreamListener* listener,
    int codecMarker,
    int numRows,
    int uncompressedSize) {
  auto result = listener->crc();
  result.process_bytes(&codecMarker, 1);
  result.process_bytes(&numRows, 4);
  result.process_bytes(&uncompressedSize, 4);
  return result.checksum();
}

void rightShiftBits(uint8_t* bits, size_t length, uint8_t n) {
  if (n == 0 || length == 0) {
    return; // No shift needed
  }

  VELOX_CHECK_LT(n, 8);

  const uint8_t leftShift = 8 - n;
  uint8_t carry = 0; // To store bits that will be carried to the next word

  for (size_t i = length; i > 0; --i) {
    uint8_t current = bits[i - 1];
    bits[i - 1] = (current >> n) | carry;
    carry = current << leftShift;
  }
}

} // namespace

IterativePartitioningSerializer::IterativePartitioningSerializer(
    const RowTypePtr inputType,
    const RowTypePtr outputType,
    int32_t numDestinations,
    const std::function<void()>& bufferReleaseFn,
    const SerdeOpts& opts,
    std::unique_ptr<core::PartitionFunction> partitionFunction,
    memory::MemoryPool* pool)
    : inputType_(inputType),
      outputType_(outputType),
      numPartitions_(numDestinations),
      bufferManager_(exec::OutputBufferManager::getInstanceRef()),
      bufferReleaseFn_(bufferReleaseFn),
      codec_(common::compressionKindToCodec(opts.compressionKind)),
      partitionFunction_(std::move(partitionFunction)),
      streamArena_(pool),
      pool_(pool),
      topRowCounts_(numPartitions_, 0),
      bytesBuffered_(0),
      rowsBuffered_(0) {
  flushingHeader_.resize(25);
  std::fill(flushingHeader_.begin(), flushingHeader_.end(), 0);
  auto codecMask = getCodecMarker();
  flushingHeader_[5] = codecMask;
}

void IterativePartitioningSerializer::append(
    RowVectorPtr& input,
    RowVectorPtr& output) {
  // VLOG(0) << "IterativePartitioningSerializer::append appending input " <<
  // input->toString();
  numColumns_ = output->children().size();

  auto numRows = input->size();

  if (numPartitions_ == 1) {
    topRowPartitions_.assign(numRows, 0);
  } else {
    VELOX_CHECK(partitionFunction_);
    const auto singlePartition = partitionFunction_->partition(
        *input->as<RowVector>(), topRowPartitions_);
    if (singlePartition.has_value()) {
      topRowPartitions_.assign(numRows, singlePartition.value());
    }
  }

  BufferPtr partitionOffsetsBuffer;
  VectorPtr vector = std::dynamic_pointer_cast<BaseVector>(output);
  auto partitionedPage = PartitionedVector::create(
      vector,
      topRowPartitions_,
      topRowOffsetsForCurrentLevel_,
      topRowOffsetsForNextLevel_,
      0,
      numPartitions_,
      beginOffsetsBuffer_,
      partitionOffsetsBuffer,
      swappingBuffer_,
      0,
      pool_);
  //  VLOG(0) << "IterativePartitioningSerializer::append partitionedPage "
  //          << partitionedPage->toString();

  auto* partitionOffsets = partitionedPage->rawPartitionOffsets();
  vector_size_t offset = 0;
  vector_size_t totalRowsInPartitions = 0;
  for (auto i = 0; i < numPartitions_; i++) {
    const auto rowsInPartition = partitionOffsets[i] - offset;
    topRowCounts_[i] += rowsInPartition;
    totalRowsInPartitions += rowsInPartition;
    offset = partitionOffsets[i];
  }
  VELOX_CHECK_EQ(
      totalRowsInPartitions,
      numRows,
      "IterativePartitioningSerializer partition row count mismatch");

  partitionedPages_.emplace_back(partitionedPage);

  bytesBuffered_ += output->inMemoryBytes();
  rowsBuffered_ += numRows;
  if (VLOG_IS_ON(1)) {
    VLOG(1) << "IterativePartitioningSerializer append. rows=" << numRows
            << " bytesBuffered=" << bytesBuffered_
            << " rowsBuffered=" << rowsBuffered_
            << " bufferedPages=" << partitionedPages_.size();
  }
}

std::map<uint32_t, std::unique_ptr<exec::SerializedPage>>
IterativePartitioningSerializer::flushUncompressed() {
  //  VLOG(0) << "IterativePartitioningSerializer::flush begin ";

  if (partitionedPages_.empty()) {
    if (VLOG_IS_ON(1)) {
      VLOG(1)
          << "IterativePartitioningSerializer flush skipped: no buffered pages";
    }
    return std::map<uint32_t, std::unique_ptr<exec::SerializedPage>>();
  }
  if (VLOG_IS_ON(1)) {
    VLOG(1) << "IterativePartitioningSerializer flush begin. rowsBuffered="
            << rowsBuffered_ << " bytesBuffered=" << bytesBuffered_
            << " bufferedPages=" << partitionedPages_.size();
  }

  char codecMask = 0;

  // Flush headers for all destinations
  std::vector<IOBufOutputStream> outputStreams;
  std::vector<std::unique_ptr<OutputStreamListener>> listeners;
  listeners.reserve(numPartitions_);
  std::vector<int32_t> beginOffsets(numPartitions_, 0);
  for (uint32_t destination = 0; destination < numPartitions_; destination++) {
    listeners.emplace_back(bufferManager_.lock()->newListener());
    outputStreams.emplace_back(
        *pool_, listeners.back().get(), bytesBuffered_ / numPartitions_);
    auto& out = outputStreams[destination];

    auto prestoListener =
        dynamic_cast<serializer::presto::PrestoOutputStreamListener*>(
            out.listener());
    if (prestoListener) {
      prestoListener->reset();
      codecMask = getCodecMarker();
    }

    beginOffsets[destination] = (int32_t)out.tellp();
    flushStart(out, destination, codecMask);
  }

  flushPartitionedRowChildren(partitionedPages_, 0, outputStreams);

  std::map<uint32_t, std::unique_ptr<exec::SerializedPage>> serializedPages;
  int64_t totalFlushedRowsThisRound{0};
  for (uint32_t destination = 0; destination < numPartitions_; destination++) {
    auto& out = outputStreams[destination];
    flushFinish(out, destination, beginOffsets[destination], codecMask);

    const int64_t flushedBytes = out.tellp();
    if (flushedBytes > 0 && topRowCounts_[destination] > 0) {
      serializedPages[destination] = std::make_unique<exec::SerializedPage>(
          //          out.getIOBuf(), nullptr, topRowCounts_[destination]);
          out.getIOBuf(bufferReleaseFn_),
          nullptr,
          topRowCounts_[destination]);
      totalFlushedRowsThisRound += topRowCounts_[destination];

      totalFlushedBytes_ += flushedBytes;
      totalFlushedRows_ += topRowCounts_[destination];
      auto ranges = out.out().ranges();
      totalNumRanges_ += ranges.size();
    }
  }
  VELOX_CHECK_EQ(
      totalFlushedRowsThisRound,
      rowsBuffered_,
      "IterativePartitioningSerializer flushed row count mismatch");
  VELOX_CHECK(
      rowsBuffered_ == 0 || !serializedPages.empty(),
      "IterativePartitioningSerializer has buffered rows but produced no serialized pages");

  numFlushes_++;
  numSerializedPages_ += serializedPages.size();

  bytesBuffered_ = 0;
  rowsBuffered_ = 0;
  topRowCounts_.assign(topRowCounts_.size(), 0);
  partitionedPages_.clear();
  if (VLOG_IS_ON(1)) {
    VLOG(1) << "IterativePartitioningSerializer flush end. serializedPages="
            << serializedPages.size()
            << " flushedRows=" << totalFlushedRowsThisRound;
  }

  return serializedPages;
}

int64_t IterativePartitioningSerializer::bytesBuffered() {
  return bytesBuffered_;
}

int64_t IterativePartitioningSerializer::rowsBuffered() {
  return rowsBuffered_;
}

void IterativePartitioningSerializer::flushPartitionedRowChildren(
    const std::vector<PartitionedVectorPtr>& partitionedRowVectors,
    uint32_t nestedLevel,
    std::vector<IOBufOutputStream>& outputStreams) {
  std::vector<PartitionedVectorPtr> tempVectors(partitionedRowVectors.size());
  VELOX_CHECK_GT(partitionedRowVectors.size(), 0);
  int32_t numColumns =
      asRowType(partitionedRowVectors[0]->baseVector()->type())->size();
  for (uint32_t column = 0; column < numColumns; column++) {
    for (int i = 0; i < partitionedRowVectors.size(); i++) {
      tempVectors[i] =
          partitionedRowVectors[i]->as<PartitionedRowVector>()->childAt(column);
    }
    // flush column to output
    flushColumn(tempVectors, nestedLevel + 1, outputStreams);
  }
}

void IterativePartitioningSerializer::flushColumn(
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    uint32_t nestedLevel,
    std::vector<IOBufOutputStream>& outputStreams) {
  VELOX_CHECK_GT(partitionedVectors.size(), 0);

  // Switching on typeKind instead of encoding, because there could be multiple
  // PartitionedRowVectors buffered, and for the same column they could be
  // plain vectors without wrapping, or DictionaryVector, ConstantVector, or
  // BiasVector on any data types.
  auto typeKind = partitionedVectors[0]->baseVector()->typeKind();
  switch (typeKind) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::TIMESTAMP:
    case TypeKind::HUGEINT:
      flushSimpleColumn(partitionedVectors, nestedLevel, outputStreams);
      break;

    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      flushStringViewColumn(
          partitionedVectors, nestedLevel, outputStreams);
      break;

    case TypeKind::ARRAY:
      flushArrayColumn(partitionedVectors, nestedLevel, outputStreams);
      break;

    case TypeKind::ROW:
      flushRowColumn(partitionedVectors, nestedLevel, outputStreams);
      break;

    case TypeKind::MAP:
      VELOX_UNSUPPORTED(
          "Unsupported vector type for OptimizedPartitionedOutput: ", typeKind);
      break;

    default:
      VELOX_UNREACHABLE(
          "Invalid vector encoding for OptimizedPartitionedOutput: ", typeKind);
  }
}

void IterativePartitioningSerializer::flushRowColumn(
    const std::vector<PartitionedVectorPtr>& partitionedRowVectors,
    uint32_t nestedLevel,
    std::vector<IOBufOutputStream>& outputStreams) {
  VELOX_CHECK_GT(partitionedRowVectors.size(), 0);

  flushHeader(kRow, outputStreams);

  const int32_t numColumns =
      asRowType(partitionedRowVectors[0]->baseVector()->type())->size();
  for (auto& out : outputStreams) {
    writeInt32(&out, numColumns);
  }

  flushPartitionedRowChildren(partitionedRowVectors, nestedLevel, outputStreams);

  // For nullsFirst == false, ROW encoding appends size + offsets + nulls
  // after children. The current optimized path doesn't support null rows, so
  // offsets are always 0..N for each destination.
  std::vector<vector_size_t> rowCounts =
      nestedLevel == 1 ? topRowCounts_
                       : countRowsInPartitions(partitionedRowVectors, false);
  for (int destination = 0; destination < numPartitions_; ++destination) {
    writeInt32(&outputStreams[destination], rowCounts[destination]);
    writeInt32(&outputStreams[destination], 0);
    for (int32_t i = 1; i <= rowCounts[destination]; ++i) {
      writeInt32(&outputStreams[destination], i);
    }
  }

  flushNullFlag(partitionedRowVectors, outputStreams);
  flushNulls(partitionedRowVectors, outputStreams);
}

void IterativePartitioningSerializer::flushArrayColumn(
    const std::vector<PartitionedVectorPtr>& partitionedArrayVectors,
    uint32_t nestedLevel,
    std::vector<IOBufOutputStream>& outputStreams) {
  flushHeader(kArray, outputStreams);

  // flush children first
  std::vector<PartitionedVectorPtr> elementsVectors;
  for (auto& partitionedArrayVector : partitionedArrayVectors) {
    elementsVectors.push_back(
        partitionedArrayVector->as<PartitionedArrayVector>()->elements());
  }

  flushColumn(elementsVectors, nestedLevel + 1, outputStreams);

  flushRowCounts(partitionedArrayVectors, nestedLevel, outputStreams);

  flushOffsets(partitionedArrayVectors, outputStreams);

  // Flush mayHaveNulls byte
  flushNullFlag(partitionedArrayVectors, outputStreams);
  flushNulls(partitionedArrayVectors, outputStreams);
}

void IterativePartitioningSerializer::flushSimpleColumn(
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    uint32_t nestedLevel,
    std::vector<IOBufOutputStream>& outputStreams) {
  flushHeader(
      typeToEncodingName(partitionedVectors[0]->baseVector()->type()),
      outputStreams);

  flushRowCounts(partitionedVectors, nestedLevel, outputStreams);

  // Flush mayHaveNulls byte
  flushNullFlag(partitionedVectors, outputStreams);

  flushNulls(partitionedVectors, outputStreams);

  for (int i = 0; i < partitionedVectors.size(); i++) {
    flushPartitionedSimpleVector(partitionedVectors[i], outputStreams);
  }
}

void IterativePartitioningSerializer::flushStringViewColumn(
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    uint32_t nestedLevel,
    std::vector<IOBufOutputStream>& outputStreams) {
  flushHeader(
      typeToEncodingName(partitionedVectors[0]->baseVector()->type()),
      outputStreams);

  flushRowCounts(partitionedVectors, nestedLevel, outputStreams);

  std::vector<int32_t> cumulativeOffsets(numPartitions_, 0);

  for (const auto& partitionedVector : partitionedVectors) {
    auto* flatVector = partitionedVector->as<PartitionedFlatVector<StringView>>();
    const auto* values =
        flatVector->baseVector()->as<FlatVector<StringView>>()->rawValues();
    const auto* offsets = flatVector->rawPartitionOffsets();
    const auto* rawNulls = flatVector->baseVector()->rawNulls();

    vector_size_t lastOffset = 0;
    for (int32_t p = 0; p < numPartitions_; ++p) {
      const auto partitionEnd = offsets[p];
      if (!flatVector->indices()) {
        for (auto i = lastOffset; i < partitionEnd; ++i) {
          if (rawNulls != nullptr && bits::isBitNull(rawNulls, i)) {
            continue;
          }
          cumulativeOffsets[p] += values[i].size();
          writeInt32(&outputStreams[p], cumulativeOffsets[p]);
        }
      } else {
        const auto* indices = flatVector->indices()->as<vector_size_t>();
        for (auto i = lastOffset; i < partitionEnd; ++i) {
          const auto sourceRow = indices[i];
          if (rawNulls != nullptr && bits::isBitNull(rawNulls, sourceRow)) {
            continue;
          }
          cumulativeOffsets[p] += values[sourceRow].size();
          writeInt32(&outputStreams[p], cumulativeOffsets[p]);
        }
      }
      lastOffset = partitionEnd;
    }
  }

  flushNullFlag(partitionedVectors, outputStreams);
  flushNulls(partitionedVectors, outputStreams);

  for (int32_t p = 0; p < numPartitions_; ++p) {
    writeInt32(&outputStreams[p], cumulativeOffsets[p]);
  }

  for (const auto& partitionedVector : partitionedVectors) {
    auto* flatVector = partitionedVector->as<PartitionedFlatVector<StringView>>();
    const auto* values =
        flatVector->baseVector()->as<FlatVector<StringView>>()->rawValues();
    const auto* offsets = flatVector->rawPartitionOffsets();
    const auto* rawNulls = flatVector->baseVector()->rawNulls();

    vector_size_t lastOffset = 0;
    for (int32_t p = 0; p < numPartitions_; ++p) {
      const auto partitionEnd = offsets[p];
      if (!flatVector->indices()) {
        for (auto i = lastOffset; i < partitionEnd; ++i) {
          if (rawNulls != nullptr && bits::isBitNull(rawNulls, i)) {
            continue;
          }
          const auto value = values[i];
          if (value.size() > 0) {
            outputStreams[p].write(value.data(), value.size());
          }
        }
      } else {
        const auto* indices = flatVector->indices()->as<vector_size_t>();
        for (auto i = lastOffset; i < partitionEnd; ++i) {
          const auto sourceRow = indices[i];
          if (rawNulls != nullptr && bits::isBitNull(rawNulls, sourceRow)) {
            continue;
          }
          const auto value = values[sourceRow];
          if (value.size() > 0) {
            outputStreams[p].write(value.data(), value.size());
          }
        }
      }
      lastOffset = partitionEnd;
    }
  }
}

void IterativePartitioningSerializer::flushOffsets(
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    std::vector<IOBufOutputStream>& outputStreams) {
  auto typeWidth = sizeof(vector_size_t);

  // Add a zero to each destination at the beginning
  for (int p = 0; p < numPartitions_; p++) {
    writeInt32(&outputStreams[p], 0);
  }

  std::vector<vector_size_t> baseOffsets(numPartitions_, 0);
  for (auto& partitionedVector : partitionedVectors) {
    auto numRows = partitionedVector->baseVector()->size();
    const auto* partitionOffsets = partitionedVector->rawPartitionOffsets();
    vector_size_t* rawSizes =
        const_cast<vector_size_t*>(partitionedVector->rawSizes());

    // populate sizes using the indices as the buffer. This is ok because the
    // children have been flushed already and the indices are not needed anymore
    if (partitionedVector->indices()) {
      auto indicesBuffer = partitionedVector->indices();
      auto* indices = indicesBuffer->asMutable<vector_size_t>();

      ensureCapacity<vector_size_t>(swappingBuffer_, numRows, pool_);
      auto* swappingBuffer = swappingBuffer_->asMutable<vector_size_t>();
      for (auto i = 0; i < numRows; i++) {
        swappingBuffer[i] = rawSizes[indices[i]];
      }
      rawSizes = swappingBuffer;
    }

    auto partitionBegin = 0;
    for (int p = 0; p < numPartitions_; p++) {
      auto partitionEnd = partitionOffsets[p];
      auto numRawSizes = partitionEnd - partitionBegin;

      // Compute offsets from sizes
      prefixSum(&(rawSizes[partitionBegin]), numRawSizes, baseOffsets[p]);
      outputStreams[p].write(
          reinterpret_cast<const char*>(&rawSizes[partitionBegin]),
          numRawSizes * typeWidth);

      if (numRawSizes > 0) {
        baseOffsets[p] = rawSizes[partitionEnd - 1];
      }
      partitionBegin = partitionEnd;
    }
  }
}

void IterativePartitioningSerializer::flushNullFlag(
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    std::vector<IOBufOutputStream>& outputStreams) {
  mayHaveNullsForPartition_.assign(numPartitions_, 0);

  for (const auto& partitionedVector : partitionedVectors) {
    const auto* rawNulls = partitionedVector->baseVector()->rawNulls();
    if (!rawNulls) {
      continue;
    }

    const auto* partitionOffsets = partitionedVector->rawPartitionOffsets();
    const auto* indices = partitionedVector->indices()
        ? partitionedVector->indices()->as<vector_size_t>()
        : nullptr;

    vector_size_t partitionBegin = 0;
    for (int destination = 0; destination < numPartitions_; ++destination) {
      if (mayHaveNullsForPartition_[destination]) {
        partitionBegin = partitionOffsets[destination];
        continue;
      }

      const auto partitionEnd = partitionOffsets[destination];
      if (!indices) {
        for (auto row = partitionBegin; row < partitionEnd; ++row) {
          if (bits::isBitNull(rawNulls, row)) {
            mayHaveNullsForPartition_[destination] = 1;
            break;
          }
        }
      } else {
        for (auto row = partitionBegin; row < partitionEnd; ++row) {
          if (bits::isBitNull(rawNulls, indices[row])) {
            mayHaveNullsForPartition_[destination] = 1;
            break;
          }
        }
      }
      partitionBegin = partitionEnd;
    }
  }

  for (int destination = 0; destination < numPartitions_; destination++) {
    outputStreams[destination].write(&mayHaveNullsForPartition_[destination], 1);
  }
}

void IterativePartitioningSerializer::flushNulls(
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    std::vector<IOBufOutputStream>& outputStreams) {
  std::vector<uint8_t> pendingByte(numPartitions_, 0);
  std::vector<uint8_t> pendingBits(numPartitions_, 0);

  auto flushByte = [&](int destination, uint8_t byte) {
    // Presto null byte format is reverse-bit-order and null=1.
    byte = ~byte;
    bits::reverseBits(&byte, 1);
    outputStreams[destination].write(reinterpret_cast<const char*>(&byte), 1);
  };

  for (const auto& partitionedVector : partitionedVectors) {
    const auto* partitionOffsets = partitionedVector->rawPartitionOffsets();
    const auto* rawNulls = partitionedVector->baseVector()->rawNulls();
    const auto* indices = partitionedVector->indices()
        ? partitionedVector->indices()->as<vector_size_t>()
        : nullptr;

    vector_size_t partitionBegin = 0;
    for (int destination = 0; destination < numPartitions_; ++destination) {
      if (!mayHaveNullsForPartition_[destination]) {
        partitionBegin = partitionOffsets[destination];
        continue;
      }

      const auto partitionEnd = partitionOffsets[destination];
      for (auto row = partitionBegin; row < partitionEnd; ++row) {
        const auto sourceRow = indices ? indices[row] : row;
        const bool isNotNull =
            (rawNulls == nullptr) || bits::isBitSet(rawNulls, sourceRow);

        pendingByte[destination] |=
            static_cast<uint8_t>(isNotNull) << pendingBits[destination];
        ++pendingBits[destination];
        if (pendingBits[destination] == 8) {
          flushByte(destination, pendingByte[destination]);
          pendingByte[destination] = 0;
          pendingBits[destination] = 0;
        }
      }
      partitionBegin = partitionEnd;
    }
  }

  for (int destination = 0; destination < numPartitions_; ++destination) {
    if (mayHaveNullsForPartition_[destination] && pendingBits[destination] > 0) {
      flushByte(destination, pendingByte[destination]);
    }
  }
}

void IterativePartitioningSerializer::flushPartitionedSimpleVector(
    const PartitionedVectorPtr& partitionedVector,
    std::vector<IOBufOutputStream>& outputStreams) {
  auto encoding = partitionedVector->baseVector()->encoding();
  auto typeKind = partitionedVector->baseVector()->typeKind();

  switch (encoding) {
    case VectorEncoding::Simple::FLAT:
      return VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
          IterativePartitioningSerializer::flushFlatVectorValues,
          typeKind,
          partitionedVector,
          outputStreams);
    case VectorEncoding::Simple::BIASED:
    case VectorEncoding::Simple::SEQUENCE:
      VELOX_UNSUPPORTED(
          "Unsupported vector encoding for OptimizedPartitionedOutput: ",
          encoding);
    default:
      VELOX_UNREACHABLE(
          "Invalid vector encoding for OptimizedPartitionedOutput:flushPartitionedSimpleVector ",
          encoding);
  }
}

template <TypeKind kind>
void IterativePartitioningSerializer::flushFlatVectorValues(
    const PartitionedVectorPtr& partitionedVector,
    std::vector<IOBufOutputStream>& outputStreams) {
  using T = typename TypeTraits<kind>::NativeType;

  auto* flatVector = partitionedVector->as<PartitionedFlatVector<T>>();
  const auto* values =
      flatVector->baseVector()->template as<FlatVector<T>>()->rawValues();
  const auto* offsets = flatVector->rawPartitionOffsets();
  const auto* rawNulls = flatVector->baseVector()->rawNulls();

  if (!flatVector->indices()) {
    flushFlatValues<T>(values, offsets, rawNulls, outputStreams);
  } else {
    auto* indices = flatVector->indices()->template as<vector_size_t>();
    reMapAndFlushFlatValues<T>(values, offsets, indices, rawNulls, outputStreams);
  }
}

template <typename T>
void IterativePartitioningSerializer::flushFlatValues(
    const T* partitionedValues,
    const vector_size_t* partitionOffsets,
    const uint64_t* rawNulls,
    std::vector<IOBufOutputStream>& outputStreams) {
  auto lastOffset = 0;
  for (int p = 0; p < numPartitions_; p++) {
    auto offset = partitionOffsets[p];
    for (auto i = lastOffset; i < offset; ++i) {
      if (rawNulls != nullptr && bits::isBitNull(rawNulls, i)) {
        continue;
      }
      outputStreams[p].write(
          reinterpret_cast<const char*>(&partitionedValues[i]), sizeof(T));
    }
    lastOffset = offset;
  }
}

template <typename T>
void IterativePartitioningSerializer::reMapAndFlushFlatValues(
    const T* values,
    const vector_size_t* partitionOffsets,
    const vector_size_t* partitionedIndices,
    const uint64_t* rawNulls,
    std::vector<IOBufOutputStream>& outputStreams) {
  auto lastOffset = 0;
  for (int p = 0; p < numPartitions_; p++) {
    auto indicesOffset = partitionOffsets[p];

    for (auto i = lastOffset; i < indicesOffset; ++i) {
      const auto sourceRow = partitionedIndices[i];
      if (rawNulls != nullptr && bits::isBitNull(rawNulls, sourceRow)) {
        continue;
      }
      outputStreams[p].write(
          reinterpret_cast<const char*>(&values[sourceRow]), sizeof(T));
    }

    lastOffset = indicesOffset;
  }
}

void IterativePartitioningSerializer::flushHeader(
    const std::string_view& name,
    std::vector<IOBufOutputStream>& outputStreams) {
  auto numBytes = name.size();
  for (int destination = 0; destination < numPartitions_; destination++) {
    writeInt32(&outputStreams[destination], numBytes);
    outputStreams[destination].write(&name[0], numBytes);
  }
}

void IterativePartitioningSerializer::flushRowCounts(
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    uint32_t nestedLevel,
    std::vector<IOBufOutputStream>& outputStreams) {
  auto rowCounts = nestedLevel == 1 ? topRowCounts_ : rowCountsForLevel_;
  // topRowCounts_ was already calculated in append(). We only need to
  // calcualte the nested levels
  if (nestedLevel > 1) {
    rowCounts.resize(numPartitions_);
    std::fill(rowCounts.begin(), rowCounts.end(), 0);
    for (auto& partitionedVector : partitionedVectors) {
      auto* partitionsOffsets = partitionedVector->rawPartitionOffsets();
      vector_size_t lastOffset = 0;
      for (auto i = 0; i < numPartitions_; ++i) {
        rowCounts[i] += partitionsOffsets[i] - lastOffset;
        lastOffset = partitionsOffsets[i];
      }
    }
  }

  for (int destination = 0; destination < numPartitions_; destination++) {
    // Write row counts for each destination
    writeInt32(&outputStreams[destination], rowCounts[destination]);
  }
}

void IterativePartitioningSerializer::flushStart(
    IOBufOutputStream& out,
    uint32_t destination,
    char codecMask) {
  auto prestoListener =
      dynamic_cast<serializer::presto::PrestoOutputStreamListener*>(
          out.listener());
  if (prestoListener) {
    prestoListener->pause();
  }

  // Write to flushingHeader_ the following: 1) the number of rows, 2)
  // codecMask, 3) Place holder for uncompressedSizeInBytes 4) Place holder
  // for sizeInBytes 4) Place holder for checksum, then write it to the output
  // stream. This is to avoid multiple small writes to the output stream.
  std::memcpy(
      &flushingHeader_[0], &topRowCounts_[destination], sizeof(vector_size_t));
  std::memcpy(
      &flushingHeader_[sizeof(vector_size_t)], &codecMask, sizeof(char));
  out.write(&flushingHeader_[0], 21);

  // Number of columns and stream content. Unpause CRC.
  if (prestoListener) {
    prestoListener->resume();
  }

  //   Write number of columns
  writeInt32(&out, numColumns_);
}

void IterativePartitioningSerializer::flushFinish(
    IOBufOutputStream& out,
    uint32_t destination,
    int32_t beginOffset,
    char codecMask) {
  auto prestoListener =
      dynamic_cast<serializer::presto::PrestoOutputStreamListener*>(
          out.listener());
  if (prestoListener) {
    prestoListener->pause();
  }

  // Fill in uncompressedSizeInBytes & sizeInBytes
  int32_t size = (int32_t)out.tellp() - beginOffset;
  const int32_t uncompressedSize = size - kHeaderSize;
  int64_t crc = 0;
  if (prestoListener) {
    crc = computeChecksum(
        prestoListener,
        codecMask,
        topRowCounts_[destination],
        uncompressedSize);
  }

  out.seekp(beginOffset + kSizeInBytesOffset);
  writeInt32(&out, uncompressedSize);
  writeInt32(&out, uncompressedSize);
  writeInt64(&out, crc);
  out.seekp(beginOffset + size);
}

std::vector<vector_size_t>
IterativePartitioningSerializer::countRowsInPartitions(
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    bool isTopLevel) {
  auto& rowCounts = isTopLevel ? topRowCounts_ : rowCountsForLevel_;
  rowCounts.resize(numPartitions_);
  std::fill(rowCounts.begin(), rowCounts.end(), 0);
  for (auto& partitionedVector : partitionedVectors) {
    auto* partitionsOffsets = partitionedVector->rawPartitionOffsets();
    vector_size_t lastOffset = 0;
    for (auto i = 0; i < numPartitions_; ++i) {
      rowCounts[i] += partitionsOffsets[i] - lastOffset;
      lastOffset = partitionsOffsets[i];
    }
  }
  return rowCounts;
}

std::unordered_map<std::string, RuntimeCounter>
IterativePartitioningSerializer::runtimeStats() {
  std::unordered_map<std::string, RuntimeCounter> map;
  map.insert(
      {{"compressedBytes",
        RuntimeCounter(
            compressionStats_.compressedBytes, RuntimeCounter::Unit::kBytes)},
       {"compressionInputBytes",
        RuntimeCounter(
            compressionStats_.compressionInputBytes,
            RuntimeCounter::Unit::kBytes)},
       {"compressionSkippedBytes",
        RuntimeCounter(
            compressionStats_.compressionSkippedBytes,
            RuntimeCounter::Unit::kBytes)}});
  return map;
}

} // namespace facebook::velox::serializer::presto

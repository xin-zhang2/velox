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
    int32_t numDestinations,
    const std::function<void()>& bufferReleaseFn,
    const SerdeOpts& opts,
    std::unique_ptr<core::PartitionFunction> partitionFunction,
    memory::MemoryPool* pool)
    : inputType_(inputType),
      outputType_(inputType),
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

void IterativePartitioningSerializer::append(RowVectorPtr& input) {
  // VLOG(0) << "IterativePartitioningSerializer::append appending input " <<
  // input->toString();
  numColumns_ = input->children().size();

  auto rowType = asRowType(input->type());
  auto numRows = input->size();

  if (numPartitions_ > 1) {
    VELOX_CHECK(partitionFunction_);
    partitionFunction_->partition(*input->as<RowVector>(), topRowPartitions_);
  }

  BufferPtr partitionOffsetsBuffer;
  VectorPtr vector = std::dynamic_pointer_cast<BaseVector>(input);
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
  for (auto i = 0; i < numPartitions_; i++) {
    topRowCounts_[i] += partitionOffsets[i] - offset;
    offset = partitionOffsets[i];
  }

  partitionedPages_.emplace_back(partitionedPage);

  bytesBuffered_ += input->inMemoryBytes();
  rowsBuffered_ += numRows;
}

std::map<uint32_t, std::unique_ptr<exec::SerializedPage>>
IterativePartitioningSerializer::flushUncompressed() {
  //  VLOG(0) << "IterativePartitioningSerializer::flush begin ";

  if (partitionedPages_.empty()) {
    return std::map<uint32_t, std::unique_ptr<exec::SerializedPage>>();
  }

  char codecMask = 0;

  // Flush headers for all destinations
  std::vector<IOBufOutputStream> outputStreams;
  std::vector<int32_t> beginOffsets(numPartitions_, 0);
  for (uint32_t destination = 0; destination < numPartitions_; destination++) {
    auto listener = bufferManager_.lock()->newListener();
    outputStreams.emplace_back(
        *pool_, listener.get(), bytesBuffered_ / numPartitions_);
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

      totalFlushedBytes_ += flushedBytes;
      totalFlushedRows_ += topRowCounts_[destination];
      auto ranges = out.out().ranges();
      totalNumRanges_ += ranges.size();
    }
  }

  numFlushes_++;
  numSerializedPages_ += serializedPages.size();

  bytesBuffered_ = 0;
  rowsBuffered_ = 0;
  topRowCounts_.assign(topRowCounts_.size(), 0);
  partitionedPages_.clear();

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
  int32_t numColumns = outputType_->children().size();
  for (uint32_t column = 0; column < numColumns; column++) {
    for (int i = 0; i < partitionedRowVectors.size(); i++) {
      tempVectors[i] =
          partitionedRowVectors[i]->as<PartitionedRowVector>()->childAt(column);
    }
    // flush column to output
    auto typeKind = outputType_->childAt(column)->kind();
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
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
    case TypeKind::TIMESTAMP:
    case TypeKind::HUGEINT:
      return flushSimpleColumn(partitionedVectors, nestedLevel, outputStreams);

    case TypeKind::ARRAY:
      return flushArrayColumn(partitionedVectors, nestedLevel, outputStreams);

    case TypeKind::ROW:
      return flushRowColumn(partitionedVectors, nestedLevel, outputStreams);

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
  //  VELOX_CHECK_GT(partitionedRowVectors.size(), 0);
  //
  //  flushHeader(kRow, outputStreams);
  //
  //  // Write number of columns to all outputStreams
  //  int32_t numColumns =
  //      asRowType(partitionedRowVectors[0]->baseVector()->type())->children().size();
  //  for (auto& out : outputStreams) {
  //    writeInt32(&out, numColumns);
  //  }
  //
  //  flushRowChildren(partitionedRowVectors, nestedLevel, outputStreams);
  //
  //  flushRowCounts(partitionedRowVectors, nestedLevel, outputStreams);
  //
  //  // flush lengths_
  //  for (auto& partitionedArrayVector : partitionedRowVectors) {
  //    const auto* partitionOffsets =
  //        partitionedArrayVector->rawPartitionOffsets();
  //    const auto* rawSizes =
  //        partitionedArrayVector->baseVector()->as<RowVectorPtr>()->rawSizes();
  //    flushFlatValues<vector_size_t>(rawSizes, partitionOffsets,
  //    outputStreams);
  //  }
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

      baseOffsets[p] = rawSizes[partitionEnd - 1];
      partitionBegin = partitionEnd;
    }
  }
}

void IterativePartitioningSerializer::flushNullFlag(
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    std::vector<IOBufOutputStream>& outputStreams) {
  // TODO: for simplicity we only check the whole vector now. The actual
  // mayHaveNulls value is one per destination
  char mayHaveNulls = 0;
  for (int i = 0; i < partitionedVectors.size(); i++) {
    PartitionedVectorPtr vector = partitionedVectors[i];
    if (vector->baseVector()->mayHaveNulls()) {
      mayHaveNulls = 1;
      VELOX_NYI("Partitioning vector with nulls is not supported yet.");
    }
  }
  for (int destination = 0; destination < numPartitions_; destination++) {
    outputStreams[destination].write(&mayHaveNulls, 1);
  }
}

void IterativePartitioningSerializer::flushNulls(
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    std::vector<IOBufOutputStream>& outputStreams) {
  std::vector<uint8_t> carryOver(numPartitions_, 0);
  std::vector<uint8_t> carryOverBits(numPartitions_, 0);

  for (auto& partitionedVector : partitionedVectors) {
    auto numRows = partitionedVector->baseVector()->size();
    auto numBytes = bits::nbytes(numRows);

    uint8_t* nulls = (uint8_t*)partitionedVector->baseVector()->rawNulls();

    if (!nulls) {
      continue;
    }

    if (partitionedVector->indices()) {
      // Remap nulls using the indices
      auto indicesBuffer = partitionedVector->indices();
      auto* indices = indicesBuffer->asMutable<vector_size_t>();

      ensureCapacity<char>(swappingBuffer_, numBytes, pool_, false, true);
      auto* swappingBuffer = swappingBuffer_->asMutable<uint8_t>();
      for (auto i = 0; i < numRows; i++) {
        size_t srcByte = i / 8;
        uint8_t srcBit = i - srcByte * 8;

        // Determine the destination byte and bit positions
        auto destIndice = indices[i];
        size_t destByte = destIndice / 8;
        uint8_t destBit = destIndice - destByte * 8;

        // Extract the bit from the source position and set it at the
        // destination
        swappingBuffer[destByte] |= ((nulls[srcByte] >> srcBit) & 1) << destBit;
      }
      nulls = swappingBuffer;
    }

    const auto* partitionOffsets = partitionedVector->rawPartitionOffsets();

    vector_size_t lastBit = 0;
    for (auto p = 0; p < numPartitions_; ++p) {
      int startBit = lastBit;
      int endBit = partitionOffsets[p];
      int numBitsInPartition = endBit - startBit;

      if (numBitsInPartition <= 0) {
        continue; // Skip empty partitions
      }

      int startByte = startBit / 8;
      int startBitOffset = startBit - startByte * 8;
      int endByte = (endBit - 1) / 8;
      int endBitOffset = endBit - endByte * 8;

      int bitOffset = startBitOffset;
      uint8_t currentByte = nulls[startByte];

      // Handle carry-over from the previous p
      uint8_t& carry = carryOver[p];
      uint8_t& numCarryBits = carryOverBits[p];

      int bitsToTake = std::min(8 - numCarryBits, numBitsInPartition);
      uint8_t mask = (1 << bitsToTake) - 1;
      uint8_t bits = (currentByte >> bitOffset) & mask;
      carry |= (bits << numCarryBits);
      numCarryBits += bitsToTake;
      bitOffset += bitsToTake;

      // Write full bytes from carry-over
      int writeCondition = (numCarryBits == 8);
      numCarryBits *= (1 - writeCondition); // Reset to 0 if writing
      outputStreams[p].write(
          reinterpret_cast<const char*>(&carry), writeCondition);
      carry *= (1 - writeCondition); // Reset to 0 if writing

      numBitsInPartition -= bitsToTake;
      bitOffset *= (numBitsInPartition > 0); // Reset to 0 if no more bits to process
      startByte += (bitOffset == 0 && numBitsInPartition > 0); // Move to next byte if needed
      currentByte =
          static_cast<uint8_t>(nulls[startByte]) * (startByte <= endByte);

      // Process full bytes
      while (numBitsInPartition >= 8) {
        uint8_t nextByte = static_cast<uint8_t>(nulls[startByte + 1]) *
            (startByte + 1 <= endByte);
        uint8_t combinedByte =
            (currentByte >> bitOffset) | (nextByte << (8 - bitOffset));
        outputStreams[p].write(reinterpret_cast<const char*>(&combinedByte), 1);
        numBitsInPartition -= 8;
        startByte += 1;
        currentByte =
            static_cast<uint8_t>(nulls[startByte]) * (startByte <= endByte);
      }

      // Handle remaining bits
      mask = (1 << numBitsInPartition) - 1;
      bits = (currentByte >> bitOffset) & mask;
      carry = bits;
      numCarryBits = numBitsInPartition;
    }
  }

  // Flush remaining carry-over bits
  for (size_t p = 0; p < numPartitions_; ++p) {
    if (carryOverBits[p] > 0) {
      outputStreams[p].write(reinterpret_cast<const char*>(&carryOver[p]), 1);
      carryOver[p] = 0;
      carryOverBits[p] = 0;
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

  if (!flatVector->indices()) {
    flushFlatValues<T>(values, offsets, outputStreams);
  } else {
    auto* indices = flatVector->indices()->template as<vector_size_t>();
    reMapAndFlushFlatValues<T>(values, offsets, indices, outputStreams);
  }
}

template <typename T>
void IterativePartitioningSerializer::flushFlatValues(
    const T* partitionedValues,
    const vector_size_t* partitionOffsets,
    std::vector<IOBufOutputStream>& outputStreams) {
  auto typeWidth = sizeof(T);

  auto lastOffset = 0;
  for (int p = 0; p < numPartitions_; p++) {
    auto offset = partitionOffsets[p];
    auto numValues = offset - lastOffset;
    outputStreams[p].write(
        reinterpret_cast<const char*>(&partitionedValues[lastOffset]),
        numValues * typeWidth);
    lastOffset = offset;
  }
}

template <typename T>
void IterativePartitioningSerializer::reMapAndFlushFlatValues(
    const T* values,
    const vector_size_t* partitionOffsets,
    const vector_size_t* partitionedIndices,
    std::vector<IOBufOutputStream>& outputStreams) {
  auto typeWidth = sizeof(T);
  auto lastOffset = 0;
  for (int p = 0; p < numPartitions_; p++) {
    auto indicesOffset = partitionOffsets[p];

    for (auto i = lastOffset; i < indicesOffset; ++i) {
      outputStreams[p].write(
          reinterpret_cast<const char*>(&values[partitionedIndices[i]]),
          typeWidth);
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

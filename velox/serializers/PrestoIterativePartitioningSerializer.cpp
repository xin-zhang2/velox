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
#include "velox/serializers/PrestoIterativePartitioningSerializer.h"

#include <algorithm>
#include <optional>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Nulls.h"
#include "velox/common/base/SimdUtil.h"
#include "velox/type/Type.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/ConstantVector.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::serializer::presto {

namespace {

constexpr int8_t kCheckSumBitMask = 4;
constexpr int64_t kVectorSizeTypeSize{sizeof(vector_size_t)};
// [numRows:4][codec:1]
constexpr int64_t kUncompressedSizeOffset{kVectorSizeTypeSize + 1};
// [numRows:4][codec:1][uncompressedSize:4][compressedSize:4][checksum:8]
constexpr int64_t kHeaderSize{kUncompressedSizeOffset + 4 + 4 + 8};

// chunk size for flushing constant values
constexpr int32_t kChunkBytes = 4096;

static inline const std::string_view kByteArray{"BYTE_ARRAY"};
static inline const std::string_view kShortArray{"SHORT_ARRAY"};
static inline const std::string_view kIntArray{"INT_ARRAY"};
static inline const std::string_view kLongArray{"LONG_ARRAY"};
static inline const std::string_view kInt128Array{"INT128_ARRAY"};
static inline const std::string_view kVariableWidth{"VARIABLE_WIDTH"};
static inline const std::string_view kRow{"ROW"};

inline void writeInt32(OutputStream* out, int32_t value) {
  out->write(reinterpret_cast<const char*>(&value), sizeof(value));
}

inline void writeInt64(OutputStream* out, int64_t value) {
  out->write(reinterpret_cast<const char*>(&value), sizeof(value));
}

char getCodecMarker(bool checksumEnabled) {
  char marker = 0;
  if (checksumEnabled) {
    marker |= kCheckSumBitMask;
  }
  return marker;
}

std::string_view typeToEncodingName(const TypePtr& type) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
      return kByteArray;
    case TypeKind::SMALLINT:
      return kShortArray;
    case TypeKind::INTEGER:
    case TypeKind::REAL:
      return kIntArray;
    case TypeKind::BIGINT:
    case TypeKind::DOUBLE:
    case TypeKind::TIMESTAMP:
      return kLongArray;
    case TypeKind::HUGEINT:
      return kInt128Array;
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      return kVariableWidth;
    case TypeKind::ROW:
      return kRow;
    default:
      VELOX_FAIL("Unsupported type kind: {}", static_cast<int>(type->kind()));
  }
}

/// Finalizes the Presto page CRC by mixing in the codec marker, row count,
/// and uncompressed size on top of the listener's accumulated data checksum.
int64_t computeChecksum(
    PrestoOutputStreamListener& listener,
    int8_t codecMarker,
    int32_t numRows,
    int32_t uncompressedSize) {
  auto crc = listener.crc();
  crc.process_bytes(&codecMarker, 1);
  crc.process_bytes(&numRows, 4);
  crc.process_bytes(&uncompressedSize, 4);
  return static_cast<int64_t>(crc.checksum());
}

/// Returns the serialized byte width of a fixed-width type, matching the
/// sizeof(T) used in flushFlatValues.
int32_t fixedTypeWidth(TypeKind kind) {
  switch (kind) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
      return 1;
    case TypeKind::SMALLINT:
      return 2;
    case TypeKind::INTEGER:
    case TypeKind::REAL:
      return 4;
    case TypeKind::BIGINT:
    case TypeKind::DOUBLE:
      return 8;
    case TypeKind::TIMESTAMP:
    case TypeKind::HUGEINT:
      return 16;
    default:
      return 0;
  }
}

/// Returns the exact bytes for one fixed-width column in one partition.
int64_t
simpleColumnBytes(const TypePtr& colType, int64_t numRows, int64_t numNulls) {
  const auto encodingName = typeToEncodingName(colType);
  return 4 + static_cast<int64_t>(encodingName.size()) + // header
      4 + // rowCount
      1 + // nullFlag
      (numNulls > 0 ? bits::nbytes(numRows) : 0) + // null bitmap
      (numRows - numNulls) * fixedTypeWidth(colType->kind()); // values
}

int64_t variableWidthColumnBytes(
    int64_t numRows,
    int64_t numNulls,
    int64_t valueBytes) {
  return 4 + static_cast<int64_t>(kVariableWidth.size()) + // header
      4 + // rowCount
      4 * numRows + // offsets
      1 + // nullFlag
      (numNulls > 0 ? bits::nbytes(numRows) : 0) + // null bitmap
      4 + // valueBytes
      valueBytes; // values
}

int64_t variableWidthDataBytes(const BaseVector& vector) {
  switch (vector.encoding()) {
    case VectorEncoding::Simple::FLAT: {
      const auto* flatVector = vector.asFlatVector<StringView>();
      VELOX_DCHECK_NOT_NULL(flatVector);

      const auto* rawValues = flatVector->rawValues();
      const auto* rawNulls = vector.rawNulls();

      int64_t dataBytes = 0;
      if (!rawNulls) {
        for (vector_size_t i = 0; i < vector.size(); ++i) {
          dataBytes += rawValues[i].size();
        }
      } else {
        for (vector_size_t i = 0; i < vector.size(); ++i) {
          if (!bits::isBitNull(rawNulls, i)) {
            dataBytes += rawValues[i].size();
          }
        }
      }
      return dataBytes;
    }
    case VectorEncoding::Simple::CONSTANT: {
      const auto* constantVector = vector.as<ConstantVector<StringView>>();
      VELOX_DCHECK_NOT_NULL(constantVector);

      if (constantVector->isNullAt(0)) {
        return 0;
      }

      return static_cast<int64_t>(vector.size()) *
          constantVector->valueAt(0).size();
    }
    case VectorEncoding::Simple::DICTIONARY: {
      const auto* simpleVector = vector.as<SimpleVector<StringView>>();
      VELOX_DCHECK_NOT_NULL(simpleVector);

      int64_t dataBytes = 0;
      for (vector_size_t i = 0; i < vector.size(); ++i) {
        if (!simpleVector->isNullAt(i)) {
          dataBytes += simpleVector->valueAt(i).size();
        }
      }
      return dataBytes;
    }
    case VectorEncoding::Simple::BIASED:
    case VectorEncoding::Simple::SEQUENCE:
      VELOX_NYI(
          "Unsupported vector encoding for variable-width size estimation: {}",
          vector.encoding());
    default:
      VELOX_UNSUPPORTED(
          "Invalid vector encoding for variable-width size estimation: {}",
          vector.encoding());
  }
}

void accumulateVariableWidthOffsetsForFlatVector(
    const PartitionedVectorPtr& partitionedVector,
    std::vector<std::vector<int32_t>>& offsetsPerPartition) {
  auto* flatVector = partitionedVector->as<PartitionedFlatVector<StringView>>();
  VELOX_DCHECK_NOT_NULL(flatVector);

  const auto* rawValues =
      flatVector->baseVector()->asFlatVector<StringView>()->rawValues();
  const auto* rawNulls = flatVector->baseVector()->rawNulls();
  const auto* partitionOffsets = partitionedVector->rawPartitionOffsets();

  vector_size_t lastPartitionOffset = 0;
  for (uint32_t p = 0; p < offsetsPerPartition.size(); ++p) {
    const auto partitionOffset = partitionOffsets[p];
    auto& offsets = offsetsPerPartition[p];
    int32_t endOffset = offsets.empty() ? 0 : offsets.back();
    if (!rawNulls) {
      for (auto i = lastPartitionOffset; i < partitionOffset; ++i) {
        endOffset += rawValues[i].size();
        offsets.push_back(endOffset);
      }
    } else {
      for (auto i = lastPartitionOffset; i < partitionOffset; ++i) {
        if (!bits::isBitNull(rawNulls, i)) {
          endOffset += rawValues[i].size();
        }
        offsets.push_back(endOffset);
      }
    }
    lastPartitionOffset = partitionOffset;
  }
}

void accumulateVariableWidthOffsetsForConstantVector(
    const PartitionedVectorPtr& partitionedVector,
    std::vector<std::vector<int32_t>>& offsetsPerPartition) {
  const auto* constantVector =
      partitionedVector->baseVector()->as<ConstantVector<StringView>>();
  VELOX_DCHECK_NOT_NULL(constantVector);

  const auto valueSize =
      constantVector->isNullAt(0) ? 0 : constantVector->valueAt(0).size();
  const auto* partitionOffsets = partitionedVector->rawPartitionOffsets();

  vector_size_t lastPartitionOffset = 0;
  for (uint32_t p = 0; p < offsetsPerPartition.size(); ++p) {
    const auto partitionOffset = partitionOffsets[p];
    auto& offsets = offsetsPerPartition[p];
    int32_t endOffset = offsets.empty() ? 0 : offsets.back();
    for (auto i = lastPartitionOffset; i < partitionOffset; ++i) {
      endOffset += valueSize;
      offsets.push_back(endOffset);
    }
    lastPartitionOffset = partitionOffset;
  }
}

void accumulateVariableWidthOffsetsForDictionaryVector(
    const PartitionedVectorPtr& partitionedVector,
    std::vector<std::vector<int32_t>>& offsetsPerPartition) {
  const auto* simpleVector =
      partitionedVector->baseVector()->as<SimpleVector<StringView>>();
  VELOX_DCHECK_NOT_NULL(simpleVector);

  const auto* partitionOffsets = partitionedVector->rawPartitionOffsets();

  vector_size_t lastPartitionOffset = 0;
  for (uint32_t p = 0; p < offsetsPerPartition.size(); ++p) {
    const auto partitionOffset = partitionOffsets[p];
    auto& offsets = offsetsPerPartition[p];
    int32_t endOffset = offsets.empty() ? 0 : offsets.back();
    for (auto i = lastPartitionOffset; i < partitionOffset; ++i) {
      if (!simpleVector->isNullAt(i)) {
        endOffset += simpleVector->valueAt(i).size();
      }
      offsets.push_back(endOffset);
    }
    lastPartitionOffset = partitionOffset;
  }
}

/// Accumulates per-partition end offsets for partitionedVector.
void accumulateVariableWidthOffsets(
    const PartitionedVectorPtr& partitionedVector,
    std::vector<std::vector<int32_t>>& offsetsPerPartition) {
  switch (partitionedVector->baseVector()->encoding()) {
    case VectorEncoding::Simple::FLAT:
      accumulateVariableWidthOffsetsForFlatVector(
          partitionedVector, offsetsPerPartition);
      break;
    case VectorEncoding::Simple::CONSTANT:
      accumulateVariableWidthOffsetsForConstantVector(
          partitionedVector, offsetsPerPartition);
      break;
    case VectorEncoding::Simple::DICTIONARY:
      accumulateVariableWidthOffsetsForDictionaryVector(
          partitionedVector, offsetsPerPartition);
      break;
    case VectorEncoding::Simple::BIASED:
    case VectorEncoding::Simple::SEQUENCE:
      VELOX_NYI(
          "Unsupported vector encoding for variable-width offset accumulation: {}",
          partitionedVector->baseVector()->encoding());
    default:
      VELOX_UNSUPPORTED(
          "Invalid vector encoding for variable-width offset accumulation: {}",
          partitionedVector->baseVector()->encoding());
  }
}

/// Returns the null counts if it can be derived without row-by-row checks,
/// otherwise returns std::nullopt.
std::optional<vector_size_t> countNulls(const BaseVector& vector) {
  if (!vector.mayHaveNulls()) {
    return 0;
  }

  if (const auto nullCount = vector.getNullCount()) {
    return *nullCount;
  }

  switch (vector.encoding()) {
    case VectorEncoding::Simple::FLAT:
    case VectorEncoding::Simple::ROW:
      return BaseVector::countNulls(vector.nulls(), vector.size());
    case VectorEncoding::Simple::CONSTANT:
      return vector.isNullAt(0) ? vector.size() : 0;
    case VectorEncoding::Simple::DICTIONARY: {
      vector_size_t nullCount = 0;
      for (auto i = 0; i < vector.size(); ++i) {
        nullCount += vector.isNullAt(i);
      }
      return nullCount;
    }
    default:
      return std::nullopt;
  }
}

/// Returns the maximum null-bitmap bytes for totalRows distributed across
/// numPartitionsWithNulls partitions. This occurs when one row is put in each
/// partition first, then one byte is added for every 8 remaining rows.
int64_t maxBitmapBytes(int64_t totalRows, int64_t numPartitionsWithNulls) {
  if (numPartitionsWithNulls == 0) {
    return 0;
  }
  VELOX_DCHECK_LE(numPartitionsWithNulls, totalRows);
  return numPartitionsWithNulls + (totalRows - numPartitionsWithNulls) / 8;
}

BufferPtr dictionaryNulls(
    const BaseVector& vector,
    memory::MemoryPool* pool) {
  if (!vector.mayHaveNulls()) {
    return nullptr;
  }

  auto nulls = AlignedBuffer::allocate<uint64_t>(
      bits::nwords(vector.size()), pool, bits::kNotNull64);
  auto* rawNulls = nulls->asMutable<uint64_t>();
  for (vector_size_t row = 0; row < vector.size(); ++row) {
    if (vector.isNullAt(row)) {
      bits::setNull(rawNulls, row);
    }
  }

  return nulls;
}

int64_t rowColumnBytes(
    int32_t numFields,
    int64_t numRows,
    int64_t numNulls,
    int64_t childBytes) {
  return 4 + static_cast<int64_t>(kRow.size()) + // encoding header
      4 + // numFields
      childBytes + 4 + // rowCount
      4 * (numRows + 1) + // offsets
      1 + // hasNulls
      (numNulls > 0 ? bits::nbytes(numRows) : 0); // null bitmap
}

} // namespace
/// Base class for column nodes in the serializer's per-partition accounting.
///
/// A node tracks exact row, null, and byte counts for one column while
/// appending partitioned vectors.
class ColumnBufferState {
 public:
  ColumnBufferState(TypePtr type, uint32_t numPartitions)
      : type_(std::move(type)),
        numPartitions_(numPartitions),
        rowsPerPartition_(numPartitions, 0),
        nullsPerPartition_(numPartitions, 0),
        bytesPerPartition_(numPartitions, 0) {}

  virtual ~ColumnBufferState() = default;

  static std::unique_ptr<ColumnBufferState> create(
      const TypePtr& type,
      uint32_t numPartitions);

  virtual void append(const PartitionedVectorPtr& partitionedVector) = 0;

  virtual void clear() {
    std::fill(rowsPerPartition_.begin(), rowsPerPartition_.end(), 0);
    std::fill(nullsPerPartition_.begin(), nullsPerPartition_.end(), 0);
    std::fill(bytesPerPartition_.begin(), bytesPerPartition_.end(), 0);
    numNonEmptyPartitions_ = 0;
    numPartitionsWithNulls_ = 0;
  }

  const std::vector<vector_size_t>& rowsPerPartition() const {
    return rowsPerPartition_;
  }

  const std::vector<int64_t>& bytesPerPartition() const {
    return bytesPerPartition_;
  }

  uint32_t numNonEmptyPartitions() const {
    return numNonEmptyPartitions_;
  }

  uint32_t numPartitionsWithNulls() const {
    return numPartitionsWithNulls_;
  }

  int64_t nullBitmapBytesBuffered() const {
    int64_t total = 0;
    for (auto p = 0; p < numPartitions_; ++p) {
      if (nullsPerPartition_[p] > 0) {
        total += bits::nbytes(rowsPerPartition_[p]);
      }
    }
    return total;
  }

 protected:
  const TypePtr type_;
  const uint32_t numPartitions_;
  std::vector<vector_size_t> rowsPerPartition_;
  std::vector<vector_size_t> nullsPerPartition_;
  std::vector<int64_t> bytesPerPartition_;

  // count of partitions with at least one buffered row
  uint32_t numNonEmptyPartitions_{0};

  // count of partitions that require a null bitmap
  uint32_t numPartitionsWithNulls_{0};
};

/// Buffer state for one fixed-width column.
class FixedWidthBufferState : public ColumnBufferState {
 public:
  FixedWidthBufferState(TypePtr type, uint32_t numPartitions)
      : ColumnBufferState(std::move(type), numPartitions) {}

  void append(const PartitionedVectorPtr& partitionedVector) override {
    for (auto p = 0; p < numPartitions_; ++p) {
      const auto numRows = partitionedVector->numRowsAt(p);
      if (numRows == 0) {
        continue;
      }

      const auto numNulls = partitionedVector->numNullsAt(p);
      auto& rows = rowsPerPartition_[p];
      auto& nulls = nullsPerPartition_[p];

      if (rows == 0) {
        ++numNonEmptyPartitions_;
      }
      if (nulls == 0 && numNulls > 0) {
        ++numPartitionsWithNulls_;
      }
      rows += numRows;
      nulls += numNulls;
      bytesPerPartition_[p] = simpleColumnBytes(type_, rows, nulls);
    }
  }
};

/// Buffer state for one VARCHAR or VARBINARY column.
class VariableWidthBufferState : public ColumnBufferState {
 public:
  VariableWidthBufferState(TypePtr type, uint32_t numPartitions)
      : ColumnBufferState(std::move(type), numPartitions),
        offsetsPerPartition_(numPartitions) {}

  void append(const PartitionedVectorPtr& partitionedVector) override {
    accumulateVariableWidthOffsets(partitionedVector, offsetsPerPartition_);

    for (auto p = 0; p < numPartitions_; ++p) {
      const auto numRows = partitionedVector->numRowsAt(p);
      if (numRows == 0) {
        continue;
      }

      const auto numNulls = partitionedVector->numNullsAt(p);
      auto& rows = rowsPerPartition_[p];
      auto& nulls = nullsPerPartition_[p];

      if (rows == 0) {
        ++numNonEmptyPartitions_;
      }
      if (nulls == 0 && numNulls > 0) {
        ++numPartitionsWithNulls_;
      }

      rows += numRows;
      nulls += numNulls;

      const auto dataBytes = offsetsPerPartition_[p].empty()
          ? 0
          : static_cast<int64_t>(offsetsPerPartition_[p].back());
      bytesPerPartition_[p] = variableWidthColumnBytes(rows, nulls, dataBytes);
    }
  }

  const std::vector<int32_t>& offsetsAt(uint32_t partition) const {
    VELOX_DCHECK_LT(partition, numPartitions_);
    return offsetsPerPartition_[partition];
  }

  void clear() override {
    ColumnBufferState::clear();
    for (auto& offsets : offsetsPerPartition_) {
      offsets.clear();
    }
  }

 private:
  // Per-partition cumulative offsets for buffered variable-width rows.
  // Contains one offset per row, null rows keep the previous offset.
  std::vector<std::vector<int32_t>> offsetsPerPartition_;
};

/// Buffer state for one nested ROW column.
///
/// Children are appended recursively, one per ROW field. The bytes for a
/// partition are this level's header, field count, offsets array, null flag
/// and bitmap, plus the sum of the children's bytes. The row count used in
/// the size estimate is the row count reported by the partitioned vector at
/// this level, which is an upper bound on the wire-format row count (the
/// wire writes only rows whose ancestors are not null), so the estimate
/// never under-allocates.
class RowVectorState : public ColumnBufferState {
 public:
  RowVectorState(
      TypePtr type,
      uint32_t numPartitions,
      std::vector<std::unique_ptr<ColumnBufferState>> children)
      : ColumnBufferState(std::move(type), numPartitions),
        children_(std::move(children)) {}

  void append(const PartitionedVectorPtr& partitionedVector) override {
    auto rowVector =
        std::dynamic_pointer_cast<PartitionedRowVector>(partitionedVector);
    VELOX_CHECK_NOT_NULL(rowVector);

    for (uint32_t col = 0; col < children_.size(); ++col) {
      children_[col]->append(rowVector->childAt(col));
    }

    for (uint32_t p = 0; p < numPartitions_; ++p) {
      const auto numRows = partitionedVector->numRowsAt(p);
      if (numRows == 0) {
        continue;
      }

      const auto numNulls = partitionedVector->numNullsAt(p);
      auto& rows = rowsPerPartition_[p];
      auto& nulls = nullsPerPartition_[p];

      if (rows == 0) {
        ++numNonEmptyPartitions_;
      }
      if (nulls == 0 && numNulls > 0) {
        ++numPartitionsWithNulls_;
      }
      rows += numRows;
      nulls += numNulls;

      int64_t childBytes = 0;
      for (const auto& child : children_) {
        childBytes += child->bytesPerPartition()[p];
      }
      bytesPerPartition_[p] = rowColumnBytes(
          static_cast<int32_t>(children_.size()), rows, nulls, childBytes);
    }
  }

  void clear() override {
    ColumnBufferState::clear();
    for (auto& child : children_) {
      child->clear();
    }
  }

  const std::vector<std::unique_ptr<ColumnBufferState>>& children() const {
    return children_;
  }

 private:
  std::vector<std::unique_ptr<ColumnBufferState>> children_;
};

std::unique_ptr<ColumnBufferState> ColumnBufferState::create(
    const TypePtr& type,
    uint32_t numPartitions) {
  switch (type->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::HUGEINT:
      return std::make_unique<FixedWidthBufferState>(type, numPartitions);
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      return std::make_unique<VariableWidthBufferState>(type, numPartitions);
    case TypeKind::ROW: {
      std::vector<std::unique_ptr<ColumnBufferState>> children;
      children.reserve(type->size());
      for (auto column = 0; column < type->size(); ++column) {
        children.push_back(
            ColumnBufferState::create(type->childAt(column), numPartitions));
      }
      return std::make_unique<RowVectorState>(
          type, numPartitions, std::move(children));
    }
    case TypeKind::TIMESTAMP:
    case TypeKind::ARRAY:
    case TypeKind::MAP:
      VELOX_NYI(
          "Unsupported type kind for createColumnBufferState: {}",
          type->kind());
    default:
      VELOX_UNSUPPORTED(
          "Unsupported type kind for createColumnBufferState: {}",
          type->kind());
  }
}

/// Top-level buffer state for one output page.
///
/// For each partition, tracks page-level headers and aggregates child column
/// sizes.
class BufferState {
 public:
  BufferState(
      uint32_t numPartitions,
      std::vector<std::unique_ptr<ColumnBufferState>> children)
      : numPartitions_(numPartitions),
        rowsPerPartition_(numPartitions, 0),
        bytesPerPartition_(numPartitions, 0),
        children_(std::move(children)) {}

  static std::unique_ptr<BufferState> create(
      const RowTypePtr& type,
      uint32_t numPartitions);

  void append(
      const PartitionedVectorPtr& partitionedVector,
      const std::vector<column_index_t>& outputToInputChannels) {
    auto rowVector =
        std::dynamic_pointer_cast<PartitionedRowVector>(partitionedVector);
    VELOX_CHECK_NOT_NULL(rowVector);

    rowsBuffered_ += partitionedVector->baseVector()->size();

    for (column_index_t column = 0; column < children_.size(); ++column) {
      const auto inputColumn = outputToInputChannels.empty()
          ? column
          : outputToInputChannels[column];
      children_[column]->append(rowVector->childAt(inputColumn));
    }

    for (auto p = 0; p < numPartitions_; ++p) {
      const auto numRows = partitionedVector->numRowsAt(p);
      if (numRows == 0) {
        continue;
      }
      if (rowsPerPartition_[p] == 0) {
        ++numNonEmptyPartitions_;
      }
      rowsPerPartition_[p] += numRows;

      int64_t partitionBytes = kHeaderSize + 4;
      for (const auto& child : children_) {
        partitionBytes += child->bytesPerPartition()[p];
      }
      bytesBuffered_ += partitionBytes - bytesPerPartition_[p];
      bytesPerPartition_[p] = partitionBytes;
    }
  }

  void clear() {
    std::fill(rowsPerPartition_.begin(), rowsPerPartition_.end(), 0);
    std::fill(bytesPerPartition_.begin(), bytesPerPartition_.end(), 0);
    numNonEmptyPartitions_ = 0;
    rowsBuffered_ = 0;
    bytesBuffered_ = 0;
    for (auto& child : children_) {
      child->clear();
    }
  }

  const std::vector<vector_size_t>& rowsPerPartition() const {
    return rowsPerPartition_;
  }

  const std::vector<int64_t>& bytesPerPartition() const {
    return bytesPerPartition_;
  }

  uint32_t numNonEmptyPartitions() const {
    return numNonEmptyPartitions_;
  }

  vector_size_t rowsBuffered() const {
    return rowsBuffered_;
  }

  int64_t bytesBuffered() const {
    return bytesBuffered_;
  }

  const std::vector<std::unique_ptr<ColumnBufferState>>& children() const {
    return children_;
  }

 private:
  const uint32_t numPartitions_;
  std::vector<vector_size_t> rowsPerPartition_;
  std::vector<int64_t> bytesPerPartition_;
  uint32_t numNonEmptyPartitions_{0};
  vector_size_t rowsBuffered_{0};
  int64_t bytesBuffered_{0};
  std::vector<std::unique_ptr<ColumnBufferState>> children_;
};

std::unique_ptr<BufferState> BufferState::create(
    const RowTypePtr& type,
    uint32_t numPartitions) {
  std::vector<std::unique_ptr<ColumnBufferState>> children;
  children.reserve(type->size());
  for (auto column = 0; column < type->size(); ++column) {
    children.push_back(
        ColumnBufferState::create(type->childAt(column), numPartitions));
  }
  return std::make_unique<BufferState>(numPartitions, std::move(children));
}

PrestoIterativePartitioningSerializer::PrestoIterativePartitioningSerializer(
    RowTypePtr outputType,
    uint32_t numPartitions,
    const SerdeOpts& opts,
    memory::MemoryPool* pool,
    std::vector<column_index_t> outputToInputChannels,
    std::function<std::unique_ptr<OutputStreamListener>()> listenerFactory)
    : outputType_(std::move(outputType)),
      outputToInputChannels_(std::move(outputToInputChannels)),
      numPartitions_(numPartitions),
      opts_(opts),
      pool_(pool),
      listenerFactory_(std::move(listenerFactory)),
      numColumns_(outputType_->size()),
      bufferState_(BufferState::create(outputType_, numPartitions_)) {
  VELOX_CHECK_GT(numPartitions_, 0);
  VELOX_CHECK_NOT_NULL(pool_);
  VELOX_CHECK(
      outputToInputChannels_.empty() ||
          outputToInputChannels_.size() == outputType_->size(),
      "outputToInputChannels size must match output column count");
}

PrestoIterativePartitioningSerializer::
    ~PrestoIterativePartitioningSerializer() = default;

int64_t PrestoIterativePartitioningSerializer::bytesBuffered() const {
  return bufferState_->bytesBuffered();
}

vector_size_t PrestoIterativePartitioningSerializer::rowsBuffered() const {
  return bufferState_->rowsBuffered();
}

void PrestoIterativePartitioningSerializer::clear() {
  partitionedRowVectors_.clear();
  bufferState_->clear();
}

void PrestoIterativePartitioningSerializer::validateOutputInputMapping(
    const RowVectorPtr& input) const {
  const auto numInputColumns = input->childrenSize();
  for (column_index_t outputColumn = 0; outputColumn < numColumns_;
       ++outputColumn) {
    const auto inputColumn = outputToInputChannel(outputColumn);
    VELOX_CHECK_LT(
        inputColumn,
        numInputColumns,
        "Output column {} maps to invalid input column {}",
        outputColumn,
        inputColumn);

    const auto& child = input->childAt(inputColumn);
    VELOX_CHECK_NOT_NULL(
        child,
        "Output column {} maps to null input column {}",
        outputColumn,
        inputColumn);

    const auto type = outputType_->childAt(outputColumn);
    VELOX_CHECK(
        child->type()->equivalent(*type),
        "Output column {} expects {}, got {} from input column {}",
        outputColumn,
        type->toString(),
        child->type()->toString(),
        inputColumn);
  }
}

int64_t PrestoIterativePartitioningSerializer::estimateBytesAfterAppend(
    const RowVectorPtr& input) const {
  VELOX_CHECK_NOT_NULL(input);
  validateOutputInputMapping(input);

  if (input->size() == 0) {
    return bytesBuffered();
  }

  const auto numRows = input->size();

  // Worst case: each input row lands in a distinct empty partition, capped by
  // the number of empty partitions.
  const auto numNewPartitions = std::min<uint32_t>(
      numRows, numPartitions_ - bufferState_->numNonEmptyPartitions());
  // One page header per newly non-empty partition.
  auto estimatedBytes =
      bufferState_->bytesBuffered() + numNewPartitions * (kHeaderSize + 4);

  const auto estimateNullBitmapGrowth =
      [&](const ColumnBufferState* columnState,
          const std::optional<vector_size_t>& inputNulls) -> int64_t {
    const auto partitionsWithNulls = std::min<uint32_t>(
        bufferState_->numNonEmptyPartitions() + numNewPartitions,
        columnState->numPartitionsWithNulls() + inputNulls.value_or(numRows));
    const auto nullBitmapBytes = maxBitmapBytes(
        bufferState_->rowsBuffered() + numRows, partitionsWithNulls);
    auto nullBitmapBytesBuffered = columnState->nullBitmapBytesBuffered();
    VELOX_DCHECK_GE(nullBitmapBytes, nullBitmapBytesBuffered);
    return nullBitmapBytes - nullBitmapBytesBuffered;
  };

  const auto estimateColumnGrowth =
      [&](const auto& self,
          const ColumnBufferState* columnState,
          const TypePtr& columnType,
          const VectorPtr& inputVector) -> int64_t {
    if (columnType->isUnknown()) {
      VELOX_UNSUPPORTED(
          "Unsupported type kind for "
          "PrestoIterativePartitioningSerializer::estimateBytesAfterAppend: {}",
          columnType->kind());
    }

    if (columnType->isFixedWidth()) {
      const auto inputNulls = countNulls(*inputVector);
      return numNewPartitions * simpleColumnBytes(columnType, 0, 0) +
          estimateNullBitmapGrowth(columnState, inputNulls) +
          static_cast<int64_t>(inputVector->size() - inputNulls.value_or(0)) *
          fixedTypeWidth(columnType->kind());
    }

    switch (columnType->kind()) {
      case TypeKind::VARCHAR:
      case TypeKind::VARBINARY: {
        const auto inputNulls = countNulls(*inputVector);
        return numNewPartitions * variableWidthColumnBytes(0, 0, 0) +
            estimateNullBitmapGrowth(columnState, inputNulls) +
            static_cast<int64_t>(inputVector->size()) * sizeof(int32_t) +
            variableWidthDataBytes(*inputVector);
      }
      case TypeKind::ROW: {
        VectorPtr normalizedVector = inputVector;
        if (normalizedVector->encoding() == VectorEncoding::Simple::DICTIONARY) {
          normalizedVector =
              RowVector::pushDictionaryToRowVectorLeaves(normalizedVector);
        }

        auto rowVector = normalizedVector->as<RowVector>();
        VELOX_CHECK_NOT_NULL(
            rowVector,
            "Expected ROW input vector for output type {}, got encoding {}",
            columnType->toString(),
            normalizedVector->encoding());

        const auto* rowState = dynamic_cast<const RowVectorState*>(columnState);
        VELOX_DCHECK_NOT_NULL(rowState);

        int64_t childGrowth = 0;
        const auto& rowSchema = columnType->asRow();
        for (column_index_t child = 0; child < rowSchema.size(); ++child) {
          childGrowth += self(
              self,
              rowState->children()[child].get(),
              rowSchema.childAt(child),
              rowVector->childAt(child));
        }

        const auto inputNulls = countNulls(*normalizedVector);
        return numNewPartitions *
                rowColumnBytes(static_cast<int32_t>(rowSchema.size()), 0, 0, 0) +
            estimateNullBitmapGrowth(columnState, inputNulls) +
            static_cast<int64_t>(normalizedVector->size()) * sizeof(int32_t) +
            childGrowth;
      }
      case TypeKind::ARRAY:
      case TypeKind::MAP:
        VELOX_NYI(
            "Unsupported type kind for "
            "PrestoIterativePartitioningSerializer::estimateBytesAfterAppend: {}",
            columnType->kind());
      default:
        VELOX_UNSUPPORTED(
            "Unsupported type kind for "
            "PrestoIterativePartitioningSerializer::estimateBytesAfterAppend: {}",
            columnType->kind());
    }
  };

  // Cache per input column. If multiple output columns map to the same input
  // column, reuse the already computed incremental bytes.
  std::vector<std::optional<int64_t>> estimatedIncrementalBytes(
      input->childrenSize());
  for (column_index_t column = 0; column < numColumns_; ++column) {
    const auto inputColumn = outputToInputChannel(column);
    if (estimatedIncrementalBytes[inputColumn].has_value()) {
      estimatedBytes += *estimatedIncrementalBytes[inputColumn];
      continue;
    }

    const auto& columnType = outputType_->childAt(column);
    const auto* columnState = bufferState_->children()[column].get();
    estimatedIncrementalBytes[inputColumn] = estimateColumnGrowth(
        estimateColumnGrowth,
        columnState,
        columnType,
        input->childAt(inputColumn));
    estimatedBytes += *estimatedIncrementalBytes[inputColumn];
  }
  return estimatedBytes;
}

void PrestoIterativePartitioningSerializer::append(
    const RowVectorPtr& input,
    const std::vector<uint32_t>& partitions) {
  VELOX_CHECK_NOT_NULL(input);
  VELOX_CHECK_EQ(
      input->size(),
      partitions.size(),
      "partitions.size() must equal input->size()");

  validateOutputInputMapping(input);

  if (input->size() == 0) {
    return;
  }

  PartitionBuildContext ctx;
  auto partitionedRowVector = PartitionedVector::create(
      std::static_pointer_cast<BaseVector>(input),
      partitions,
      numPartitions_,
      ctx,
      pool_);

  bufferState_->append(partitionedRowVector, outputToInputChannels_);
  partitionedRowVectors_.push_back(std::move(partitionedRowVector));
}

// ---------------------------------------------------------------------------
// Top-level flush
// ---------------------------------------------------------------------------

std::map<uint32_t, std::pair<std::unique_ptr<folly::IOBuf>, vector_size_t>>
PrestoIterativePartitioningSerializer::flush() {
  auto pages =
      (opts_.compressionKind == common::CompressionKind::CompressionKind_NONE)
      ? flushUncompressed()
      : flushCompressed();

  clear();

  return pages;
}

std::map<uint32_t, std::pair<std::unique_ptr<folly::IOBuf>, vector_size_t>>
PrestoIterativePartitioningSerializer::flushUncompressed() {
  if (partitionedRowVectors_.empty()) {
    return {};
  }

  // 1. Determine non-empty partitions.
  std::vector<uint32_t> nonEmptyPartitions;
  for (uint32_t p = 0; p < numPartitions_; ++p) {
    if (bufferState_->rowsPerPartition()[p] > 0) {
      nonEmptyPartitions.push_back(p);
    }
  }
  const auto& rowSchema = outputType_->asRow();

  // 2. Create per-partition listeners first so the codec mask can be derived
  // from whether the factory actually produced a listener. The factory may
  // return nullptr (e.g. when OutputBufferManager has no listener factory
  // set), in which case checksumming is skipped and the checksum bit must not
  // be set in the codec byte.
  std::vector<std::unique_ptr<OutputStreamListener>> listeners(numPartitions_);
  for (uint32_t p : nonEmptyPartitions) {
    if (listenerFactory_) {
      listeners[p] = listenerFactory_();
    }
  }
  const bool checksumEnabled = !nonEmptyPartitions.empty() &&
      listeners[nonEmptyPartitions[0]] != nullptr;
  const char codecMask = getCodecMarker(checksumEnabled);

  // 3. Create output streams sized to the exact bytes each partition will need,
  // so that the entire payload fits. This avoids multiple resizing and copying.
  std::vector<std::unique_ptr<IOBufOutputStream>> outputStreams(numPartitions_);
  std::vector<IOBufOutputStream*> rawOutputStreams(numPartitions_);
  std::vector<std::streampos> beginStreamPositions(numPartitions_);

  for (uint32_t p : nonEmptyPartitions) {
    outputStreams[p] = std::make_unique<IOBufOutputStream>(
        *pool_, listeners[p].get(), bufferState_->bytesPerPartition()[p]);
    rawOutputStreams[p] = outputStreams[p].get();
    beginStreamPositions[p] = outputStreams[p]->tellp();

    flushStart(*outputStreams[p], p, codecMask);
  }

  // 4. Flush column data.
  SerializerContext context;
  context.rowCounts = bufferState_->rowsPerPartition();
  // Top level parentNulls are null
  context.parentNulls.resize(partitionedRowVectors_.size());
  context.hasParentNulls = false;
  context.parentNullCounts.resize(partitionedRowVectors_.size());
  flushRowChildren(
      partitionedRowVectors_,
      rowSchema,
      nonEmptyPartitions,
      rawOutputStreams,
      context);

  // 5. Finalize the page by seeking back to fill in sizes and CRC, and get the
  // IOBuf and numOfRows from each stream.
  std::map<uint32_t, std::pair<std::unique_ptr<folly::IOBuf>, vector_size_t>>
      result;
  for (uint32_t p : nonEmptyPartitions) {
    flushFinish(
        *outputStreams[p],
        p,
        beginStreamPositions[p],
        codecMask,
        listeners[p].get());
    result[p] = std::make_pair(
        outputStreams[p]->getIOBuf(), bufferState_->rowsPerPartition()[p]);
  }

  return result;
}

std::map<uint32_t, std::pair<std::unique_ptr<folly::IOBuf>, vector_size_t>>
PrestoIterativePartitioningSerializer::flushCompressed() {
  VELOX_NYI();
}

// ---------------------------------------------------------------------------
// Second level functions: start, columns and finish
// ---------------------------------------------------------------------------

void PrestoIterativePartitioningSerializer::flushStart(
    IOBufOutputStream& out,
    uint32_t partition,
    char codecMask) const {
  auto* listener = dynamic_cast<PrestoOutputStreamListener*>(out.listener());
  if (listener) {
    listener->pause();
  }

  // Write 21-byte Presto page header; sizes and CRC are filled in later.
  const int32_t numRows =
      static_cast<int32_t>(bufferState_->rowsPerPartition()[partition]);
  char header[kHeaderSize] = {};
  std::memcpy(&header[0], &numRows, 4);
  std::memcpy(&header[4], &codecMask, 1);
  out.write(header, kHeaderSize);

  if (listener) {
    listener->resume();
  }

  // Number of columns is included in the CRC.
  const int32_t numCols = static_cast<int32_t>(numColumns_);
  out.write(reinterpret_cast<const char*>(&numCols), 4);
}

void PrestoIterativePartitioningSerializer::flushRowChildren(
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    const RowType& rowSchema,
    const std::vector<uint32_t>& nonEmptyPartitions,
    const std::vector<IOBufOutputStream*>& outputStreams,
    const SerializerContext& context) const {
  for (uint32_t col = 0; col < rowSchema.size(); ++col) {
    std::vector<PartitionedVectorPtr> column;
    column.reserve(partitionedVectors.size());
    for (const auto& partitionedVector : partitionedVectors) {
      const auto& partitionedRowVector =
          std::dynamic_pointer_cast<PartitionedRowVector>(partitionedVector);
      VELOX_DCHECK_NOT_NULL(partitionedRowVector.get());
      column.push_back(
          partitionedRowVector->childAt(outputToInputChannel(col)));
    }

    const auto& columnState = *bufferState_->children()[col];
    flushColumn(
        columnState,
        column,
        rowSchema.childAt(col),
        nonEmptyPartitions,
        outputStreams,
        context);
  }
}

void PrestoIterativePartitioningSerializer::flushFinish(
    IOBufOutputStream& out,
    uint32_t partition,
    std::streampos beginOffset,
    char codecMask,
    OutputStreamListener* listener) const {
  auto* prestoListener = dynamic_cast<PrestoOutputStreamListener*>(listener);
  if (prestoListener) {
    prestoListener->pause();
  }

  const std::streampos totalSize =
      static_cast<int32_t>(out.tellp() - beginOffset);
  const std::streampos uncompressedSize = totalSize - kHeaderSize;
  int64_t crc = 0;
  if (prestoListener) {
    crc = computeChecksum(
        *prestoListener,
        static_cast<int8_t>(codecMask),
        static_cast<int32_t>(bufferState_->rowsPerPartition()[partition]),
        uncompressedSize);
  }

  out.seekp(beginOffset + kUncompressedSizeOffset);
  writeInt32(&out, uncompressedSize);
  writeInt32(&out, uncompressedSize); // TODO: compressedSize
  writeInt64(&out, crc);
  out.seekp(beginOffset + totalSize);
}

// ---------------------------------------------------------------------------
// Column-level dispatch
// ---------------------------------------------------------------------------

void PrestoIterativePartitioningSerializer::flushColumn(
    const ColumnBufferState& columnState,
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    const TypePtr& colType,
    const std::vector<uint32_t>& nonEmptyPartitions,
    const std::vector<IOBufOutputStream*>& outputStreams,
    const SerializerContext& context) const {
  VELOX_CHECK_GT(partitionedVectors.size(), 0);

  auto typeKind = partitionedVectors[0]->baseVector()->typeKind();
  switch (typeKind) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::HUGEINT:
      flushSimpleColumn(
          partitionedVectors,
          colType,
          nonEmptyPartitions,
          outputStreams,
          context);
      break;

    case TypeKind::ROW:
      flushRowColumn(
          columnState,
          partitionedVectors,
          colType,
          nonEmptyPartitions,
          outputStreams,
          context);
      break;
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      flushVariableWidthColumn(
          columnState,
          partitionedVectors,
          colType,
          nonEmptyPartitions,
          outputStreams,
          context);
      break;

    case TypeKind::TIMESTAMP:
    case TypeKind::ARRAY:
    case TypeKind::MAP:
      VELOX_NYI(
          "Unsupported vector type kind for PrestoIterativePartitioningSerializer: {}",
          typeKind);

    default:
      VELOX_UNSUPPORTED(
          "Invalid vector type kind for PrestoIterativePartitioningSerializer: {}",
          typeKind);
  }
}

void PrestoIterativePartitioningSerializer::flushSimpleColumn(
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    const TypePtr& colType,
    const std::vector<uint32_t>& nonEmptyPartitions,
    const std::vector<IOBufOutputStream*>& outputStreams,
    const SerializerContext& context) const {
  flushHeader(typeToEncodingName(colType), nonEmptyPartitions, outputStreams);
  flushRowCounts(nonEmptyPartitions, outputStreams, context);
  flushNulls(partitionedVectors, nonEmptyPartitions, outputStreams, context);

  for (size_t i = 0; i < partitionedVectors.size(); i++) {
    const auto* parentNulls = context.hasParentNulls
        ? context.parentNulls[i]->as<uint64_t>()
        : nullptr;
    const std::vector<vector_size_t>* parentNullCountsPerPartition =
        context.hasParentNulls ? &context.parentNullCounts[i] : nullptr;
    flushSingleSimpleVector(
        partitionedVectors[i],
        outputStreams,
        parentNulls,
        parentNullCountsPerPartition);
  }
}

namespace {

// Appends the low 'count' bits of 'value' (count <= 64) to 'target' starting
// at bit 'bitOffset'. 'target' must be zero-initialized over the written range
// and own one extra addressable word past the last written bit.
inline void appendLowBits(
    uint64_t* target,
    uint64_t bitOffset,
    uint64_t value,
    uint32_t count) {
  const uint64_t word = bitOffset >> 6;
  const uint32_t shift = static_cast<uint32_t>(bitOffset & 63);
  target[word] |= value << shift;
  if (shift + count > 64) {
    target[word + 1] |= value >> (64 - shift);
  }
}

// Gathers the bits of 'source' at the positions in [begin, end) where 'mask'
// is set (every position when 'mask' is nullptr) and appends them, preserving
// order, to 'target' starting at bit 'targetBitOffset'. Returns the number of
// bits appended. Processes one 64-bit word at a time using bits::extractBits
// (parallel bit extract), so there is no per-row branching. 'target' must be
// zeroed over the written range with one extra addressable word past the last
// written bit.
int32_t compactBits(
    const uint64_t* source,
    const uint64_t* mask,
    int32_t begin,
    int32_t end,
    uint64_t* target,
    uint64_t targetBitOffset) {
  uint64_t outBit = targetBitOffset;
  bits::forEachWord(begin, end, [&](int32_t index, uint64_t wordMask) {
    const uint64_t selected = (mask ? mask[index] : ~0ULL) & wordMask;
    const uint64_t packed =
        bits::extractBits<uint64_t>(source[index], selected);
    const uint32_t count = __builtin_popcountll(selected);
    appendLowBits(target, outBit, packed, count);
    outBit += count;
  });
  return static_cast<int32_t>(outBit - targetBitOffset);
}

} // namespace

void PrestoIterativePartitioningSerializer::flushRowColumn(
    const ColumnBufferState& columnState,
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    const TypePtr& colType,
    const std::vector<uint32_t>& nonEmptyPartitions,
    const std::vector<IOBufOutputStream*>& outputStreams,
    const SerializerContext& context) const {
  const auto& rowSchema = colType->asRow();
  const int32_t numFields = static_cast<int32_t>(rowSchema.size());
  const size_t numVectors = partitionedVectors.size();
  const auto* rowState = dynamic_cast<const RowVectorState*>(&columnState);
  VELOX_DCHECK_NOT_NULL(rowState);

  // Number of parent-live rows that are null at this ROW level, per partition.
  std::vector<vector_size_t> nullCounts(numPartitions_, 0);

  SerializerContext childContext;
  childContext.hasParentNulls = true;
  childContext.rowCounts.assign(numPartitions_, 0);
  childContext.parentNulls.resize(numVectors);
  childContext.parentNullCounts.assign(
      numVectors, std::vector<vector_size_t>(numPartitions_, 0));

  // Step 1 + 2. For every batch, AND the incoming parentNulls into this
  // level's own nulls in place so the result marks the rows that are live for
  // the children (parent-live and not null here), then count live and null
  // rows per partition with bits::countBits. No new per-batch buffers are
  // allocated: the AND result is held in this vector's own nulls buffer, or
  // the parent's buffer is shared when there are no own nulls.
  for (size_t vectorIndex = 0; vectorIndex < numVectors; ++vectorIndex) {
    const auto& partitionedVector = partitionedVectors[vectorIndex];
    auto baseVector = partitionedVector->baseVector();
    const vector_size_t numRows = baseVector->size();
    const auto* partitionOffsets = partitionedVector->rawPartitionOffsets();
    const auto* parentNulls = context.hasParentNulls
        ? context.parentNulls[vectorIndex]->as<uint64_t>()
        : nullptr;
    const bool hasOwnNulls = baseVector->rawNulls() != nullptr;

    BufferPtr childLive;
    const uint64_t* rawChildLive{nullptr};
    if (hasOwnNulls) {
      auto* mutableNulls = baseVector->mutableRawNulls();
      if (parentNulls != nullptr) {
        bits::andBits(mutableNulls, parentNulls, 0, numRows);
      }
      childLive = baseVector->nulls();
      rawChildLive = mutableNulls;
    } else if (parentNulls != nullptr) {
      // No own nulls: live rows are exactly the parent-live rows. Share the
      // parent's buffer instead of allocating a copy.
      childLive = context.parentNulls[vectorIndex];
      rawChildLive = parentNulls;
    } else {
      // No nulls anywhere up to and including this level: all rows are live.
      baseVector->mutableRawNulls();
      childLive = baseVector->nulls();
      rawChildLive = childLive->as<uint64_t>();
    }
    childContext.parentNulls[vectorIndex] = childLive;

    vector_size_t begin = 0;
    for (uint32_t p = 0; p < numPartitions_; ++p) {
      const vector_size_t end = partitionOffsets[p];
      if (outputStreams[p] != nullptr && end > begin) {
        const vector_size_t parentLive = parentNulls != nullptr
            ? bits::countBits(parentNulls, begin, end)
            : end - begin;
        const vector_size_t live = bits::countBits(rawChildLive, begin, end);
        childContext.parentNullCounts[vectorIndex][p] = live;
        childContext.rowCounts[p] += live;
        nullCounts[p] += parentLive - live;
      }
      begin = end;
    }
  }

  // Header: "ROW" encoding name + numFields.
  flushHeader(kRow, nonEmptyPartitions, outputStreams);
  for (uint32_t p : nonEmptyPartitions) {
    writeInt32(outputStreams[p], numFields);
  }

  // Recurse into each child column with the propagated parent-null context.
  for (uint32_t col = 0; col < static_cast<uint32_t>(numFields); ++col) {
    std::vector<PartitionedVectorPtr> childVectors;
    childVectors.reserve(numVectors);
    for (const auto& pv : partitionedVectors) {
      childVectors.push_back(
          std::dynamic_pointer_cast<PartitionedRowVector>(pv)->childAt(col));
    }
    const auto& childState = *rowState->children()[col];
    flushColumn(
        childState,
        childVectors,
        rowSchema.childAt(col),
        nonEmptyPartitions,
        outputStreams,
        childContext);
  }

  // Step 3. Footer. The number of rows at this level equals the number of
  // parent-live rows, which the parent recorded in context.rowCounts. Only
  // partitions that have nulls at this level need a compacted bitmap; the
  // rest use sequential offsets and no null section.
  std::vector<BufferPtr> bitmaps(numPartitions_);
  std::vector<uint64_t*> rawBitmaps(numPartitions_, nullptr);
  std::vector<uint64_t> bitmapBitOffsets(numPartitions_, 0);
  for (uint32_t p : nonEmptyPartitions) {
    if (nullCounts[p] > 0) {
      const auto numWords = bits::nwords(context.rowCounts[p]) + 1;
      bitmaps[p] = AlignedBuffer::allocate<uint64_t>(numWords, pool_, 0);
      rawBitmaps[p] = bitmaps[p]->asMutable<uint64_t>();
    }
  }

  // Compact this level's live bits into each partition's bitmap, in batch
  // order, keeping only the positions where the parent is live.
  for (size_t vectorIndex = 0; vectorIndex < numVectors; ++vectorIndex) {
    const auto& partitionedVector = partitionedVectors[vectorIndex];
    const auto* partitionOffsets = partitionedVector->rawPartitionOffsets();
    const auto* parentNulls = context.hasParentNulls
        ? context.parentNulls[vectorIndex]->as<uint64_t>()
        : nullptr;
    const auto* rawChildLive =
        childContext.parentNulls[vectorIndex]->as<uint64_t>();

    vector_size_t begin = 0;
    for (uint32_t p = 0; p < numPartitions_; ++p) {
      const vector_size_t end = partitionOffsets[p];
      if (rawBitmaps[p] != nullptr && end > begin) {
        bitmapBitOffsets[p] += compactBits(
            rawChildLive,
            parentNulls,
            begin,
            end,
            rawBitmaps[p],
            bitmapBitOffsets[p]);
      }
      begin = end;
    }
  }

  for (uint32_t p : nonEmptyPartitions) {
    const int32_t numRows = static_cast<int32_t>(context.rowCounts[p]);
    writeInt32(outputStreams[p], numRows);

    if (nullCounts[p] == 0) {
      // No nulls at this level: offsets are sequential, no null section.
      for (int32_t i = 0; i <= numRows; ++i) {
        writeInt32(outputStreams[p], i);
      }
      const char hasNulls = 0;
      outputStreams[p]->write(&hasNulls, 1);
      continue;
    }

    // The offsets are the running count of non-null rows: a prefix sum over
    // the compacted live bitmap, where a set bit means not null here.
    const uint64_t* live = rawBitmaps[p];
    int32_t offset = 0;
    writeInt32(outputStreams[p], 0);
    for (int32_t i = 0; i < numRows; ++i) {
      offset += bits::isBitSet(live, i) ? 1 : 0;
      writeInt32(outputStreams[p], offset);
    }

    const char hasNulls = 1;
    outputStreams[p]->write(&hasNulls, 1);

    // Convert Velox format (LSB-first, 1 == not null) to Presto wire format
    // (MSB-first, 1 == null). Pad bits past numRows stay not-null.
    const int32_t numBytes = bits::nbytes(numRows);
    bits::fillBits(rawBitmaps[p], numRows, numBytes * 8, bits::kNotNull);
    auto* bytes = reinterpret_cast<uint8_t*>(rawBitmaps[p]);
    for (int32_t i = 0; i < numBytes; ++i) {
      bytes[i] = ~bytes[i];
      bits::reverseBits(&bytes[i], 1);
    }
    outputStreams[p]->write(reinterpret_cast<const char*>(bytes), numBytes);
  }
}

void PrestoIterativePartitioningSerializer::flushVariableWidthColumn(
    const ColumnBufferState& columnState,
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    const TypePtr& colType,
    const std::vector<uint32_t>& nonEmptyPartitions,
    const std::vector<IOBufOutputStream*>& outputStreams,
    const SerializerContext& context) const {
  flushHeader(typeToEncodingName(colType), nonEmptyPartitions, outputStreams);
  flushRowCounts(nonEmptyPartitions, outputStreams, context);
  flushOffsets(columnState, nonEmptyPartitions, outputStreams);
  flushNulls(partitionedVectors, nonEmptyPartitions, outputStreams, context);

  const auto* variableWidthState =
      dynamic_cast<const VariableWidthBufferState*>(&columnState);
  VELOX_DCHECK_NOT_NULL(variableWidthState);

  for (auto p : nonEmptyPartitions) {
    const auto& offsets = variableWidthState->offsetsAt(p);
    writeInt32(outputStreams[p], offsets.empty() ? 0 : offsets.back());
  }

  for (const auto& partitionedVector : partitionedVectors) {
    flushSingleVariableWidthVector(partitionedVector, outputStreams);
  }
}

template <TypeKind kind>
void PrestoIterativePartitioningSerializer::flushSingleFlatVector(
    const PartitionedVectorPtr& partitionedVector,
    const std::vector<IOBufOutputStream*>& outputStreams,
    const uint64_t* parentNulls) const {
  using T = typename TypeTraits<kind>::NativeType;
  auto* flatVector = partitionedVector->as<PartitionedFlatVector<T>>();
  VELOX_DCHECK_NOT_NULL(flatVector);

  const auto* rawValues =
      flatVector->baseVector()->template as<FlatVector<T>>()->rawValues();
  // rawNulls() may be nullptr when the column has no nulls. Do not use
  // mutableRawNulls() here: it would materialize an all-not-null buffer and
  // mask the "no nulls" fast path.
  const auto* rawNulls = flatVector->baseVector()->rawNulls();
  const auto* partitionOffsets = flatVector->rawPartitionOffsets();

  flushFlatValues<T>(
      rawValues, rawNulls, parentNulls, partitionOffsets, outputStreams);
}

// BOOLEAN columns use kByteArray encoding: FlatVector<bool> stores bits
// packed, so rawValues() is unsupported. Each non-null value is written as
// one byte (0x00 or 0x01).
template <>
void PrestoIterativePartitioningSerializer::flushSingleFlatVector<
    TypeKind::BOOLEAN>(
    const PartitionedVectorPtr& partitionedVector,
    const std::vector<IOBufOutputStream*>& outputStreams,
    const uint64_t* parentNulls) const {
  auto* flatVector = partitionedVector->as<PartitionedFlatVector<bool>>();
  VELOX_DCHECK_NOT_NULL(flatVector);

  const auto* rawBoolValues =
      flatVector->baseVector()->as<FlatVector<bool>>()->rawValues<uint64_t>();
  const auto* rawNulls = flatVector->baseVector()->rawNulls();
  const auto* partitionOffsets = flatVector->rawPartitionOffsets();

  // TODO: Improve performance
  vector_size_t lastOffset = 0;
  for (uint32_t p = 0; p < numPartitions_; ++p) {
    const auto offset = partitionOffsets[p];
    const auto numValues = offset - lastOffset;
    if (outputStreams[p] != nullptr && numValues > 0) {
      if (!parentNulls && !rawNulls) {
        for (vector_size_t i = lastOffset; i < offset; ++i) {
          const int8_t val = bits::isBitSet(rawBoolValues, i) ? 1 : 0;
          outputStreams[p]->write(reinterpret_cast<const char*>(&val), 1);
        }
      } else {
        for (vector_size_t i = lastOffset; i < offset; ++i) {
          const bool parentLive =
              !parentNulls || bits::isBitSet(parentNulls, i);
          const bool rowIsNull = rawNulls && bits::isBitNull(rawNulls, i);
          if (parentLive && !rowIsNull) {
            const int8_t val = bits::isBitSet(rawBoolValues, i) ? 1 : 0;
            outputStreams[p]->write(reinterpret_cast<const char*>(&val), 1);
          }
        }
      }
    }
    lastOffset = offset;
  }
}

template <TypeKind kind>
void PrestoIterativePartitioningSerializer::flushSingleConstantVector(
    const PartitionedVectorPtr& partitionedVector,
    const std::vector<IOBufOutputStream*>& outputStreams,
    const uint64_t* parentNulls,
    const std::vector<vector_size_t>* parentNullCountsPerPartition) const {
  if constexpr (
      kind == TypeKind::VARCHAR || kind == TypeKind::VARBINARY ||
      kind == TypeKind::TIMESTAMP) {
    VELOX_NYI(
        "flushSingleConstantVector does not support variable-length type: {}",
        kind);
  }

  using T = typename TypeTraits<kind>::NativeType;
  auto* constantVector =
      partitionedVector->baseVector()->template as<ConstantVector<T>>();
  VELOX_DCHECK_NOT_NULL(constantVector);

  if (constantVector->isNullAt(0)) {
    return;
  }

  const auto value = constantVector->valueAtFast(0);
  const auto* partitionOffsets = partitionedVector->rawPartitionOffsets();

  Scratch scratch;
  ScratchPtr<T> values(scratch);
  const auto numRowsPerChunk =
      std::max<vector_size_t>(1, kChunkBytes / sizeof(T));
  const char* chunkBytes = nullptr;

  vector_size_t lastOffset = 0;
  for (uint32_t p = 0; p < numPartitions_; ++p) {
    const auto offset = partitionOffsets[p];
    auto numRows = parentNullCountsPerPartition != nullptr
        ? (*parentNullCountsPerPartition)[p]
        : offset - lastOffset;
    if (numRows > 0) {
      VELOX_DCHECK_NOT_NULL(outputStreams[p]);

      if (chunkBytes == nullptr) {
        auto* ptr = values.get(numRowsPerChunk);
        std::fill_n(ptr, numRowsPerChunk, value);
        chunkBytes = reinterpret_cast<const char*>(ptr);
      }

      if (!parentNulls) {
        while (numRows > 0) {
          auto n = std::min<vector_size_t>(numRowsPerChunk, numRows);
          outputStreams[p]->write(chunkBytes, n * sizeof(T));
          numRows -= n;
        }
      } else {
        for (vector_size_t i = lastOffset; i < offset; ++i) {
          if (bits::isBitSet(parentNulls, i)) {
            outputStreams[p]->write(
                reinterpret_cast<const char*>(&value), sizeof(T));
          }
        }
      }
    }
    lastOffset = offset;
  }
}

template <TypeKind kind>
void flushSingleDictionaryVectorValues(
    const BaseVector& vector,
    const vector_size_t* partitionOffsets,
    const std::vector<IOBufOutputStream*>& outputStreams,
    const uint64_t* parentNulls,
    uint32_t numPartitions) {
  using T = typename TypeTraits<kind>::NativeType;
  auto* simpleVector = vector.as<SimpleVector<T>>();
  VELOX_DCHECK_NOT_NULL(simpleVector);

  vector_size_t lastOffset = 0;
  for (uint32_t p = 0; p < numPartitions; ++p) {
    const auto offset = partitionOffsets[p];
    if (outputStreams[p] != nullptr && offset > lastOffset) {
      for (vector_size_t i = lastOffset; i < offset; ++i) {
        if ((parentNulls == nullptr || bits::isBitSet(parentNulls, i)) &&
            !simpleVector->isNullAt(i)) {
          if constexpr (kind == TypeKind::BOOLEAN) {
            const int8_t value = simpleVector->valueAt(i) ? 1 : 0;
            outputStreams[p]->write(
                reinterpret_cast<const char*>(&value), sizeof(value));
          } else {
            const T value = simpleVector->valueAt(i);
            outputStreams[p]->write(
                reinterpret_cast<const char*>(&value), sizeof(value));
          }
        }
      }
    }
    lastOffset = offset;
  }
}

void PrestoIterativePartitioningSerializer::flushSingleSimpleVector(
    const PartitionedVectorPtr& partitionedVector,
    const std::vector<IOBufOutputStream*>& outputStreams,
    const uint64_t* parentNulls,
    const std::vector<vector_size_t>* parentNullCountsPerPartition) const {
  auto encoding = partitionedVector->baseVector()->encoding();
  auto typeKind = partitionedVector->baseVector()->typeKind();

  switch (encoding) {
    case VectorEncoding::Simple::FLAT:
      VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          flushSingleFlatVector,
          typeKind,
          partitionedVector,
          outputStreams,
          parentNulls);
      break;
    case VectorEncoding::Simple::CONSTANT:
      VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          flushSingleConstantVector,
          typeKind,
          partitionedVector,
          outputStreams,
          parentNulls,
          parentNullCountsPerPartition);
      break;
    case VectorEncoding::Simple::DICTIONARY:
      VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH(
          flushSingleDictionaryVectorValues,
          typeKind,
          *partitionedVector->baseVector(),
          partitionedVector->rawPartitionOffsets(),
          outputStreams,
          parentNulls,
          numPartitions_);
      break;
    case VectorEncoding::Simple::BIASED:
    case VectorEncoding::Simple::SEQUENCE:
      VELOX_NYI(
          "Unsupported vector encoding for PrestoIterativePartitioningSerializer: {}",
          encoding);
    default:
      VELOX_UNSUPPORTED(
          "Invalid vector encoding for PrestoIterativePartitioningSerializer:flushSingleSimpleVector: {}",
          encoding);
  }
}

void PrestoIterativePartitioningSerializer::flushSingleVariableWidthFlatVector(
    const PartitionedVectorPtr& partitionedVector,
    const std::vector<IOBufOutputStream*>& outputStreams) const {
  auto* flatVector = partitionedVector->as<PartitionedFlatVector<StringView>>();
  VELOX_DCHECK_NOT_NULL(flatVector);

  const auto* rawValues =
      flatVector->baseVector()->as<FlatVector<StringView>>()->rawValues();
  const auto* rawNulls = flatVector->baseVector()->rawNulls();
  const auto* partitionOffsets = flatVector->rawPartitionOffsets();

  vector_size_t lastOffset = 0;
  for (uint32_t p = 0; p < numPartitions_; ++p) {
    const auto offset = partitionOffsets[p];
    if (outputStreams[p] != nullptr) {
      if (!rawNulls) {
        for (auto i = lastOffset; i < offset; ++i) {
          outputStreams[p]->write(rawValues[i].data(), rawValues[i].size());
        }
      } else {
        for (auto i = lastOffset; i < offset; ++i) {
          if (!bits::isBitNull(rawNulls, i)) {
            outputStreams[p]->write(rawValues[i].data(), rawValues[i].size());
          }
        }
      }
    }
    lastOffset = offset;
  }
}

void PrestoIterativePartitioningSerializer::
    flushSingleVariableWidthConstantVector(
        const PartitionedVectorPtr& partitionedVector,
        const std::vector<IOBufOutputStream*>& outputStreams) const {
  const auto* constantVector =
      partitionedVector->baseVector()->as<ConstantVector<StringView>>();
  VELOX_DCHECK_NOT_NULL(constantVector);

  if (constantVector->isNullAt(0)) {
    return;
  }

  const auto value = constantVector->valueAt(0);
  const auto valueSize = value.size();
  if (valueSize == 0) {
    return;
  }

  const auto* partitionOffsets = partitionedVector->rawPartitionOffsets();

  const auto numRowsPerChunk =
      std::max<vector_size_t>(1, kChunkBytes / valueSize);
  const char* chunkBytes = value.data();
  Scratch scratch;
  ScratchPtr<char> chunk(scratch);
  if (numRowsPerChunk > 1) {
    auto* ptr = chunk.get(numRowsPerChunk * valueSize);
    for (vector_size_t i = 0; i < numRowsPerChunk; ++i) {
      simd::memcpy(ptr + i * valueSize, value.data(), valueSize);
    }
    chunkBytes = ptr;
  }

  vector_size_t lastOffset = 0;
  for (uint32_t p = 0; p < numPartitions_; ++p) {
    const auto offset = partitionOffsets[p];
    auto numRows = offset - lastOffset;
    if (numRows > 0) {
      VELOX_DCHECK_NOT_NULL(outputStreams[p]);
      while (numRows > 0) {
        const auto n = std::min<vector_size_t>(numRowsPerChunk, numRows);
        outputStreams[p]->write(chunkBytes, n * valueSize);
        numRows -= n;
      }
    }
    lastOffset = offset;
  }
}

void flushSingleVariableWidthDictionaryVector(
    const PartitionedVectorPtr& partitionedVector,
    const std::vector<IOBufOutputStream*>& outputStreams,
    uint32_t numPartitions) {
  const auto* simpleVector =
      partitionedVector->baseVector()->as<SimpleVector<StringView>>();
  VELOX_DCHECK_NOT_NULL(simpleVector);

  const auto* partitionOffsets = partitionedVector->rawPartitionOffsets();

  vector_size_t lastOffset = 0;
  for (uint32_t p = 0; p < numPartitions; ++p) {
    const auto offset = partitionOffsets[p];
    if (outputStreams[p] != nullptr && offset > lastOffset) {
      for (vector_size_t i = lastOffset; i < offset; ++i) {
        if (!simpleVector->isNullAt(i)) {
          const auto value = simpleVector->valueAt(i);
          outputStreams[p]->write(value.data(), value.size());
        }
      }
    }
    lastOffset = offset;
  }
}

void PrestoIterativePartitioningSerializer::flushSingleVariableWidthVector(
    const PartitionedVectorPtr& partitionedVector,
    const std::vector<IOBufOutputStream*>& outputStreams) const {
  const auto encoding = partitionedVector->baseVector()->encoding();

  switch (encoding) {
    case VectorEncoding::Simple::FLAT:
      flushSingleVariableWidthFlatVector(partitionedVector, outputStreams);
      break;
    case VectorEncoding::Simple::CONSTANT:
      flushSingleVariableWidthConstantVector(partitionedVector, outputStreams);
      break;
    case VectorEncoding::Simple::DICTIONARY:
      flushSingleVariableWidthDictionaryVector(
          partitionedVector, outputStreams, numPartitions_);
      break;
    case VectorEncoding::Simple::BIASED:
    case VectorEncoding::Simple::SEQUENCE:
      VELOX_NYI(
          "Unsupported vector encoding for PrestoIterativePartitioningSerializer: {}",
          encoding);
    default:
      VELOX_UNSUPPORTED(
          "Invalid vector encoding for PrestoIterativePartitioningSerializer: {}",
          encoding);
  }
}

// ---------------------------------------------------------------------------
// Column building blocks
// ---------------------------------------------------------------------------

void PrestoIterativePartitioningSerializer::flushHeader(
    std::string_view name,
    const std::vector<uint32_t>& nonEmptyPartitions,
    const std::vector<IOBufOutputStream*>& outputStreams) const {
  const int32_t nameLen = static_cast<int32_t>(name.size());
  for (uint32_t p : nonEmptyPartitions) {
    writeInt32(outputStreams[p], nameLen);
    outputStreams[p]->write(name.data(), nameLen);
  }
}

void PrestoIterativePartitioningSerializer::flushRowCounts(
    const std::vector<uint32_t>& nonEmptyPartitions,
    const std::vector<IOBufOutputStream*>& outputStreams,
    const SerializerContext& context) const {
  for (uint32_t p : nonEmptyPartitions) {
    writeInt32(outputStreams[p], static_cast<int32_t>(context.rowCounts[p]));
  }
}

void PrestoIterativePartitioningSerializer::flushNulls(
    const std::vector<PartitionedVectorPtr>& partitionedVectors,
    const std::vector<uint32_t>& nonEmptyPartitions,
    const std::vector<IOBufOutputStream*>& outputStreams,
    const SerializerContext& context) const {
  const size_t numVectors = partitionedVectors.size();

  // Per-partition null bitmap accumulated across all batches, in Velox format
  // (1 == not null). One extra word so the bit appender in compactBits can
  // always touch the word past the last bit.
  std::vector<vector_size_t> nullCounts(numPartitions_, 0);
  std::vector<BufferPtr> bitmaps(numPartitions_);
  std::vector<uint64_t*> rawBitmaps(numPartitions_, nullptr);
  std::vector<uint64_t> bitOffsets(numPartitions_, 0);
  for (uint32_t p : nonEmptyPartitions) {
    const auto numWords = bits::nwords(context.rowCounts[p]) + 1;
    bitmaps[p] = AlignedBuffer::allocate<uint64_t>(numWords, pool_, 0);
    rawBitmaps[p] = bitmaps[p]->asMutable<uint64_t>();
  }

  for (size_t vectorIndex = 0; vectorIndex < numVectors; ++vectorIndex) {
    const auto& pv = partitionedVectors[vectorIndex];
    const auto* partitionOffsets = pv->rawPartitionOffsets();
    const auto* parentNulls = context.hasParentNulls
        ? context.parentNulls[vectorIndex]->as<uint64_t>()
        : nullptr;
    const auto encoding = pv->baseVector()->encoding();

    // validBits == nullptr means every present row in this batch is not null;
    // allNull means every present row is null. Otherwise validBits is a
    // full-row-space bitmap where a set bit means not null.
    const uint64_t* validBits{nullptr};
    bool allNull{false};
    BufferPtr rawNulls{nullptr};
    switch (encoding) {
      case VectorEncoding::Simple::FLAT:
        validBits = pv->baseVector()->rawNulls();
        break;
      case VectorEncoding::Simple::CONSTANT:
        allNull = pv->baseVector()->isNullAt(0);
        break;
      case VectorEncoding::Simple::DICTIONARY: {
        rawNulls = dictionaryNulls(*pv->baseVector(), pool_);
        validBits = rawNulls ? rawNulls->as<uint64_t>() : nullptr;
        break;
      }
      case VectorEncoding::Simple::BIASED:
      case VectorEncoding::Simple::SEQUENCE:
        VELOX_NYI(
            "Unsupported vector encoding for PrestoIterativePartitioningSerializer: {}",
            encoding);
      default:
        VELOX_UNSUPPORTED(
            "Invalid vector encoding for PrestoIterativePartitioningSerializer: {}",
            encoding);
    }

    vector_size_t begin = 0;
    for (uint32_t p = 0; p < numPartitions_; ++p) {
      const vector_size_t end = partitionOffsets[p];
      if (outputStreams[p] != nullptr && end > begin) {
        const vector_size_t present = parentNulls != nullptr
            ? bits::countBits(parentNulls, begin, end)
            : end - begin;
        if (allNull) {
          // Leave the compacted bits at 0 (null) and advance the cursor.
          bitOffsets[p] += present;
          nullCounts[p] += present;
        } else if (validBits == nullptr) {
          // No nulls in this batch: mark all present rows not null.
          bits::fillBits(
              rawBitmaps[p],
              bitOffsets[p],
              bitOffsets[p] + present,
              bits::kNotNull);
          bitOffsets[p] += present;
        } else {
          compactBits(
              validBits, parentNulls, begin, end, rawBitmaps[p], bitOffsets[p]);
          const auto valid = bits::countBits(
              rawBitmaps[p], bitOffsets[p], bitOffsets[p] + present);
          nullCounts[p] += present - valid;
          bitOffsets[p] += present;
        }
      }
      begin = end;
    }
  }

  for (uint32_t p : nonEmptyPartitions) {
    const char hasNulls = nullCounts[p] > 0 ? 1 : 0;
    outputStreams[p]->write(&hasNulls, 1);
  }

  const bool hasAnyNulls = std::any_of(
      nonEmptyPartitions.begin(), nonEmptyPartitions.end(), [&](uint32_t p) {
        return nullCounts[p] > 0;
      });
  if (!hasAnyNulls) {
    return;
  }

  for (uint32_t p : nonEmptyPartitions) {
    if (nullCounts[p] == 0) {
      continue;
    }
    // Convert Velox format (LSB-first, 1 == not null) to Presto wire format
    // (MSB-first, 1 == null). Pad bits past the row count stay not-null.
    const int32_t numRows = static_cast<int32_t>(context.rowCounts[p]);
    const int32_t numBytes = bits::nbytes(numRows);
    bits::fillBits(rawBitmaps[p], numRows, numBytes * 8, bits::kNotNull);
    auto* bytes = reinterpret_cast<uint8_t*>(rawBitmaps[p]);
    for (int32_t i = 0; i < numBytes; ++i) {
      bytes[i] = ~bytes[i];
      bits::reverseBits(&bytes[i], 1);
    }
    outputStreams[p]->write(reinterpret_cast<const char*>(bytes), numBytes);
  }
}

template <typename T>
void PrestoIterativePartitioningSerializer::flushFlatValues(
    const T* partitionedValues,
    const uint64_t* rawNulls,
    const uint64_t* parentNulls,
    const vector_size_t* partitionOffsets,
    const std::vector<IOBufOutputStream*>& outputStreams) const {
  const auto typeWidth = sizeof(T);
  vector_size_t lastOffset = 0;
  for (uint32_t p = 0; p < numPartitions_; ++p) {
    const auto offset = partitionOffsets[p];
    const auto numValues = offset - lastOffset;
    if (numValues > 0) {
      VELOX_CHECK_NOT_NULL(outputStreams[p]);

      if (!parentNulls && !rawNulls) {
        outputStreams[p]->write(
            reinterpret_cast<const char*>(&partitionedValues[lastOffset]),
            numValues * typeWidth);
      } else {
        // Presto writes only the rows that are live (the parent is not null)
        // and not null themselves; null slots are omitted. parentNulls and
        // rawNulls are indexed in the full row space [0, size), so iterate the
        // partition's own range [lastOffset, offset).
        // TODO: Improve performance.
        for (vector_size_t i = lastOffset; i < offset; ++i) {
          const bool parentLive =
              !parentNulls || bits::isBitSet(parentNulls, i);
          const bool rowIsNull = rawNulls && bits::isBitNull(rawNulls, i);
          if (parentLive && !rowIsNull) {
            outputStreams[p]->write(
                reinterpret_cast<const char*>(&partitionedValues[i]),
                typeWidth);
          }
        }
      }
    }
    lastOffset = offset;
  }
}

void PrestoIterativePartitioningSerializer::flushSequentialOffsets(
    const std::vector<uint32_t>& nonEmptyPartitions,
    const std::vector<IOBufOutputStream*>& outputStreams) const {
  for (uint32_t p : nonEmptyPartitions) {
    const int32_t numRows =
        static_cast<int32_t>(bufferState_->rowsPerPartition()[p]);
    for (int32_t i = 0; i <= numRows; ++i) {
      writeInt32(outputStreams[p], i);
    }
  }
}

void PrestoIterativePartitioningSerializer::flushOffsets(
    const ColumnBufferState& columnState,
    const std::vector<uint32_t>& nonEmptyPartitions,
    const std::vector<IOBufOutputStream*>& outputStreams) const {
  const auto* variableWidthState =
      dynamic_cast<const VariableWidthBufferState*>(&columnState);
  VELOX_DCHECK_NOT_NULL(variableWidthState);

  for (auto p : nonEmptyPartitions) {
    const auto& offsets = variableWidthState->offsetsAt(p);
    VELOX_DCHECK_EQ(offsets.size(), bufferState_->rowsPerPartition()[p]);

    if (!offsets.empty()) {
      outputStreams[p]->write(
          reinterpret_cast<const char*>(offsets.data()),
          offsets.size() * sizeof(int32_t));
    }
  }
}

} // namespace facebook::velox::serializer::presto

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

#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/SelectivityVector.h"

namespace facebook::velox::exec {

/// Collects the rows whose value in any non-constant partition-key column is
/// null. Shared by PartitionedOutput and OptimizedPartitionedOutput to
/// implement replicateNullsAndAny. Reusable across batches; not thread-safe.
class NullKeyRowsCollector {
 public:
  /// Returns the rows of 'input' where any key in 'keyChannels' is null.
  /// Entries equal to kConstantChannel are skipped, as are key columns whose
  /// vectors cannot contain nulls. The returned reference remains valid until
  /// the next call.
  const SelectivityVector& collect(
      const RowVector& input,
      const std::vector<column_index_t>& keyChannels);

 private:
  // All rows of the current input; the decode target.
  SelectivityVector allRows_;
  // Rows with a null in any non-constant key column.
  SelectivityVector nullRows_;
  // One decoder per key channel, reused across batches.
  std::vector<DecodedVector> decodedVectors_;
};

} // namespace facebook::velox::exec

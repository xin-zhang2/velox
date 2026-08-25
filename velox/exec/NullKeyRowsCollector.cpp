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

#include "velox/exec/NullKeyRowsCollector.h"

namespace facebook::velox::exec {

const SelectivityVector& NullKeyRowsCollector::collect(
    const RowVector& input,
    const std::vector<column_index_t>& keyChannels) {
  const auto size = input.size();
  allRows_.resize(size);
  allRows_.setAll();

  nullRows_.resize(size);
  nullRows_.clearAll();

  decodedVectors_.resize(keyChannels.size());

  for (size_t keyChannelIndex = 0; keyChannelIndex < keyChannels.size();
       ++keyChannelIndex) {
    const auto keyChannel = keyChannels[keyChannelIndex];
    // Skip constant channel.
    if (keyChannel == kConstantChannel) {
      continue;
    }
    const auto& keyVector = input.childAt(keyChannel);
    if (keyVector->mayHaveNulls()) {
      DecodedVector& decodedVector = decodedVectors_[keyChannelIndex];
      decodedVector.decode(*keyVector, allRows_);
      if (auto* rawNulls = decodedVector.nulls(&allRows_)) {
        bits::orWithNegatedBits(
            nullRows_.asMutableRange().bits(), rawNulls, 0, size);
      }
    }
  }
  nullRows_.updateBounds();
  return nullRows_;
}

} // namespace facebook::velox::exec

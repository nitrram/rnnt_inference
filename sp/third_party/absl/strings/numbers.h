//
// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef ABSL_STRINGS_NUMBERS_H_
#define ABSL_STRINGS_NUMBERS_H_

#include <sstream>

#include "./string_view.h"

namespace absl {

// TODO(taku): Re-implement this, as it is slow.
template <typename T>
inline bool SimpleAtoi(absl::string_view s, T *result) {
  std::stringstream ss;
  return (ss << s.data() && ss >> *result);
}

}  // namespace absl
#endif  // ABSL_STRINGS_NUMBERS_H_

// Copyright 2022 Seznam.cz, a.s.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.!

#pragma once

#include <cstdint>
#include <cstddef>

namespace spr::inference {

  // returns feat_mat_size divided by number of the mel banks
  size_t create_feat_inp(const int16_t*, size_t, float**);

  // logs the input
  size_t norm_inp(float*, size_t);
}
//eof

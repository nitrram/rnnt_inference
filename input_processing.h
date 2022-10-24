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

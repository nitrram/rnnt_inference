#include "softmax_neon.h"
#include <iostream>

int main(int argc, char *argv[]) {

  float arr[5] = {1.5107, 0.3022, 1.0195, 2.6572,  0.3313};

  /*  float32x4_t exp1_inp = vld1q_f32(arr);
  //float32x4_t exp1 = exp_ps(exp1_inp);
  //float32x4_t log1 = log_ps(exp1);

  std::cout << "[" << exp1[0] << ", " << exp1[1] << ", " << exp1[2] << ", " << exp1[3] << "]\n";

  std::cout << "[" << log1[0] << ", " << log1[1] << ", " << log1[2] << ", " << log1[3] << "]\n";
  */
  

  softmax(arr, 5);

  for(size_t i =0 ; i < 5; ++i) {
    std::cout << arr[i] << ", ";
  }
  std::cout << std::endl;
  

  return 0;
}

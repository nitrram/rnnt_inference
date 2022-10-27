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

#include "pulse_rec.h"

#include <pulse/error.h>

#ifdef __cplusplus
#include <cstdio>
#include <cstdlib>
#else
#include <stdio.h>
#include <stdlib.h>
#endif

extern int _exitting;


int start_recording(int (*callback)(buf_t *buf, size_t siz)) {

  static const pa_sample_spec ss = {
    .format = PA_SAMPLE_S16LE,
      .rate = PULSE_REC_SAMPLE_RATE,
      .channels = 1
  };
  pa_simple *s = NULL;
  int ret = 1;
  int error;

  int size = 512 * sizeof(buf_t);
  int psize;

  _buffer = (buf_t*)malloc(size);


  /* Create the recording stream */
  if (!(s = pa_simple_new(NULL, "spr_pulse", PA_STREAM_RECORD, NULL, "rnnt inference", &ss, NULL, NULL, &error))) {
    fprintf(stderr, __FILE__": pa_simple_new() failed: %s\n", pa_strerror(error));
    goto finish;
  }

  int rc;
  while (!_exitting) {

    /* Record some data ... */
    if ( (rc=pa_simple_read(s, _buffer, size, &error)) < 0) {
      fprintf(stderr, __FILE__": pa_simple_read() failed: %s\n", pa_strerror(error));
      goto finish;
    }

    psize = size;
    while( (psize -= callback(_buffer, size) ) > 0 ) {
      fprintf(stderr, "short write: wrote %d bytes\n", rc);
    }
  }

  ret = 0;

 finish:

  if (s)
    pa_simple_free(s);

  if(_buffer)
    free(_buffer);

  return ret;
}
//eof

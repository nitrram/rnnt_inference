#pragma once


namespace spr::decoder {

    class TransducerBeamSearch {
    public:

        int search(float *enc_out, size_t total_size, size_t time_steps);
    };

}
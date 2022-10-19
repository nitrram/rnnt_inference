# rnnt_inference
RNN-T onnx model decoder

# Prerequisities
## libonnxruntime 
Compile your own (optionally using required_operators.config from the followin link, in order to compress the shared object size), or use libonnxruntime.so.1.12.0 from here (https://drive.google.com/drive/folders/1CgBwNie4_myV0v8z8--3khZMDddstaDo?usp=sharing)
 => copy the library among the source files & headers; create `ln -s libonnxruntime.so.1.12.0 libonnxruntime.so`
 
## model
https://drive.google.com/drive/folders/1CgBwNie4_myV0v8z8--3khZMDddstaDo?usp=sharing
Copy the model files (rnnt_tn.ort, rnnt_pn.ort and rnnt_cn.ort) into the directory where lays the exe binary.

## wav
https://drive.google.com/drive/folders/1CgBwNie4_myV0v8z8--3khZMDddstaDo?usp=sharing
Copy the wav (common_voice_cs_25695144_16.wav) next to the exe binary too.


# Run
./onnxinfer [path_to_16bit_16khz_wav]
 

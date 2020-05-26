# faceDetection
Face detector
create some folders:
```
mkdir build
mkdir bin
```
to build project execute following:
```
cd build
cmake ..
make
```
add path to external dependecies:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/faceDetection/Plugins/cuda/lib
```
to launch project:
```
../bin/libil ../image
```

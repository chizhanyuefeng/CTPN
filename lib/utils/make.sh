cython bbox.pyx
cython cython_nms.pyx
cython gpu_nms.pyx
python3 setup.py build_ext --inplace
rm -rf build

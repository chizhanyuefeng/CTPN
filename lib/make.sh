cython ./bbox_utils/bbox.pyx
cython ./nms/cython_nms.pyx
python3 setup.py build_ext --inplace
rm -rf build

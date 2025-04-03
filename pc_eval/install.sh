# sudo rm /usr/local/cuda 
# sudo ln -s /usr/local/cuda-11.1 /usr/local/cuda

cd ./chamfer3D
python setup.py install

cd ../emd_module
python setup.py install

cd ../evaluate_code
cmake .
make

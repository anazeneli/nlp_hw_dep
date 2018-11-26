Environment Setup  
virtualenv nlphw  
source nlphw/bin/activate  
  
cd nlphw/  
pip install numpy  
pip install matplotlib  
pip install mercurial  
pip install RISE  
  
mkdir dynet-base  
cd dynet-base  
pip install cython  
git clone https://github.com/clab/dynet.git  
hg clone https://bitbucket.org/eigen/eigen -r 699b659  
cd dynet  
mkdir build  
cd build  
export pth="$PWD"  
cmake .. -DEIGEN3_INCLUDE_DIR=$pth/../../eigen/ -DPYTHON=$pth/../../../bin/python  
make -j 2  
cd python  
python ../../setup.py build --build-dir=.. --skip-build install  
export DYLD_LIBRARY_PATH=$pth/dynet/:$DYLD_LIBRARY_PATH  
export LD_LIBRARY_PATH=$pth/dynet/:$LD_LIBRARY_PATH  

Vocabulary Creation for Features   
Deep networks work with vectors, not strings. Therefore, first you have to  
create vocabularies for word features, POS features, dependency label features  
and parser actions.  
Run the following command:  
python src/gen_vocab.py trees/train.conll data/vocabs  
  
After that, you should have the following files:    
• data/vocabs.word  
• data/vocabs.pos  
• data/vocabs.labels  
• data/vocabs.actions  
  
Data Generation   
python src/gen.py trees/train.conll data/train.data  
python src/gen.py trees/dev.conll data/dev.data  
  
Neural Network Implementation   
Train the network as follows:   
python src/depModel.py trees/dev.conll outputs/dev.out models/model_part1  
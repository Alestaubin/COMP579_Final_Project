## Installation
```
conda create -n comp579 python=3.9 -y
conda activate comp579
git clone https://github.com/jpmorganchase/abides-jpmc-public
cd abides-jpmc-public
sh install.sh
# Change the requirements.txt file
echo -e "ray[rllib]==1.8.0\npomegranate==0.15.0\ncoloredlogs==15.0.1\ngym==0.23.0\nnumpy==1.22.0\npandas==2.0.0\npsutil==5.8.0\nscipy==1.10.0\ntqdm==4.61.1" > requirements.txt
pip install -r requirements.txt
pip install "protobuf<3.21"
cd ..
python tester.py
```

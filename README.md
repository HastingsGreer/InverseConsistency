How to run on triangles and circles dataset, from the beginning

1:Install anaconda
2:
git clone https://github.com/HastingsGreer/InverseConsistency

conda create -n inverseConsistency python==3.7

conda activate inverseConsistency

conda install pytorch
pip install -r requirements.txt

python training_scripts/triangle_example.py

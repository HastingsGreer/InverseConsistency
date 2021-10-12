# ICON: Learning Regular Maps through Inverse Consistency

![Demo figure](notebooks/paper_figures/Intro_NewLabels-2.png)

This is the official repository for  
ICON: Learning Regular Maps through Inverse Consistency.  
Hastings Greer, Roland Kwitt, Francois-Xavier Vialard, Marc Niethammer.  
ICCV 2021 https://arxiv.org/abs/2105.04459



## Running our code

To run our pretrained model in the cloud on 4 sample image pairs from OAI knees (as above), visit [our google colab notebook](https://colab.research.google.com/drive/1Pd3ua_NZTem3xtBvDxertzi7u3E233ZL?usp=sharing)

----------------

To train from scratch on the synthetic triangles and circles dataset:

1: Install pytorch

2:
```
git clone https://github.com/HastingsGreer/InverseConsistency
cd InverseConsistency

pip install -r requirements.txt

python training_scripts/2d_triangles_example.py
```

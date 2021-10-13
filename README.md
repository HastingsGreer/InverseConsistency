# ICON: Learning Regular Maps through Inverse Consistency

![Demo figure](notebooks/paper_figures/Intro_NewLabels-2.png)

This is the official repository for  

**ICON: Learning Regular Maps through Inverse Consistency.**   
Hastings Greer, Roland Kwitt, Francois-Xavier Vialard, Marc Niethammer.  
_ICCV 2021_ https://arxiv.org/abs/2105.04459

## Video Presentation

https://www.youtube.com/watch?v=7kZsJ3zWDCA



## Running our code

We are available on PyPI!
```
pip install icon-registration
````

To run our pretrained model in the cloud on 4 sample image pairs from OAI knees (as above), visit [our google colab notebook](https://colab.research.google.com/drive/1Pd3ua_NZTem3xtBvDxertzi7u3E233ZL?usp=sharing)

----------------

To train from scratch on the synthetic triangles and circles dataset:

```
git clone https://github.com/HastingsGreer/InverseConsistency
cd InverseConsistency

pip install -r requirements.txt

python training_scripts/2d_triangles_example.py
```

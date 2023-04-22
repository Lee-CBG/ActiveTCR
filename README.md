# ActiveTCR: An Active Learning Framework for Cost-Effective TCR-Epitope Binding Affinity Prediction

ActiveTCR is a unified framework designed to minimize the annotation cost and maximize the predictive performance of T-cell receptor (TCR) and epitope binding affinity prediction models. It incorporates active learning techniques to iteratively search for the most informative unlabeled TCR-epitope pairs, reducing annotation costs and redundancy. By leveraging four query strategies and comparing them to a random sampling baseline, ActiveTCR demonstrates significant cost reduction and improved performance in TCR-epitope binding affinity prediction. ActiveTCR is the first systematic investigation of data optimization in the context of TCR-epitope binding affinity prediction.
<br/>
<br/>

<p align="center"><img width=100% alt="Overview" src="https://github.com/Lee-CBG/ActiveTCR/blob/main/figures/Fig5_method.png"></p>

## Publication
<b>An Active Learning Framework for Cost-Effective TCR-Epitope Binding Affinity Prediction </b> <br/>
[Pengfei Zhang](https://github.com/pzhang84)<sup>1,2</sup>, [Seojin Bang](http://seojinb.com/)<sup>3</sup>, [Heewook Lee](https://scai.engineering.asu.edu/faculty/computer-science-and-engineering/heewook-lee/)<sup>1,2, *</sup><br/>
<sup>1 </sup>School of Computing and Augmented Intelligence, Arizona State University, <sup>2 </sup>Biodesign Institute, Arizona State University, <sup>3 </sup>Google Research, Brain Team, Google <br/>
Published in: **To be added**


[Paper](#) | [Code](https://github.com/Lee-CBG/ActiveTCR) | [Poster](#) | [Slides](#) | Presentation ([YouTube](#))

## Major results of ActiveTCR
1. **Use case a: reducing more than 40% annotation cost for unlabel TCR-epitope pools.**
<br/>
<p align="center"><img width=100% alt="Overview" src="https://github.com/Lee-CBG/ActiveTCR/blob/main/figures/Fig1.png"></p>
<br/>

2. **Use case b: minimizing more than 40% redundancy among already annotated TCR-epitope pairs.**
<br/>
<p align="center"><img width=100% alt="Overview" src="https://github.com/Lee-CBG/ActiveTCR/blob/main/figures/Fig2.png"></p>
<br/>

## Dependencies

+ Linux
+ Python 3.6.13
+ Keras 2.6.0
+ TensorFlow 2.6.0

## Steps to train a Binding Affinity Prediction model for TCR-epitope pairs.

### 1. Clone the repository
```bash
git clone https://github.com/Lee-CBG/ActiveTCR
cd ActiveTCR/
conda create --name bap python=3.6.13
pip install -r requirements.txt
source activate bap
```

### 2. Prepare TCR-epitope pairs for training and testing
- Download training and testing data from `datasets` folder.
- Obtain embeddings for TCR and epitopes following instructions of [`catELMo`](https://github.com/Lee-CBG/catELMo/tree/main/embedders). Or directly download embeddings from [Dropbox](https://www.dropbox.com/s/zkjs8od969rtbzq/catELMo_4_layers_1024.pkl?dl=0).


### 3. Train and test models
An example for `use case a` of ActiveTCR: reducing annotation cost for unlabeled TCR-epitope pools.

```bash
python -W ignore main.py \
                --split epi \
                --active_learning True \
                --query_strategy entropy_sampling \
                --train_strategy retrain \
                --query_balanced unbalanced \
                --gpu 0 \
                --run 0
```

An example for `use case b` of ActiveTCR: minimizing redundancy among labeled TCR-epitope pairs.

```bash
python -W ignore main.py \
                --split epi \
                --query_strategy entropy_sampling \
                --train_strategy retrain \
                --query_balanced unbalanced \
                --gpu 1 \
                --run 0
```

## Citation
If you use this code or use our catELMo for your research, please cite our paper:
```

```

## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
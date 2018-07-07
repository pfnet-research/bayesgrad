# Bayesgrad
BayesGrad: Explaining Predictions of Graph Convolutional Networks

The paper is available on arXiv, [https://arxiv.org/abs/1807.01985](https://arxiv.org/abs/1807.01985).

<p float="left" align="middle">
  <img src="https://github.com/pfnet-research/bayesgrad/blob/master/assets/tox21_pyridine/6_bayes.png" width="250" /> 
  <img src="https://github.com/pfnet-research/bayesgrad/blob/master/assets/tox21_srmmp/3_bayes.png" width="250" />
  <img src="https://github.com/pfnet-research/bayesgrad/blob/master/assets/delaney_solubility/3.png" width="250" />
</p>
<p align="middle">
  From left: tox21 pyridine (C5H5N), tox21 SR-MMP, delaney solubility visualization.
</p>

## Citation
If you find our work useful in your research, please consider citing:

```
@article{akita2018bayesgrad,
  title={BayesGrad: Explaining Predictions of Graph Convolutional Networks},
  author={Akita, H, and Nakago, K and Komatsu, T and Sugawara, Y and Maeda, S and Baba, Y and Kashima, H},
  journal={arXiv preprint arXiv:1807.01985},
  year={2018}
}
```

## Setup

[Chainer Chemistry](https://github.com/pfnet-research/chainer-chemistry) [1] is used in our code.
It is an extension library for deep learning framework [Chainer](https://github.com/chainer/chainer) [2],
and it supports several graph-convolutional neural network together with chemical dataset management.

The experiment is executed under following environment:

 - OS: Linux
 - python: 3.6.1
 - conda version: 4.4.4

```bash
conda create -n bayesgrad python=3.6.1
source activate bayesgrad
pip install chainer==4.2.0
# install master branch of chainer-chemistry
pip install git+https://github.com/pfnet-research/chainer-chemistry
conda install -c rdkit rdkit==2017.09.3.0
pip install matplotlib==2.2.2
pip install future==0.16.0
pip install cairosvg==2.1.3
pip install ipython==5.1.0
```

[Note]
Please install specified python version & rdkit version.
Latest python version and rdkit may not work well as discussed [here](https://github.com/pfnet-research/chainer-chemistry/issues/138).
If you face error try
```bash
conda install libgcc
```

If you want to use GPU, please install `cupy` as well.
```bash
# XX should be CUDA version (80, 90 or 91)
pip install cupy-cudaXX==4.2.0
```

## Experiments

Each experiment can be executed as follows.

### Tox21 Pyridine experiment
Experiments described in Section 4.1 in the paper. Tox21 [3] dataset is used.

```bash
cd experiments/tox21
```

#### Training with all train data, plot precision-recall curve

Set `-g -1` to use CPU, `-g 0` to use GPU. 
```bash
python train_tox21.py --iterator-type=balanced --label=pyridine --method=ggnndrop --epoch=50 --unit-num=16 --n-layers=1 -b 32 --conv-layers=4 --num-train=-1 --dropout-ratio=0.25 --out=results/ggnndrop_pyridine -g 0
python plot_precision_recall.py --dirpath=results/ggnndrop_pyridine
```

#### Visualization with trained model
See `visualize-saliency-pyrigine.ipynb`.

<p float="left" align="middle">
  <img src="https://github.com/pfnet-research/bayesgrad/blob/master/assets/tox21_pyridine/6_bayes.png" width="250" /> 
  <img src="https://github.com/pfnet-research/bayesgrad/blob/master/assets/tox21_pyridine/13_bayes.png" width="250" /> 
  <img src="https://github.com/pfnet-research/bayesgrad/blob/master/assets/tox21_pyridine/21_bayes.png" width="250" /> 
</p>

Our method successfully focuses on pyridine (C5H5N) substructures.
 
#### Training 30 different models with few train data, calculate RPC-AUC score
Argument: `-1` to use CPU, `0` to use GPU. 

Note that this experiment takes time (took around 2.5 hours with GPU in our environment),
since it trains 30 different models.

```bash
bash -x ./train_few_with_seeds.sh 0
bash -x ./calc_prcauc_with_seeds.sh 0
```

Then see `results/ggnndrop_pyridin_numtrain1000-seed0-29/prcauc_stats_absolute_0.15.csv` for the results.

### Tox21 SR-MMP experiment
Experiments described in Section 4.2 in the paper.  Tox21 [3] dataset is used.

```bash
cd experiments/tox21
```

#### Training the model
Set `-g -1` to use CPU, `-g 0` to use GPU.
```bash
python train_tox21.py --iterator-type=balanced --label=SR-MMP --method=nfpdrop --epoch=200 --unit-num=16 --n-layers=1 -b 32 --conv-layers=4 --num-train=-1 --dropout-ratio=0.25 --out=results/nfpdrop_srmmp -g 0
```

#### Visualization of tox21 data & Tyrphostin 9 with trained model
See `visualize-saliency-tox21.ipynb`.

Jupyter notebook interactive visualization:
<p float="left" align="middle">
  <img src="https://user-images.githubusercontent.com/4609798/42087970-d47bb666-7bd2-11e8-8d75-5770ccfbdb3b.gif" />
</p>

Several picked images:
<p float="left" align="middle">
  <img src="https://github.com/pfnet-research/bayesgrad/blob/master/assets/tox21_srmmp/2_bayes.png" width="250" />
  <img src="https://github.com/pfnet-research/bayesgrad/blob/master/assets/tox21_srmmp/3_bayes.png" width="250" />
  <img src="https://github.com/pfnet-research/bayesgrad/blob/master/assets/tox21_srmmp/27_bayes.png" width="250" />
</p>

Toxicity mechanism is still in an active research topic and it is difficult to quantitatively analyze its results.
We hope these visualization helps to analyze and establish further knowledge about toxicity.

### Solubility experiment
Experiment done in Section 4.3 in the paper. ESOL [4] dataset (provided by MoleculeNet [5]) is used.

```bash
cd experiments/delaney
```

#### Training the model
Set `-g -1` to use CPU, `-g 0` to use GPU.

```bash
python train.py -e 100 -n 3 --method=nfpdrop -g 0
```

#### Visualization with trained model
```bash
python plot.py --dirpath=./results/nfpdrop_M30_conv3_unit32_b32
```

<p float="left" align="middle">
  <img src="https://github.com/pfnet-research/bayesgrad/blob/master/assets/delaney_solubility/3.png" width="250" />
  <img src="https://github.com/pfnet-research/bayesgrad/blob/master/assets/delaney_solubility/19.png" width="250" />
  <img src="https://github.com/pfnet-research/bayesgrad/blob/master/assets/delaney_solubility/54.png" width="250" />
</p>

Red color represents these atoms are hydrophilic, and blue color represents hydrophobic.
Above figure is consistent with fundamental physicochemical knowledge as explained in the paper.

## Saliency Calculation

Although only results of gradient method [6, 7, 8] are reported in the paper, 
this repository contains saliency calculation code for several other algorithms as well.

We can apply SmoothGrad [8] and/or BayesGrad (Ours) into following algorithms.

- Vanilla Gradients [6, 7]
- Integrated Gradients [9]
- Occlusion [10]

The code design is inspired by [PAIR-code/saliency](https://github.com/PAIR-code/saliency).

## License

Our code is released under MIT License (see [LICENSE](https://github.com/pfnet-research/bayesgrad/blob/master/LICENSE) file for details).

## Reference

[1] pfnet research. chainer-chemistry https://github.com/pfnet-research/chainer-chemistry

[2] Seiya Tokui, Kenta Oono, Shohei Hido, and Justin Clayton. Chainer: a next-generation open source framework for deep learning. In *Proceedings of Workshop on Machine Learning Systems (LearningSys) in Advances in Neural Information Processing System (NIPS) 28*, 2015.

[3] Ruili Huang, Menghang Xia, Dac-Trung Nguyen, Tongan Zhao, Srilatha Sakamuru, Jinghua Zhao, Sampada A Shahane, Anna Rossoshek, and Anton Simeonov. Tox21challenge to build predictive models of nuclear receptor and stress response pathways as mediated by exposure to environmental chemicals and drugs. Frontiers in Environmental Science, 3:85, 2016.

[4] John S. Delaney. Esol: Estimating aqueous solubility directly from molecular structure. Journal of Chemical Information and Computer Sciences, 44(3):1000{1005,2004. PMID: 15154768.

[5] Zhenqin Wu, Bharath Ramsundar, Evan N. Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S. Pappu, Karl Leswing, Vijay Pande, MoleculeNet: A Benchmark for Molecular Machine Learning, arXiv preprint, arXiv: 1703.00564, 2017.

[6] Dumitru Erhan, Yoshua Bengio, Aaron Courville, Pascal Vincent. Visualizing Higher-Layer Features of a Deep Network. 2009.

[7] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: Visualising image classication models and saliency maps. arXiv preprint arXiv:1312.6034, 2013.

[8] Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viegas, and Martin Wattenberg. SmoothGrad: removing noise by adding noise. arXiv preprint arXiv:1706.03825, 2017.

[9] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In Doina Precup and Yee Whye Teh (eds.), 
Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pp. 3319–3328, International Convention Centre, Sydney, Australia, 06–11 Aug 2017. PMLR.
URL http://proceedings.mlr.press/v70/sundararajan17a.html.

[10] Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In
European conference on computer vision, pp. 818–833. Springer, 2014.

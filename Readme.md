# ContribNet
This code implements an encoder-decoder model called **ContribNet** for detecting the attribute-object composition or attribute-object pair e.g. diced cheese, engraved coin etc. in  the given image.

![contribnet](https://user-images.githubusercontent.com/63097128/97393718-2d479c80-1909-11eb-8b25-a40ec919d6ee.png)

The codebase has implementations of four different models:
1. contribnet
2. genmodel
3. contribvisprod
4. visprod 


## Prerequisites
The code is tested on Python v3.6 and PyTorch v1.5

**Packages**: Install using `pip install -r utils/requirements.txt`

**Datasets and Features**: We include a script to download all the necessary data: images, features and metadata for the two datasets, pretrained SVM classifier weights, tensor completion code and pretrained models. It must be run before training the models.
```bash
bash utils/download_data.sh
```
(Same as the setup instructions at: `https://github.com/Tushar-N/attributes-as-operators`)

## Training the model

Training the models with different parameters
```bash
python train.py --p_model resnet152 --dataset zappos --data_dir data/ut-zap50k --batch_size 128 --lr 1e-4 --max_epochs 600 --glove_init --model genmodel --dropout=0.3 --cv_dir cv/zappos/genmodel
```

```bash
python train.py --p_model resnet152 --dataset mitstates --data_dir data/mit-states --batch_size 128 --lr 1e-4 --max_epochs 600 --glove_init --model contribnet --dropout=0.3 --cv_dir cv/mitstates/contribnet
```

## Evaluating the model
Evaluating the models with different parameters

```bash
python test.py --p_model resnet152 --dataset zappos --data_dir data/ut-zap50k --batch_size 512 --glove_init --model genmodel --dropout=0.3 --load cv/zappos/genmodel/resnet152_ckpt_E_600_At_0.628_O_0.747_Cl_0.462_Op_0.204.t7
```

```bash
python test.py --p_model resnet152 --dataset mitstates --data_dir data/mit-states --batch_size 512 --glove_init --model contribnet --dropout=0.3 --load cv/mitstates/contribnet/resnet152_ckpt_E_600_At_0.188_O_0.227_Cl_0.120_Op_0.114.t7
```

## Citations

If you find this project helps your research, please cite the repository and paper in your work.

The repository:
```
@misc{github-contribnet-patil,
title = {ContribNet Codebase in PyTorch},
author = {Patil, Charulata},
year = {2020},
note = {\url{https://github.com/cjpatil/contribnet}},
}
```
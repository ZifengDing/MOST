# MOST

This is the code for our paper:

Learning Meta Representations of One-shot Relations for Temporal Knowledge Graph Link Prediction

## Installation
Create a conda environment

```
conda create --prefix ./most_env python=3.9
conda activate ./most_env
```

Install MOST requirements
```
pip install -r requirements.txt
```
Install PyTorch
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
Install torch-scatter
```
conda install pytorch-scatter -c pyg
```

## Datasets Generation

Download and unzip data.zip and software.zip. 

Generated ICEWS-one_int and ICEWS-one_ext datasets are already in data/dataset. Please copy to the software folder for model training and evaluation.

Due to the file size limit on softconf, we cannot upload the whole GDELT-based datasets. We will release GDELT-based datasets after our paper is accepted.

In case you want to generate your own dataset, please use files in data/data_generator. E.g. generate two datasets of ICEWS-one_int and ICEWS-one_ext:

```
cd ICEWS0515
python MOST_ICEWS0515_INT.py
python MOST_ICEWS0515_EXT.py
```

## Training Model

Training MOST-TA on all four datasets:

```
python main.py --dataset ICEWS0515_INTERPOLATION --is_rawts --embsize 100
python main.py --dataset ICEWS0515_EXTRAPOLATION --is_rawts --embsize 100
python main.py --dataset GDELT_INTERPOLATION --is_rawts --embsize 50
python main.py --dataset GDELT_EXTRAPOLATION --is_rawts --embsize 100
```

Training MOST-TD on all four datasets:

```
python main.py --dataset ICEWS0515_INTERPOLATION --embsize 100
python main.py --dataset ICEWS0515_EXTRAPOLATION --embsize 200
python main.py --dataset GDELT_INTERPOLATION --embsize 50
python main.py --dataset GDELT_EXTRAPOLATION --embsize 100
```

# MOST

This is the code for our paper:

[Learning Meta Representations of One-shot Relations for Temporal Knowledge Graph Link Prediction](https://arxiv.org/pdf/2205.10621)

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

Download and unzip data.zip from this [link](https://www.dropbox.com/scl/fo/95hiznnm5fd5rdwu33jr4/AHtqm52fhjM-Tey-Ug0TmWc?rlkey=igk6rb4fg9sf57j5c54uca5zj&st=ot7lxgtt&dl=0). 

Generated ICEWS-one_int and ICEWS-one_ext datasets are already in data/dataset. Please copy to the software folder for model training and evaluation.

Please use files in data/data_generator to generate GDELT-one_int and GDELT-one_ext:

```
cd GDELT
python MOST_GDELT_INT.py
python MOST_GDELT_EXT.py
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

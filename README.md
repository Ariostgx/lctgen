# Language Conditioned Traffic Generation

[**Language Conditioned Traffic Generation**](https://arxiv.org/abs/2307.07947)                                     
[Shuhan Tan](https://ariostgx.github.io/website/)<sup>1</sup>, [Boris Ivanovic](https://www.borisivanovic.com/)<sup>2</sup>,   [Xinshuo Weng](https://www.xinshuoweng.com/)<sup>2</sup>,  [Marco Pavone](https://research.nvidia.com/person/marco-pavone/)<sup>2</sup>,  [Philipp Krähenbühl](https://www.philkr.net/)<sup>1</sup>

<sup>1</sup>UT Austin, <sup>2</sup> NVIDIA

Conference on Robot Learning (CoRL), 2023

[**Webpage**](https://ariostgx.github.io/lctgen/) | 
[**Video**](https://www.youtube.com/watch?v=T5GFOxzw0aw) |
[**Paper (Arxiv)**](https://arxiv.org/abs/2307.07947) |
[**Demo (Colab)**](https://colab.research.google.com/drive/1acVvMsts464_HRgGStjvI1n1b55wuLb4?usp=sharing)

## News
* **`2 Oct, 2023`:**  Initial code release.
* **`30 Aug, 2023`:**  Our paper was accepted at [CoRL 2023](https://www.corl2023.org/)!
* **`16 Jul, 2023`:** We released our paper on [arXiv](https://arxiv.org/abs/2307.07947).
## Demo
We provide an online demo in [**Colab**](https://colab.research.google.com/drive/1acVvMsts464_HRgGStjvI1n1b55wuLb4?usp=sharing). You can try it without any local installation. This demo includes:

1. Generate traffic scene with existing Structured Representation.
2. Generate traffic scene from existing LLM output.
3. Generate traffic scene with natural language and GPT-4 (requires OpenAI API Key).

Please also refer to the code inside for demonstrations of basic model usage.

## Setup local environment

```bash
# Clone the code to local
git clone https://github.com/Ariostgx/lctgen.git
cd lctgen

# Create virtual environment
conda create -n lctgen python=3.8
conda activate lctgen

# You should install pytorch by yourself to make them compatible with your GPU
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install other dependency
pip install -r requirements.txt
```

## Quick start
We provide a demo dataset and a pretrained model for quick start. 

This is meant to be a quick demonstration of the model usage. For complete training and evaluation, please refer to the next section.

### Download demo data 
Please download the demo dataset of 30 scene clips `demo_data.zip` from [here](https://drive.google.com/file/d/17_TI-q4qkCOt988spWIZCqDLkZpMSptO/view?usp=drive_link).
 
And then unzip all the files inside into `data/demo/waymo` folder.

### Download pretrained model

Please download the example model checkpoint `example.ckpt` from [here](https://drive.google.com/file/d/1_s_35QO6OiHHgDxHHAa7Djadm-_I7Usr/view?usp=drive_link).

And then put it into `checkpoints` folder.

### Setup logger

By default we use [wandb](https://wandb.ai/site) to log the training process. You will need to login to wandb first.

You can also use [tensorboard](https://www.tensorflow.org/tensorboard) by setting `_C.LOGGER` to `tsboard` in `lctgen/config/path_cfg.py`.

### Evaluate with demo data
````
python lctgen/main.py  --run-type eval --exp-config cfgs/demo_inference.yaml
````

### Training with demo data
````
python lctgen/main.py  --run-type train --exp-config cfgs/demo_train.yaml
````

## Training and evaluation

### Download full dataset
We follow the data processing process in [**TrafficGen**](https://github.com/metadriverse/trafficgen/tree/main#cluster-training):

1. Download from Waymo Open Dataset:

- Register your Google account in: https://waymo.com/open/
- Open the following link with your Google account logged in: https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_1_0
- Download all the proto files from ``waymo_open_dataset_motion_v_1_1_0/uncompressed/scenario/training_20s``
- Move download files to ``PATH_A``, where you store the raw tf_record files.


2. Data Preprocess
````
python scripts/process_all_data.py PATH_A PATH_B
````
 - Note: ``PATH_B`` is where you store the processed data.

3. Change `_C.DATASET.DATA_PATH` to ``PATH_B`` in `lctgen/config/path_cfg.py`.

### Evaluate with demo data
````
python lctgen/main.py  --run-type eval --exp-config cfgs/inference.yaml
````

### Training with full data
````
python lctgen/main.py  --run-type train --exp-config cfgs/train.yaml
````

## Related repositories

We use code in [**TrafficGen**](https://github.com/metadriverse/trafficgen/) for data processing and visualization. TrafficGen related code is in `trafficgen` folder.

## To Do
- [ ] Add instructive scene editing.
- [ ] Add dataset of input text descriptions and LLM outputs.
- [x] Initial repo & demo

## Acknowledgement
We thank Yuxiao Chen, Yulong Cao, and Danfei Xu for their insightful discussions. This project is supported by the National Science Foundation under Grant No. IIS-1845485.

We also thank authors of [**TrafficGen**](https://github.com/metadriverse/trafficgen/) for their open source code.

## Citation

```latex
@inproceedings{
    tan2023language,
    title={Language Conditioned Traffic Generation},
    author={Shuhan Tan and Boris Ivanovic and Xinshuo Weng and Marco Pavone and Philipp Kraehenbuehl},
    booktitle={7th Annual Conference on Robot Learning},
    year={2023},
    url={https://openreview.net/forum?id=PK2debCKaG}
}
```
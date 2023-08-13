<p align="center"><img src="hpsv2/assets/hps_banner.png"/ width="100%"><br></p>

# HPS v2: Benchmarking Text-to-Image Generative Models

[![PyPI](https://img.shields.io/pypi/v/hpsv2)](https://pypi.org/project/hpsv2/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/hpsv2)
[![Arxiv](https://img.shields.io/badge/Arxiv-2306.09341-pink)](https://arxiv.org/abs/2306.09341)
[![Huggingface](https://img.shields.io/badge/Hugging_face-HPSv2-yellow)](https://huggingface.co/spaces/xswu/HPSv2)
[![PyPI - License](https://img.shields.io/pypi/l/hpsv2)](https://www.apache.org/licenses/LICENSE-2.0.html)

This is the official repository for the paper: [Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis](https://arxiv.org/abs/2306.09341). 

## Updates
*  [08/02/2023] We released the [PyPI package](https://pypi.org/project/hpsv2/). You can learn how to use it from the [Quick start section](#quick-start).
*  [08/02/2023] Updated [test.json](https://huggingface.co/datasets/zhwang/HPDv2/blob/main/test.json) to include raw annotation by each annotator.
*  [07/29/2023] We included `SDXL Refiner 0.9` model in the benchmark.
*  [07/29/2023] We released [the benchmark and HPD v2 test data](https://huggingface.co/datasets/zhwang/HPDv2). HPD v2 train data will be released soon.
*  [07/27/2023] We included `SDXL Base 0.9` model in the benchmark.
*  [07/26/2023] We updated our [compressed checkpoint](https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt).
*  [07/19/2023] Live demo is available at 🤗[Hugging Face](https://huggingface.co/spaces/xswu/HPSv2).
*  [07/18/2023] We released our [test data](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155172150_link_cuhk_edu_hk/EVnjOngvDO1MhIp7hVr8GXgBmxVDcSk7s9Xuu9srO4YLbA?e=8PqYud).

## Overview 
<p align="center"><img src="hpsv2/assets/overview.png"/ width="100%"><br></p>

**Human Preference Dataset v2 (HPD v2)**: a large-scale (798k preference choices / 430k images), a well-annotated dataset of human preference choices on images generated by text-to-image generative models. 

**Human Preference Score v2 (HPS v2)**: a preference prediction model trained on HPD v2. HPS v2 can be used to compare images generated with the same prompt. We also provide a fair, stable, and easy-to-use set of evaluation prompts for text-to-image generative models.

## The HPS v2 benchmark
The HPS v2 benchmark evaluates models' capability of generating images of 4 styles: *Animation*, *Concept-art*, *Painting*, and *Photo*. 

**The benchmark is actively updating, email us @ tgxs002@gmail.com or raise an issue if you feel your model/method needs to be included in this benchmark!**

| Model                 | Animation | Concept-art | Painting | Photo    | Averaged |
| ---------------------| --------- | ----------- | -------- | -------- | -------- |
| Dreamlike Photoreal 2.0 | 0.2824  | 0.2760      | 0.2759   | 0.2799   | 0.2786 |
| SDXL Refiner 0.9      | 0.2845    | 0.2766      | 0.2767   | 0.2746   | 0.2780 |
| Realistic Vision      | 0.2822    | 0.2753      | 0.2756   | 0.2775   | 0.2777 |
| SDXL Base 0.9         | 0.2842    | 0.2763      | 0.2760   | 0.2729   | 0.2773 |
| Deliberate            | 0.2813    | 0.2746      | 0.2745   | 0.2762   | 0.2767 |
| ChilloutMix           | 0.2792    | 0.2729      | 0.2732   | 0.2761   | 0.2754 |
| MajicMix Realistic    | 0.2788    | 0.2719      | 0.2722   | 0.2764   | 0.2748 |
| Openjourney           | 0.2785    | 0.2718      | 0.2725   | 0.2753   | 0.2745 |
| DeepFloyd-XL          | 0.2764    | 0.2683      | 0.2686   | 0.2775   | 0.2727 |
| Epic Diffusion        | 0.2757    | 0.2696      | 0.2703   | 0.2749   | 0.2726 |
| Stable Diffusion v2.0 | 0.2748    | 0.2689      | 0.2686   | 0.2746   | 0.2717 |
| Stable Diffusion v1.4 | 0.2726    | 0.2661      | 0.2666   | 0.2727   | 0.2695 |
| DALL·E 2              | 0.2734    | 0.2654      | 0.2668   | 0.2724   | 0.2695 |
| Versatile Diffusion   | 0.2659    | 0.2628      | 0.2643   | 0.2705   | 0.2659 |
| CogView2              | 0.2650    | 0.2659      | 0.2633   | 0.2644   | 0.2647 |
| VQGAN + CLIP          | 0.2644    | 0.2653      | 0.2647   | 0.2612   | 0.2639 |
| DALL·E mini           | 0.2610    | 0.2556      | 0.2556   | 0.2612   | 0.2583 |
| Latent Diffusion      | 0.2573    | 0.2515      | 0.2525   | 0.2697   | 0.2578 |
| FuseDream             | 0.2526    | 0.2515      | 0.2513   | 0.2557   | 0.2528 |
| VQ-Diffusion          | 0.2497    | 0.2470      | 0.2501   | 0.2571   | 0.2510 |
| LAFITE                | 0.2463    | 0.2438      | 0.2443   | 0.2581   | 0.2481 |
| GLIDE                 | 0.2334    | 0.2308      | 0.2327   | 0.2450   | 0.2355 |

## Quick Start

### Installation

```shell
# Method 1: Pypi download and install
pip install hpsv2

# Method 2: install locally
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2
python -m pip install . 

# Optional: checkpoint and images will be downloaded here
# default: ~/.cache/hpsv2/
export HPS_ROOT=/your/cache/path
```

After installation, we show how to:
- [Compare images using HPS v2](#image-comparison).
- [Reproduce our benchmark](#benchmark-reproduction).
- [Evaluate your own model using HPS v2](#custom-evaluation).
- [Evaluate our preference model](#preference-model-evaluation).

We also provide [command line interfaces](#command-line-interface) for debugging purposes.

### Image Comparison

You can score and compare several images generated by the same prompt by running the following code:

```python
import hpsv2

result = hpsv2.score_(imgs_path, '<prompt>') 
# imgs_path can be a list of image paths, with the images generated by the same prompt
# or image path of string type
# or image of PIL.Image.Image type
```

**Note**: Comparison is only meaningful for images generated by the **same prompt**.


### Benchmark Reproduction

We also provide [images](https://huggingface.co/datasets/zhwang/HPDv2/tree/main/benchmark/benchmark_imgs) generated by models in our [benchmark](#the-hps-v2-benchmark) used for evaluation. You can easily download the data and evaluate the models by running the following code.

```python
import hpsv2

print(hpsv2.available_models) # Get models that have access to data
hpsv2.evaluate_benchmark('<model_name>')
```

### Custom Evaluation

To evaluate your own text-to-image generative model, you can prepare the images for evaluation base on the [benchmark prompts](https://huggingface.co/datasets/zhwang/HPDv2/tree/main/benchmark) we provide by running the following code:

```python
import os
import hpsv2

# Get benchmark prompts (<style> = all, anime, concept-art, paintings, photo)
all_prompts = hpsv2.benchmark_prompts('all') 

# Iterate over the benchmark prompts to generate images
for style, prompts in all_prompts.items():
    for prompt in prompts:
        image = TextToImageModel(prompt) 
        # TextToImageModel is the model you want to evaluate
        image.save(os.path.join("<image_path>", style, "<image_name>")) 
        # <image_path> is the folder path to store generated images, as the input of hpsv2.evaluate().
        # <image_name> is of the form of '00xxx.jpg', with 'xxx' ranging from '000' to '799' corresponding to each prompt.

```

And then run the following code to conduct evaluation:

```python
import hpsv2

hpsv2.evaluate_("<images_path>") 
# <image_path> is the same as <image_path> in the prevoius part
```

### Preference Model Evaluation

Evaluating HPS v2's correlation with human preference choices:
|  Model | Acc. on ImageReward test set (%)| Acc. on HPD v2 test set (%)
| :-----: | :-----: |:-----: |
|  [Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) | 57.4 | 76.8 |
|  [ImageReward](https://github.com/THUDM/ImageReward) | 65.1 | 74.0 |
|  [HPS](https://github.com/tgxs002/align_sd) | 61.2 | 77.6 |
|  [PickScore](https://github.com/yuvalkirstain/PickScore) | 62.9 | 79.8 |
|  Single Human | 65.3 | 78.1 |
|  HPS v2 | 65.7 | 83.3 |

HPS v2 checkpoint can be downloaded from [here](https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt). The model and live demo is also hosted on 🤗 Hugging Face at [here](https://huggingface.co/spaces/xswu/HPSv2).

Run the following commands to evaluate the HPS v2 model on HPD v2 test set and ImageReward test set **(Need to install the package `hpsv2` first)**:
```shell
# evaluate on HPD v2 test set
python evaluate.py --data-type test --data-path /path/to/HPD --image-path /path/to/image_folder

# evaluate on ImageReward test set
python evaluate.py --data-type ImageReward --data-path /path/to/IR --image-path /path/to/image_folder
```

## Human Preference Dataset v2
The prompts in our dataset are sourced from DiffusionDB and MSCOCO Captions. Prompts from DiffusionDB are first cleaned by ChatGPT to remove biased function words. Human annotators are tasked to rank images generated by different text-to-image generative models from the same prompt. Totally there are about 798k pairwise comparisons of images for over 430k images and 107k prompts, 645k pairs for training split and 153k pairs for test split.

Image sources of HPD v2:
|  Source | # of images 
| :-----: | :-----: |
| CogView2 | 73697 |
| DALL·E 2 | 101869 | 
| GLIDE (mini) | 400 |
| Stable Diffusion v1.4 | 101869 |
| Stable Diffusion v2.0 | 101869 | 
| LAFITE | 400 | 
| VQ-GAN+CLIP | 400 |
| VQ-Diffusion | 400 |
| FuseDream | 400 |
| COCO Captions | 28272 |

Currently, the test data can be downloaded from [here](https://huggingface.co/datasets/zhwang/HPDv2). The training dataset will be **released soon**.
Once unzipped, you should get a folder with the following structure:
```
HPD
---- train/
-------- {image_id}.jpg
---- test/
-------- {image_id}.jpg
---- train.json
---- test.json
---- benchmark/
-------- benchmark_imgs/
------------ {model_id}/
---------------- {image_id}.jpg
-------- drawbench/
------------ {model_id}/
---------------- {image_id}.jpg
-------- anime.json
-------- concept-art.json
-------- paintings.json
-------- photo.json
-------- drawbench.json
```

The annotation file, `train.json`, is organized as:
```
[
    {
        'human_preference': list[int], # 1 for preference
        'prompt': str,
        'file_path': list[str],
        'user_hash': str,
    },
    ...
]
```

The annotation file, `test.json`, is organized as:
```
[
    {
        'prompt': str,
        'image_path': list[str],
        'rank': list[int], # averaged ranking result for image at the same index in image_path,
        'raw_annotation': list[{'rank', 'user_hash'}]  # raw ranking result from each annotator
    },
    ...
]
```

The benchmark prompts file, ie. `anime.json` is pure prompts. The corresponding image can be found in the folder of the corresponding model by indexing the prompt.

## Command Line Interface

### Evaluating Text-to-image Generative Models using HPS v2
The generated images in our experiments can be downloaded from [here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155172150_link_cuhk_edu_hk/EVnjOngvDO1MhIp7hVr8GXgBmxVDcSk7s9Xuu9srO4YLbA?e=8PqYud). 

The following script reproduces the [benchmark table](#the-hps-v2-benchmark) and our results on DrawBench (reported in the paper) **(Need to install the package `hpsv2` first)**:
```shell
# HPS v2 benchmark (for more than one models)
python evaluate.py --data-type benchmark_all --data-path /path/to/HPD/benchmark --image-path /path/to/benchmark_imgs

# HPS v2 benchmark (for only one models)
python evaluate.py --data-type benchmark --data-path /path/to/HPD/benchmark --image-path /path/to/benchmark_imgs/${model_name}

# DrawBench
python evaluate.py --data-type drawbench --data-path /path/to/HPD/benchmark --image-path /path/to/drawbench_imgs
```

### Scoring Single Generated Image and Corresponding Prompt

We provide one example image in the `asset/images` directory of this repo. The corresponding prompt is `"A cat with two horns on its head"`.

Run the following commands to score the single generated image and the corresponding prompt **(Need to install the package `hpsv2` first)**:
```shell
python score.py --image-path assets/demo_image.jpg --prompt 'A cat with two horns on its head'
```
where the parameter `image` can accept multiple values.

## Train Human Preference Predictor
To train your own human preference predictor, just change the corresponding path in `configs/controller.sh` and run the following command:
```shell
# if you are running locally
bash configs/HPSv2.sh train 8 local
# if you are running on slurm
bash configs/HPSv2.sh train 8 ${quota_type}
```

## BibTeX
```
@article{wu2023human,
  title={Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis},
  author={Wu, Xiaoshi and Hao, Yiming and Sun, Keqiang and Chen, Yixiong and Zhu, Feng and Zhao, Rui and Li, Hongsheng},
  journal={arXiv preprint arXiv:2306.09341},
  year={2023}
}
```

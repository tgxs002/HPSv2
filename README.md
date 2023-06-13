## Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis

This is the official repository for the paper: Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis. 

## Overview 
We provide Human Preference Dataset v2, a large-scale, well-annotated dataset for researches of human preferences on images generated by text-to-image generative models. Based on HPDv2 we trained HPS v2, a better human preference prediction model against existing ones. We also provide a fair, stable and easy-to-use set of evaluation prompts for text-to-image generative models.

## Human Preference Dataset v2
The prompts in our dataset are sourced from DiffusionDB and MSCOCO Captions. Prompts from DiffusionDB are first cleaned by ChatGPT to remove biased function words. Human annotators are tasked to rank images generated by different text-to-image generative models from the same prompt. Totally there are about 798K pairwise comparisons of images for over 410k images and 107k prompts, 645k pairs for training split and 153k pairs for test split.

The compressed dataset can be downloaded from [here](??????????).
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
        'rank': list[int], # ranking for image at the same index in image_path
    },
    ...
]
```

The benchmark prompts file, ie. `anime.json` is pure prompts. The corresponding image can be found in the folder of the corresponding model by indexing the prompt.

### environments

```
# environments
pip install -r requirements.txt 
```

## Evaluation
HPSv2 checkpoint can be downloaded from [here](???????????????)
Run the following command for evaluating the HPSv2 model:
```
python evaluate.py --data-type test --data-path /path/to/HPD --image-path /path/to/image_folder --batch-size 10 --checkpoint /path/to/HPSv2.pt

#for example
python evaluate.py --data-type test --data-path data/HPD --image-path data/HPD/test --batch-size 10 --checkpoint ckpt/HPSv2.pt

```
## Train Human Preference Predictor
To train your own human preference predictor, you can just change the coresponding path in `configs/controller.sh` and run the following command:
```
# if you are running locally
bash configs/HPSv2.sh train ${GPUS} local
# if you are running on slurm
bash configs/HPSv2.sh train ${GPUS} ${quota_type}
```

## Citation
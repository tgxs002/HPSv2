from cProfile import label
import os
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
import requests
from clint.textui import progress
import huggingface_hub

import torch
from torch.utils.data import Dataset, DataLoader

from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from hpsv2.src.training.train import calc_ImageReward, inversion_score
from hpsv2.src.training.data import ImageRewardDataset, collate_rank, RankingDataset
from hpsv2.utils import root_path, hps_version_map


class BenchmarkDataset(Dataset):
    def __init__(self, meta_file, image_folder,transforms, tokenizer):
        self.transforms = transforms
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.open_image = Image.open
        with open(meta_file, 'r') as f:
            prompts = json.load(f)
        used_prompts = []
        files = []
        for idx, prompt in enumerate(prompts):
            filename = os.path.join(self.image_folder, f'{idx:05d}.jpg')
            if os.path.exists(filename):
                used_prompts.append(prompt)
                files.append(filename)
            else:
                print(f"missing image for prompt: {prompt}")
        self.prompts = used_prompts
        self.files = files
            
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        images = self.transforms(self.open_image(img_path))
        caption = self.tokenizer(self.prompts[idx])
        return images, caption

def evaluate_IR(data_path, image_folder, model, batch_size, preprocess_val, tokenizer, device):
    meta_file = data_path + '/ImageReward_test.json'
    dataset = ImageRewardDataset(meta_file, image_folder, preprocess_val, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_rank)
    
    score = 0
    total = len(dataset)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, num_images, labels, texts = batch
            images = images.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            num_images = num_images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(images, texts)
                image_features, text_features, logit_scale = outputs["image_features"], outputs["text_features"], outputs["logit_scale"]
                logits_per_image = logit_scale * image_features @ text_features.T * 100
                paired_logits_list = [logit[:,i] for i, logit in enumerate(logits_per_image.split(num_images.tolist()))]

            predicted = [torch.argsort(-k) for k in paired_logits_list]
            hps_ranking = [[predicted[i].tolist().index(j) for j in range(n)] for i,n in enumerate(num_images)]
            labels = [label for label in labels.split(num_images.tolist())]
            score +=sum([calc_ImageReward(paired_logits_list[i].tolist(), labels[i]) for i in range(len(hps_ranking))])
    print('ImageReward:', score/total)

def evaluate_rank(data_path, image_folder, model, batch_size, preprocess_val, tokenizer, device):
    meta_file = data_path + '/test.json'
    dataset = RankingDataset(meta_file, image_folder, preprocess_val, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_rank)
    
    score = 0
    total = len(dataset)
    all_rankings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images, num_images, labels, texts = batch
            images = images.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            num_images = num_images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(images, texts)
                image_features, text_features, logit_scale = outputs["image_features"], outputs["text_features"], outputs["logit_scale"]
                logits_per_image = logit_scale * image_features @ text_features.T
                paired_logits_list = [logit[:,i] for i, logit in enumerate(logits_per_image.split(num_images.tolist()))]

            predicted = [torch.argsort(-k) for k in paired_logits_list]
            hps_ranking = [[predicted[i].tolist().index(j) for j in range(n)] for i,n in enumerate(num_images)]
            labels = [label for label in labels.split(num_images.tolist())]
            all_rankings.extend(hps_ranking)
            score += sum([inversion_score(hps_ranking[i], labels[i]) for i in range(len(hps_ranking))])
    print('ranking_acc:', score/total)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    with open('logs/hps_rank.json', 'w') as f:
        json.dump(all_rankings, f)

def collate_eval(batch):
    images = torch.stack([sample[0] for sample in batch])
    captions = torch.cat([sample[1] for sample in batch])
    return images, captions

def evaluate_benchmark(data_path, img_path, model, batch_size, preprocess_val, tokenizer, device):
    meta_dir = data_path
    style_list = os.listdir(img_path)
    model_id = img_path.split('/')[-1]

    score = {}
    
    score[model_id]={}
    for style in style_list:
        # score[model_id][style] = [0] * 10
        score[model_id][style] = []
        image_folder = os.path.join(img_path, style)
        meta_file = os.path.join(meta_dir, f'{style}.json')
        dataset = BenchmarkDataset(meta_file, image_folder, preprocess_val, tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_eval)

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = model(images, texts)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T * 100
                # score[model_id][style][i] = torch.sum(torch.diagonal(logits_per_image)).cpu().item() / 80
                score[model_id][style].extend(torch.diagonal(logits_per_image).cpu().tolist())
    print('-----------benchmark score ---------------- ')
    for model_id, data in score.items():
        all_score = []
        for style , res in data.items():
            avg_score = [np.mean(res[i:i+80]) for i in range(0, len(res), 80)]
            all_score.extend(res)
            print(model_id, '{:<15}'.format(style), '{:.2f}'.format(np.mean(avg_score)), '\t', '{:.4f}'.format(np.std(avg_score)))
        print(model_id, '{:<15}'.format('Average'), '{:.2f}'.format(np.mean(all_score)), '\t')

def evaluate_benchmark_all(data_path, root_dir, model, batch_size, preprocess_val, tokenizer, device):
    meta_dir = data_path
    model_list = os.listdir(root_dir)
    style_list = os.listdir(os.path.join(root_dir, model_list[0]))

    score = {}
    for model_id in tqdm(model_list):
        score[model_id]={}
        for style in style_list:
            # score[model_id][style] = [0] * 10
            score[model_id][style] = []
            image_folder = os.path.join(root_dir, model_id, style)
            meta_file = os.path.join(meta_dir, f'{style}.json')
            dataset = BenchmarkDataset(meta_file, image_folder, preprocess_val, tokenizer)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_eval)

            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    images, texts = batch
                    images = images.to(device=device, non_blocking=True)
                    texts = texts.to(device=device, non_blocking=True)

                    with torch.cuda.amp.autocast():
                        outputs = model(images, texts)
                        image_features, text_features = outputs["image_features"], outputs["text_features"]
                        logits_per_image = image_features @ text_features.T * 100
                    # score[model_id][style][i] = torch.sum(torch.diagonal(logits_per_image)).cpu().item() / 80
                    score[model_id][style].extend(torch.diagonal(logits_per_image).cpu().tolist())
    print('-----------benchmark score ---------------- ')
    for model_id, data in score.items():
        all_score = []
        for style , res in data.items():
            avg_score = [np.mean(res[i:i+80]) for i in range(0, len(res), 80)]
            all_score.extend(res)
            print(model_id, '{:<15}'.format(style), '{:.2f}'.format(np.mean(avg_score)), '\t', '{:.4f}'.format(np.std(avg_score)))
        print(model_id, '{:<15}'.format('Average'), '{:.2f}'.format(np.mean(all_score)), '\t')
        
def evaluate_benchmark_DB(data_path, root_dir, model, batch_size, preprocess_val, tokenizer, device):
    meta_file = data_path + '/drawbench.json'
    model_list = os.listdir(root_dir)
    

    score = {}
    for model_id in model_list:
        image_folder = os.path.join(root_dir, model_id)
        dataset = BenchmarkDataset(meta_file, image_folder, preprocess_val, tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_eval)
        score[model_id] = 0
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images, texts = batch
                images = images.to(device=device, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = model(images, texts)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T * 100
                    diag = torch.diagonal(logits_per_image)
                score[model_id] += torch.sum(diag).cpu().item()
            score[model_id] = score[model_id] / len(dataset)
    # with open('logs/benchmark_score_DB.json', 'w') as f:
    #     json.dump(score, f)
    print('-----------drawbench score ---------------- ')
    for model, data in score.items():
        print(model, '\t', '\t', np.mean(data))

model_dict = {}
model_name = "ViT-H-14"
precision = 'amp'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def initialize_model():
    if not model_dict:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            model_name,
            None,
            precision=precision,
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        model_dict['model'] = model
        model_dict['preprocess_val'] = preprocess_val

        
def evaluate(mode: str, root_dir: str, data_path: str = os.path.join(root_path,'datasets/benchmark'), checkpoint_path: str = None, batch_size: int = 20, hps_version: str = "v2.1") -> None:
    
    # check if the default checkpoint exists
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    
    if checkpoint_path is None:
        checkpoint_path = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
    
    initialize_model()
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']

    print('Loading model ...')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device)
    model.eval()
    print('Loading model successfully!')
    
    
    if mode == 'ImageReward':
        evaluate_IR(data_path, root_dir, model, batch_size, preprocess_val, tokenizer, device)
    elif mode == 'test':
        evaluate_rank(data_path, root_dir, model, batch_size, preprocess_val, tokenizer, device)
    elif mode == 'benchmark_all':
        evaluate_benchmark_all(data_path, root_dir, model, batch_size, preprocess_val, tokenizer, device)
    elif mode == 'benchmark':
        evaluate_benchmark(data_path, root_dir, model, batch_size, preprocess_val, tokenizer, device)
    elif mode == 'drawbench':
        evaluate_benchmark_DB(data_path, root_dir, model, batch_size, preprocess_val, tokenizer, device)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument('--data-type', type=str, required=True, choices=['benchmark', 'benchmark_all', 'test', 'ImageReward', 'drawbench'])
    parser.add_argument('--data-path', type=str, required=True, help='path to dataset')
    parser.add_argument('--image-path', type=str, required=True, help='path to image files')
    parser.add_argument('--checkpoint', type=str, default=os.path.join(root_path,'HPS_v2_compressed.pt'), help='path to checkpoint')
    parser.add_argument('--batch-size', type=int, default=20)
    args = parser.parse_args()
    
    evaluate(mode=args.data_type, data_path=args.data_path, root_dir=args.image_path, checkpoint_path=args.checkpoint, batch_size=args.batch_size)

    
    
    

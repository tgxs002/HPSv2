from cProfile import label
import os
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from src.open_clip import create_model_and_transforms, get_tokenizer
from src.training.train import calc_ImageReward, inversion_score
from src.training.data import ImageRewardDataset, collate_rank, RankingDataset


parser = ArgumentParser()
parser.add_argument('--data-type', type=str, choices=['benchmark', 'test', 'ImageReward', 'drawbench'])
parser.add_argument('--data-path', type=str, help='path to dataset')
parser.add_argument('--image-path', type=str, help='path to image files')
parser.add_argument('--checkpoint', type=str, default='logs/ranking_only/top_2.pt')
parser.add_argument('--batch-size', type=int, default=20)
args = parser.parse_args()

batch_size = args.batch_size
args.model = "ViT-H-14"
args.precision = 'amp'
print(args.model)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess_train, preprocess_val = create_model_and_transforms(
    args.model,
    'laion2B-s32B-b79K',
    precision=args.precision,
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

checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['state_dict'])
tokenizer = get_tokenizer(args.model)
model.eval()

class BenchmarkDataset(Dataset):
    def __init__(self, meta_file, image_folder,transforms, tokenizer):
        self.transforms = transforms
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.open_image = Image.open
        with open(meta_file, 'r') as f:
            self.annotations = json.load(f)
            
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.image_folder, f'{idx:05d}.jpg')
            images = self.transforms(self.open_image(os.path.join(img_path)))
            caption = self.tokenizer(self.annotations[idx])
            return images, caption
        except:
            print('file not exist')
            return self.__getitem__((idx + 1) % len(self))

def evaluate_IR(data_path, image_folder, model):
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
                logits_per_image = logit_scale * image_features @ text_features.T
                paired_logits_list = [logit[:,i] for i, logit in enumerate(logits_per_image.split(num_images.tolist()))]

            predicted = [torch.argsort(-k) for k in paired_logits_list]
            hps_ranking = [[predicted[i].tolist().index(j) for j in range(n)] for i,n in enumerate(num_images)]
            labels = [label for label in labels.split(num_images.tolist())]
            score +=sum([calc_ImageReward(paired_logits_list[i].tolist(), labels[i]) for i in range(len(hps_ranking))])
    print('ImageReward:', score/total)

def evaluate_rank(data_path, image_folder, model):
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
    with open('logs/hps_rank.json', 'w') as f:
        json.dump(all_rankings, f)

def collate_eval(batch):
    images = torch.stack([sample[0] for sample in batch])
    captions = torch.cat([sample[1] for sample in batch])
    return images, captions


def evaluate_benchmark(data_path, root_dir, model):
    meta_dir = data_path
    model_list = os.listdir(root_dir)
    style_list = os.listdir(os.path.join(root_dir, model_list[0]))

    score = {}
    for model_id in model_list:
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
                        logits_per_image = image_features @ text_features.T
                    # score[model_id][style][i] = torch.sum(torch.diagonal(logits_per_image)).cpu().item() / 80
                    score[model_id][style].extend(torch.diagonal(logits_per_image).cpu().tolist())
    print('-----------benchmark score ---------------- ')
    for model_id, data in score.items():
        for style , res in data.items():
            avg_score = [np.mean(res[i:i+80]) for i in range(0, 800, 80)]
            print(model_id, '\t', style, '\t', np.mean(avg_score), '\t', np.std(avg_score))


def evaluate_benchmark_DB(data_path, root_dir, model):
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
                    logits_per_image = image_features @ text_features.T
                    diag = torch.diagonal(logits_per_image)
                score[model_id] += torch.sum(diag).cpu().item()
            score[model_id] = score[model_id] / len(dataset)
    # with open('logs/benchmark_score_DB.json', 'w') as f:
    #     json.dump(score, f)
    print('-----------drawbench score ---------------- ')
    for model, data in score.items():
        print(model, '\t', '\t', np.mean(data))


if args.data_type == 'ImageReward':
    evaluate_IR(args.data_path, args.image_path, model)
elif args.data_type == 'test':
    evaluate_rank(args.data_path, args.image_path, model)
elif args.data_type == 'benchmark':
    evaluate_benchmark(args.data_path, args.image_path, model)
elif args.data_type == 'drawbench':
    evaluate_benchmark_DB(args.data_path, args.image_path, model)
else:
    raise NotImplementedError





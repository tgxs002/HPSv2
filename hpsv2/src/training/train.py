import hashlib
import itertools
import json
import logging
import math
import random
import os
import tempfile
import time
import einops
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

from .data import ImageRewardDataset, RankingDataset

from ..open_clip import get_cast_dtype, CLIP, CustomTextCLIP
from .distributed import is_master, barrier
from .zero_shot import zero_shot_eval
from .precision import get_autocast
from ..open_clip.loss import PreferenceLoss, RankingLoss, HPSLoss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

def random_sampling_iterator(iterators, sampling_ratios, data_types, num_iters):
    iterators = [iter(iterator) for iterator in iterators]
    num_iterators = len(iterators)
    loop_counter = 0

    while loop_counter < num_iters:
        current_state = random.getstate()
        random.seed(loop_counter)
        iterator_idx = random.choices(range(num_iterators), sampling_ratios)[0]
        random.setstate(current_state)
        yield next(iterators[iterator_idx]), data_types[iterator_idx]
        loop_counter += 1


def train_iters(model, data, iterations, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()
    
    ce_loss = PreferenceLoss()
    mse_loss = torch.nn.MSELoss()
    rk_loss = RankingLoss()
    hps_loss = HPSLoss()
    if args.distill:
        dist_model.eval()

    for train_set in data['train']:
        train_set.set_epoch(0)  # set epoch in process safe manner via sampler or shared_epoch
    data_types = [d.data_type for d in data['train']]
    train_data_sample_ratios = [sample_ratio for sample_ratio, ignore in zip(args.train_data_sample_ratio, args.ignore_in_train) if not ignore]
    dataloader = random_sampling_iterator([dataset.dataloader for dataset in data['train']], train_data_sample_ratios, data_types, iterations)


    sample_digits = math.ceil(math.log(sum([dataset.dataloader.num_samples for dataset in data['train']]) + 1, 10))
    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for step, (batch, data_type) in enumerate(dataloader):
        # TODO: currently only test on accum_freq==1

        if not args.skip_scheduler:
            scheduler(step)

        if data_type == 'preference':
            images, num_images, labels, texts = batch
            texts = texts.to(device=device, non_blocking=True)
        elif data_type == 'rating':
            images, labels = batch
        elif data_type == 'regional':
            images, labels = batch
        elif data_type == 'ranking':
            images, num_images, labels, texts = batch
            texts = texts.to(device=device, non_blocking=True)
        elif data_type == 'HPD':
            images, labels, texts = batch
            # num_per_prompts = num_per_prompts.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        if args.accum_freq == 1:
            with autocast():

                if data_type == 'rating' or args.no_text_condition:
                    image_features = unwrap_model(model).visual(images)
                    scores = unwrap_model(model).score_predictor(image_features)
                    if args.no_text_condition:  
                        paired_logits_list = [logit[:,0] for i, logit in enumerate(scores.split(num_images.tolist()))]
                        paired_logits = pad_sequence(paired_logits_list, batch_first=True, padding_value=-999)
                        total_loss = F.cross_entropy(paired_logits, labels)
                    else:
                        total_loss = mse_loss(scores.squeeze(), labels.to(scores.dtype))
                elif data_type == 'preference' :
                    output = model(images, texts)
                    image_features, text_features, logit_scale = output["image_features"], output["text_features"], output["logit_scale"]
                    # total_loss = loss(image_features, text_features, logit_scale)
                    logits_per_image = logit_scale * image_features @ text_features.T
                    total_loss = ce_loss(logits_per_image, num_images, labels)
                elif data_type == 'HPD':
                    output = model(images, texts)
                    image_features, text_features, logit_scale = output["image_features"], output["text_features"], output["logit_scale"]
                    logits_per_text = logit_scale * text_features @ image_features.T
                    total_loss = hps_loss(logits_per_text, labels)
                elif data_type == 'ranking':
                    output = model(images, texts)
                    image_features, text_features, logit_scale = output["image_features"], output["text_features"], output["logit_scale"]
                    # logits_per_image = logit_scale * image_features @ text_features.T
                    score =  logit_scale * image_features @ text_features.T
                    total_loss = rk_loss(score, num_images, labels, args.margin)

                elif data_type == 'regional':
                    # logit_scale = model.logit_scale
                    feature_map = unwrap_model(model).visual(images, skip_pool=True)[:, 1:]
                    logits = unwrap_model(model).region_predictor(feature_map)
                    wh = int(math.sqrt(feature_map.size(1)))
                    ps = images.size(2) // wh
                    logits = logits.unflatten(1, (wh, wh))[:,:,:,0]
                    # downsample the labels to match the feature map size
                    patches = einops.reduce(labels, 'b (h s1) (w s2) -> b h w', 'mean', s1=ps, s2=ps)
                    patches = (patches > 0).float()
                    total_loss = mse_loss(logits.sigmoid(), patches.to(patches.dtype))

            backward(total_loss, scaler)
            losses = dict(total_loss=total_loss)

            if scaler is not None:
                if args.horovod:
                    optimizer.synchronize()
                    scaler.unscale_(optimizer)
                    if args.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    with optimizer.skip_synchronize():
                        scaler.step(optimizer)
                else:
                    if args.grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                    scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = step + 1
        if is_master(args) and (step % args.log_every_n_steps == 0 or batch_count == iterations):
            batch_size = len(images)
            num_samples = batch_count * args.accum_freq 
            percent_complete = 100.0 * batch_count / iterations

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = unwrap_model(model).logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq / batch_time_m.val
            logging.info(
                f"Train iterations: [{num_samples:>{sample_digits}}/{iterations} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()

def evaluate_preference(model, data, args):
    model = unwrap_model(model)
    model.eval()
    dataloader = data.dataloader
    samples_per_val = dataloader.num_samples

    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    total = 0
    correct = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % args.world_size != args.rank:
                continue
            images, num_images, labels, texts = batch
            images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            with autocast():

                if args.no_text_condition:
                    image_features = model.visual(images)
                    logit_scale = model.logit_scale
                    scores = model.score_predictor(image_features)
                    paired_logits_list = [logit[:,0] for i, logit in enumerate(scores.split(num_images.tolist()))]
                else:
                    outputs = model(images, texts)
                    image_features, text_features, logit_scale = outputs["image_features"], outputs["text_features"], outputs["logit_scale"]
                    logits_per_image = logit_scale * image_features @ text_features.T
                    paired_logits_list = [logit[:,i] for i, logit in enumerate(logits_per_image.split(num_images.tolist()))]
                predicted = torch.tensor([k.argmax().item() for k in paired_logits_list])
                correct += (predicted == labels).int().sum().item()
                total += predicted.numel()
    
    # write to a temp file
    file_name = hashlib.md5(str(args.name).encode()).hexdigest()
    with open(f"{file_name}_{args.rank}.json", "w") as f:
        json.dump(dict(
            correct=correct,
            total=total,
        ), f)
    time.sleep(0.1)
        
    barrier(args)

    correct = 0
    total = 0
    if is_master(args):
        for i in range(args.world_size):
            with open(f"{file_name}_{i}.json", "r") as f:
                data = json.load(f)
                correct += data["correct"]
                total += data["total"]
            os.remove(f"{file_name}_{i}.json")

        logging.info(
            f"Final Acc: {correct / total:.4f}\t")
        
    return correct / (total + 1e-6)
            
def evaluate_regional(model, data, args):
    dataloader = data.dataloader
    samples_per_val = dataloader.num_samples

    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)
    num_samples = len(dataloader)
    threshold = 0.5
    with torch.no_grad():
        score = 0
        total = 0
        for i, batch in enumerate(dataloader):
            images, labels = batch
            images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)

            with autocast():
                feature_map = model.visual(images, skip_pool=True)[:, 1:]
                logits = model.region_predictor(feature_map)
                wh = int(math.sqrt(feature_map.size(1)))
                ps = images.size(2) // wh
                logits = logits.unflatten(1, (wh, wh))[:,:,:,0]
                # downsample the labels to match the feature map size
                patches = einops.reduce(labels, 'b (h s1) (w s2) -> b h w', 'mean', s1=ps, s2=ps)
                patches = (patches > 0).float()
                pred_mask = (logits.sigmoid() > threshold).float()
                #calc IOU
                intersection = (pred_mask * patches).sum()
                union = pred_mask.sum() + patches.sum() - intersection
                iou_score = intersection / union
            score += iou_score
            total += 1

            if is_master(args) and (i % 100) == 0:
                logging.info(
                    # f"[{i} / {samples_per_val}]\t"
                    f"[{i} / {len(dataloader)}]\t"
                    f"Current IoU: {score / (total + 0.001):.4f}\t")

    if is_master(args):
        logging.info(
            f"Final IoU: {score / (total + 0.001):.4f}\t")

    return score / (total + 0.001)

def inversion_score(p1, p2):
    assert len(p1) == len(p2), f'{len(p1)}, {len(p2)}'
    n = len(p1)
    cnt = 0
    for i in range(n-1):
        for j in range(i+1, n):
            if p1[i] > p1[j] and p2[i] < p2[j]:
                cnt += 1
            elif p1[i] < p1[j] and p2[i] > p2[j]:
                cnt += 1
    return 1 - cnt / (n * (n - 1) / 2)

def model_pair_score(score:dict, p1, p2, num_image):
    model_pairs = set()
    for i in range(num_image):
        if i not in score.keys():
            score[i] = {}

        for j in range(num_image):
            if j not in score[i].keys():
                score[i][j] = 0
            if j == i or (i, j) in model_pairs or (j, i) in model_pairs:
                continue
            model_pairs.add((i,j))
            if (p1[i] - p1[j]) * (p2[i] - p2[j]) > 0:
                score[i][j] += 1
    return score

def all_gather(tensor):
    world_size = torch.distributed.get_world_size()
    tensor_list = [torch.ones_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor, async_op=False)
    return torch.cat(tensor_list, dim=0)

def evaluate_ranking(model, data, args):
    model = unwrap_model(model)
    model.eval()
    dataloader = data.dataloader
    samples_per_val = dataloader.num_samples

    device = torch.device(args.device)
    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    score = 0 
    # pair_score = {}
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % args.world_size != args.local_rank:
                continue

            images, num_images, labels, texts = batch
            images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            num_images = num_images.to(device=device, non_blocking=True)
            labels = labels.to(device=device, non_blocking=True)
            
            with autocast():
                if args.no_text_condition:
                    image_features = model.visual(images)
                    logit_scale = model.logit_scale
                    scores = model.score_predictor(image_features)
                    paired_logits_list = [logit[:,0] for i, logit in enumerate(scores.split(num_images.tolist()))]
                else:
                    outputs = model(images, texts)
                    image_features, text_features, logit_scale = outputs["image_features"], outputs["text_features"], outputs["logit_scale"]
                    logits_per_image = logit_scale * image_features @ text_features.T
                    paired_logits_list = [logit[:,i] for i, logit in enumerate(logits_per_image.split(num_images.tolist()))]
                predicted = [torch.argsort(-k) for k in paired_logits_list]
                hps_ranking = [[predicted[i].tolist().index(j) for j in range(n)] for i,n in enumerate(num_images)]
                labels = [label for label in labels.split(num_images.tolist())] 
                if isinstance(dataloader.dataset, RankingDataset):
                    score += sum([inversion_score(hps_ranking[i], labels[i]) for i in range(len(hps_ranking))])
                elif isinstance(dataloader.dataset, ImageRewardDataset):
                    score +=sum([calc_ImageReward(paired_logits_list[i].tolist(), labels[i]) for i in range(len(hps_ranking))])

    # write score to a tempfile, file name is a hash string
    file_name = hashlib.md5(str(args.name).encode()).hexdigest()
    with open(f"{file_name}_{args.rank}.tmp", "w") as f:
        f.write(str(score))
    time.sleep(0.1)
        
    barrier(args)

    score = 0
    if is_master(args):
        for i in range(args.world_size):
            with open(f"{file_name}_{i}.tmp", "r") as f:
                score += float(f.read())
            os.remove(f"{file_name}_{i}.tmp")

        score = score / samples_per_val

        logging.info(
            f"Final Acc: {score:.4f}\t")
        # return score, pair_score
    return score

def calc_ImageReward( pred, gt):
    # using inversion score calculate method in ImageReward
    # There's some little difference because ImageReward benchmark has tie rankings
    tol_cnt = 0.
    true_cnt = 0.
    for idx in range(len(gt)):
        item_base = gt
        item = pred
        for i in range(len(item_base)):
            for j in range(i+1, len(item_base)):
                if item_base[i] > item_base[j]:
                    if item[i] >= item[j]:
                        tol_cnt += 1
                    elif item[i] < item[j]:
                        tol_cnt += 1
                        true_cnt += 1
                elif item_base[i] < item_base[j]:
                    if item[i] > item[j]:
                        tol_cnt += 1
                        true_cnt += 1
                    elif item[i] <= item[j]:
                        tol_cnt += 1
    
    return true_cnt / tol_cnt

def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)

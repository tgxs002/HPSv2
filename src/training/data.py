from __future__ import annotations
import ast
import copy
from curses import meta
from email.mime import image
import json
import logging
import math
import os
import random
import sys
import time
import io
import itertools
import braceexpand
from dataclasses import dataclass
from multiprocessing import Value
import pyarrow as pa

import numpy as np
import pandas as pd
import functools
import torch
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF
import torch.distributed as dist
import webdataset as wds
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler, Sampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

from ..open_clip import transform

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

try:
    from petrel_client.client import Client
except ImportError as E:
    "petrel_client.client cannot be imported"
    pass

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    return Image.open(buff).convert("RGB")

@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD

def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if dist.get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output

def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
        If workers need a shared RNG, they can use this shared seed to
        create one.

    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2**31)
    all_ints = all_gather(ints)
    return all_ints[0]

class TrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """
    
    def __init__(self, dataset, num_replicas=None, rank=None, local_rank=None, local_size=None, shuffle=True, seed = None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas)) -1 
        self.total_size = len(dataset)
        self.shuffle = shuffle
        # self.dataset_repeat = dataset_repeat
        if seed is None:
            seed = shared_random_seed()
        self.seed = int(seed)

    def __len__(self):
        return self.num_samples 
    
    def __iter__(self):
        start = self.rank
        yield from itertools.islice(self._infinite_indices(), start, None, self.num_replicas)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.total_size, generator=g).tolist()
            else:
                yield from torch.arange(self.total_size).tolist()

class TCSLoader(object):

    def __init__(self, time_limit=3):
        conf_path = os.environ.get('CEPH_CONFIG', './petreloss.config')
        self.client = Client(conf_path)
        self.time_limit = time_limit

    def __call__(self, fn):
        try:
            img_value_str = self.client.get(fn)
            img = pil_loader(img_value_str)
            return img
        except Exception as e:
            print('Read image failed ({})'.format(fn))
            raise e
        

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    data_type: str
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist), f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        if is_train:
            data_path = args.imagenet_train
            preprocess_fn = preprocess_train
        else:
            data_path = args.imagenet_val
            preprocess_fn = preprocess_val
        assert data_path

        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr

        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=sampler,
    )

    return DataInfo(dataloader=dataloader, sampler=sampler, data_type='classification')


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        weights=None,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights), f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic
            if self.worker_seed is None:
                # pytorch worker seed should be deterministic due to being init by arg.seed + rank + worker id
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    
    if resampled:
        pipeline = [ResampledShards2(input_shards, weights=args.train_data_upsampling_factors, deterministic=True, epoch=shared_epoch)]
    else:
        assert args.train_data_upsampling_factors is None, "--train_data_upsampling_factors is only supported when sampling with replacement (together with --dataset-resampled)."
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch, data_type='image-text')


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and not args.distributed and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader=dataloader, sampler=sampler, data_type='image-text')


class SyntheticDataset(Dataset):

    def __init__(self, transform=None, image_size=(224, 224), caption="Dummy caption", dataset_size=100, tokenizer=None):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and not args.distributed and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader=dataloader, sampler=sampler, data_type='image-text')

class PreferenceDataset(Dataset):
    def __init__(self, meta_file, image_folder, transforms, tokenizer, extra_data=(None, None)):
        extra_meta, extra_folder = extra_data
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.open_image = Image.open
        if image_folder.startswith('s3://'):
            loader = TCSLoader()
            self.open_image = loader
        if meta_file is not None:
            with open(meta_file, 'r') as f:
                self.table = pa.Table.from_pylist(json.load(f))
            self.image_folder = image_folder
        else:
            # self.captions = pa.array()
            self.table = []
        if extra_meta:
            with open(extra_meta, 'r') as f:
                meta = json.load(f)
            self.files = [t['files'] for t in meta]
            self.extra_captions = [t['caption'] for t in meta]
            self.extra_label = [t['human_preference'] for t in meta]
            self.extra_image_folder = extra_folder
        else:
            self.extra_captions = []
        
    def __len__(self):
        return len(self.table) + len(self.extra_captions)

    def __getitem__(self, idx):
        try:
            if idx < len(self.table):
                images = [self.transforms(self.open_image(os.path.join(self.image_folder, file_names))) for file_names in self.table.column('file_path')[idx].as_py()]
                if not len(set([i.size() for i in images])) == 1:
                    return self.__getitem__((idx + 1) % len(self))
                label = self.table.column('pap_pref')[idx].as_py()
                caption = self.tokenizer(self.table.column('prompt')[idx].as_py())
            else:
                idx = idx - len(self.captions)
                images = [self.transforms(self.open_image(os.path.join(self.extra_image_folder, f))) for f in self.files[idx]]
                label = self.extra_label[idx]
                caption = self.tokenizer(self.extra_captions[idx])
            if not len(set([i.size() for i in images])) == 1:
                return self.__getitem__((idx + 1) % len(self))
            else:
                return images, label, caption
        except:
            return self.__getitem__((idx + 1) % len(self))

class HPDDataset(Dataset):
    def __init__(self, meta_file, image_folder, transforms, tokenizer, is_train=True):
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.open_image = Image.open
        self.is_train = is_train
        if image_folder.startswith('s3://'):
            loader = TCSLoader()
            self.open_image = loader
        if meta_file is not None:
            with open(meta_file, 'r') as f:
                self.table = pa.Table.from_pylist(json.load(f))
            self.image_folder = image_folder
        else:
            self.table = []
        
    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        try:
            if self.is_train:
                images = [self.transforms(self.open_image(os.path.join(self.image_folder, file_names))) for file_names in self.table.column('file_path')[idx].as_py()]
                if not len(set([i.size() for i in images])) == 1:
                    return self.__getitem__((idx + 1) % len(self))
                label = self.table.column('human_preference')[idx].as_py()
                caption = self.tokenizer(self.table.column('prompt')[idx].as_py())
                # num_per_prompt = self.table.column('num_per_prompt')[idx].as_py()
                return images, label, caption
            else:
                images = [self.transforms(self.open_image(os.path.join(self.image_folder, file_names))) for file_names in self.table.column('file_path')[idx].as_py()]
                if not len(set([i.size() for i in images])) == 1:
                    return self.__getitem__((idx + 1) % len(self))
                label = self.table.column('human_preference')[idx].as_py()
                caption = self.tokenizer(self.table.column('prompt')[idx].as_py())
                return images, label, caption
        except:
            return self.__getitem__((idx + 1) % len(self))

        
class RatingDataset(Dataset):
    def __init__(self, meta_file, image_folder, transforms):
        self.transforms = transforms
        self.image_folder = image_folder
        self.open_image = Image.open
        self.max_size = 224
        if image_folder.startswith('s3://'):
            loader = TCSLoader()
            self.open_image = loader
        with open(meta_file, 'r') as f:
            self.table = pa.Table.from_pylist(json.load(f))

        
    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        try:
            images = self.transforms(self.open_image(os.path.join(self.image_folder, self.table.column('path')[idx].as_py())))
            img_weight, img_height = images.shape[1:]
            if img_weight != self.max_size or img_height != self.max_size:
                return self.__getitem__((idx + 10) % len(self))
            label = self.table.column('rating')[idx].as_py()
            return images, label
        except:
            return self.__getitem__((idx + 1) % len(self))

class RankingDataset(Dataset):
    def __init__(self, meta_file, image_folder, transforms, tokenizer):
        self.transforms = transforms
        self.image_folder = image_folder     
        self.open_image = Image.open
        if image_folder.startswith('s3://'):
            loader = TCSLoader()
            self.open_image = loader
        self.tokenizer = tokenizer

        with open(meta_file, 'r') as f:
            self.table = pa.Table.from_pylist(json.load(f))

    
    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        try:
            images = [self.transforms(self.open_image(os.path.join(self.image_folder, file_names))) for file_names in self.table.column('image_path')[idx].as_py()]
            label = self.table.column('rank')[idx].as_py()
            caption = self.tokenizer(self.table.column('prompt')[idx].as_py())
            return images, label, caption
        except:
            return self.__getitem__((idx + 1) % len(self))

class RegionDataset(Dataset):
    def __init__(self, meta_file, image_folder, transforms):
        self.transforms = transforms
        self.image_folder = image_folder
        self.open_image = Image.open

        with open(meta_file,'r') as f:
            self.table = pa.Table.from_pylist(json.load(f))


    
    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, idx):
        try:
            img = self.open_image(os.path.join(self.image_folder, self.table.column('image_path')[idx].as_py()))
            mask = self.open_image(os.path.join(self.image_folder, self.table.column('mask_path')[idx].as_py()))
            img.putalpha(mask)
            masked_image = self.transforms(img)
            image = masked_image[:3]
            mask = masked_image[3]
            return image, mask
        except:
            return self.__getitem__((idx + 1) % len(self))

class ImageRewardDataset(Dataset):
    def __init__(self, meta_file, image_folder,transforms, tokenizer):
        self.transforms = transforms
        self.image_folder = image_folder     
        self.open_image = Image.open
        self.tokenizer = tokenizer

        with open(meta_file, 'r') as f:
            self.table = pa.Table.from_pylist(json.load(f))
        
    def __len__(self):
        return len(self.table)
    
    def __getitem__(self, idx):

        images = [self.transforms(self.open_image(os.path.join(self.image_folder, file_names))) for file_names in self.table.column('generations')[idx].as_py()]
        label = self.table.column('ranking')[idx].as_py()
        caption = self.tokenizer(self.table.column('prompt')[idx].as_py())
        return images, label, caption


def set_env_vars(something):
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''

def collate_rating(batch):
    images = [sample[0] for sample in batch]
    labels = torch.tensor([sample[1] for sample in batch])
    images = torch.stack(images)
    return images, labels

def get_rating_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    # only training data
    assert is_train
    dataset = RatingDataset(meta_file=args.train_data,
        image_folder=args.train_folder,
        transforms=preprocess_fn)
    num_samples = len(dataset)
    sampler = TrainingSampler(dataset) if args.distributed else None
    shuffle = is_train and not args.distributed

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=collate_rating,
        worker_init_fn=set_env_vars,
        persistent_workers=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader=dataloader, sampler=sampler, data_type='rating')

def collate_pref(batch):
    images = [torch.stack(sample[0]) for sample in batch]
    num_images = torch.tensor([g.size(0) for g in images])
    labels = torch.tensor([sample[1] for sample in batch])
    captions = torch.cat([sample[2] for sample in batch])
    images = torch.cat(images)
    return images, num_images, labels, captions

def get_preference_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None, extra_val=False):
    if is_train:
        extra_data = (args.extra_train_data, args.extra_train_folder)
        dataset = PreferenceDataset(meta_file=args.train_data if is_train else args.val_data,
            image_folder=args.train_folder if is_train else args.val_folder,
            transforms=preprocess_fn, tokenizer=tokenizer, extra_data=extra_data)
    else:
        if extra_val:
            dataset = PreferenceDataset(meta_file=None,
                image_folder=None,
                transforms=preprocess_fn, tokenizer=tokenizer, extra_data=(args.extra_val_data, args.extra_val_folder))
        else:
            dataset = PreferenceDataset(meta_file=args.val_data,
                image_folder=args.val_folder,
                transforms=preprocess_fn, tokenizer=tokenizer)

    num_samples = len(dataset)
    sampler = TrainingSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and not args.distributed and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=collate_pref,
        worker_init_fn=set_env_vars,
        persistent_workers=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader=dataloader, sampler=sampler, data_type='preference')

def collate_HPD(batch):
    image_1 = torch.stack([sample[0][0] for sample in batch])
    image_2 = torch.stack([sample[0][1] for sample in batch])
    label_1 = torch.tensor([sample[1][0] for sample in batch])
    label_2 = torch.tensor([sample[1][1] for sample in batch])
    labels = torch.cat([label_1, label_2], dim=0)
    captions = torch.cat([sample[2] for sample in batch])
    images = torch.cat([image_1, image_2])
    return images, labels, captions

def get_HPD_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    dataset = HPDDataset(meta_file=args.train_data if is_train else args.val_data,
        image_folder=args.train_folder if is_train else args.val_folder,
        transforms=preprocess_fn, tokenizer=tokenizer, is_train=is_train)

    num_samples = len(dataset)
    sampler = TrainingSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and not args.distributed and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=collate_HPD if is_train else collate_pref,
        worker_init_fn=set_env_vars,
        persistent_workers=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader=dataloader, sampler=sampler, data_type='HPD' if is_train else 'preference')

def get_ranking_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    if is_train:
        dataset = RankingDataset(meta_file=args.train_data,
            image_folder=args.train_folder, transforms=preprocess_fn, tokenizer=tokenizer)
    else:
        dataset = RankingDataset(meta_file=args.val_data,
            image_folder=args.val_folder, transforms=preprocess_fn, tokenizer=tokenizer)

    num_samples = len(dataset)
    sampler = TrainingSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and not args.distributed and sampler is None
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        collate_fn=collate_rank,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader=dataloader, sampler=sampler, data_type='ranking')

def get_regional_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    if is_train:
        dataset = RegionDataset(
            meta_file=args.train_data,
            image_folder=args.train_folder,
            transforms=preprocess_fn
        )
    else:
        dataset = RegionDataset(
            meta_file=args.val_data,
            image_folder=args.val_folder,
            transforms=preprocess_fn
        )
    num_samples = len(dataset)
    sampler = TrainingSampler(dataset) if args.distributed else None
    shuffle = is_train and not args.distributed

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
        worker_init_fn=set_env_vars,
        persistent_workers=True,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader=dataloader, sampler=sampler, data_type='regional')

def collate_rank(batch):
    images = [torch.stack(sample[0]) for sample in batch]
    num_images = torch.tensor([g.size(0) for g in images])
    labels = [torch.tensor(sample[1]) for sample in batch]
    captions = torch.cat([sample[2] for sample in batch])
    images = torch.cat(images)
    labels = torch.cat(labels)
    return images, num_images, labels, captions

def get_imagereward_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    #only support evaluation
    if not is_train:
        dataset = ImageRewardDataset(
            meta_file=args.val_data,
            image_folder = args.val_folder,
            transforms=preprocess_fn,
            tokenizer=tokenizer
        )
        num_samples = len(dataset)
        sampler = TrainingSampler(dataset) if args.distributed and is_train else None
        shuffle = is_train and not args.distributed
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=is_train,
            worker_init_fn=set_env_vars,
            collate_fn=collate_rank,
            persistent_workers=True,
        )
        dataloader.num_samples = num_samples
        dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader=dataloader, sampler=sampler, data_type='ImageReward')

def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    elif dataset_type == "preference":
        return get_preference_dataset
    elif dataset_type == "rating":
        return get_rating_dataset
    elif dataset_type == 'ranking':
        return get_ranking_dataset
    elif dataset_type == 'regional':
        return get_regional_dataset
    elif dataset_type == 'ImageReward':
        return get_imagereward_dataset
    elif dataset_type == "HPD":
        return get_HPD_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        assert len(args.train_data) == len(args.dataset_type) == len(args.batch_size) == len(args.workers) == len(args.train_folder) == len(args.train_data_sample_ratio) == len(args.ignore_in_train)
        for train_data, dataset_type, batch_size, workers, train_folder, train_data_sample_ratio, ignore in zip(args.train_data, args.dataset_type, args.batch_size, args.workers, args.train_folder, args.train_data_sample_ratio, args.ignore_in_train):
            if ignore:
                continue
            if 'train' not in data:
                data['train'] = []
            new_args = copy.deepcopy(args)
            new_args.train_data = train_data
            new_args.dataset_type = dataset_type
            new_args.batch_size = batch_size
            new_args.workers = workers
            new_args.train_folder = train_folder
            new_args.train_data_sample_ratio = train_data_sample_ratio
            dataset = get_dataset_fn(new_args.train_data, new_args.dataset_type)(
                new_args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)
            data['train'].append(dataset)

    if args.val_data[0]:
        assert len(args.val_data) == len(args.dataset_type) == len(args.batch_size) == len(args.workers) == len(args.val_folder) == len(args.ignore_in_val)
        # data['val'] = []
        for val_data, dataset_type, batch_size, workers, val_folder ,ignore in zip(args.val_data, args.dataset_type, args.batch_size, args.workers, args.val_folder, args.ignore_in_val):
            if ignore:
                continue
            if 'val' not in data:
                data['val'] = []
            new_args = copy.deepcopy(args)
            new_args.val_data = val_data
            new_args.dataset_type = dataset_type
            new_args.batch_size = batch_size
            new_args.workers = workers
            new_args.val_folder = val_folder
            dataset = get_dataset_fn(new_args.val_data, new_args.dataset_type)(
                new_args, preprocess_val, is_train=False, tokenizer=tokenizer)
            data['val'].append(dataset)

    if args.extra_val_data:
        assert False
        data["extra_val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer, extra_val=True)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data

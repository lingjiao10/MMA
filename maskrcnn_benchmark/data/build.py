# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging

import torch.utils.data
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.imports import import_file

from . import datasets as D
from . import samplers

from .collate_batch import BatchCollator
from .transforms import build_transforms


def build_dataset(dataset_list, transforms, dataset_catalog, is_train=True, external_proposal=False, old_classes=None, new_classes=None, excluded_classes=None):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e. coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError("dataset_list should be a list of strings, got {}".format(dataset_list))
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]
        # for COCO Dataset, we want to remove images without annotations during training
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = is_train
            args["is_train"] = is_train
        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train  # during training, do not use difficult
            args["external_proposal"] = external_proposal  # whether use external proposals
            args["old_classes"] = old_classes
            args["new_classes"] = new_classes
            args["excluded_classes"] = excluded_classes
            args["is_train"] = is_train
        if data["factory"] == "PascalVOCDataset2012":
            args["use_difficult"] = not is_train  # during training, do not use difficult
            args["external_proposal"] = external_proposal  # whether use external proposals
            args["old_classes"] = old_classes
            args["new_classes"] = new_classes
            args["excluded_classes"] = excluded_classes
            args["is_train"] = is_train
        if data["factory"] == "GlomDataset":
            args["is_train"] = is_train
        if data["factory"] == "CompressionDataset":
            args["is_train"] = is_train
        args["transforms"] = transforms

        # make dataset from factory
        # print('args: {0}'.format(args))
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed, num_gpus=1, rank=0):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle, num_replicas=num_gpus, rank=rank)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0):

    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(sampler, group_ids, images_per_batch, drop_uneven=False)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, images_per_batch, drop_last=False)
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)

    return batch_sampler


def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0, external_proposal=False, compression_not_shuffle=False, num_gpus=1, rank=0):
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (images_per_batch % num_gpus == 0), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        if compression_not_shuffle:
            shuffle = False
        else:
            shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (images_per_batch % num_gpus == 0), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file("maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True)
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    transforms = build_transforms(cfg, is_train)
    old_classes = cfg.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES
    new_classes = cfg.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES
    excluded_classes = cfg.MODEL.ROI_BOX_HEAD.NAME_EXCLUDED_CLASSES

    datasets = build_dataset(dataset_list, transforms, DatasetCatalog, is_train, external_proposal, old_classes, new_classes, excluded_classes)

    data_loaders = []
    for i, dataset in enumerate(datasets):
        print(f"Dataset {dataset_list[i]} contains {len(dataset)} images.")
        sampler = make_data_sampler(dataset, shuffle, is_distributed, num_gpus, rank)
        batch_sampler = make_batch_data_sampler(dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter)
        collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler,
                                                  collate_fn=collator, num_workers=num_workers)
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders

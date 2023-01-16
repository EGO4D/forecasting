#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# Lightning Trainer should be considered beta at this point
# We have confirmed that training and validation run correctly and produce correct results
# Depending on how you launch the trainer, there are issues with processes terminating correctly
# This module is still dependent on D2 logging, but could be transferred to use Lightning logging

import logging
import os
import time
import weakref
from collections import OrderedDict
from typing import Any, Dict, List
import pytorch_lightning as pl  # type: ignore
from pytorch_lightning import LightningDataModule, LightningModule
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2 import model_zoo

bases = {
    'faster_rcnn' : {
        'R50_FPN_1x' : "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",
        'R50_FPN_3x' : "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        "R101_FPN_3x" : "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    },
    'retinanet' : {
        'R50_FPN_1x' : "COCO-Detection/retinanet_R_50_FPN_1x.yaml"
    }
}

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
)
from detectron2.evaluation import print_csv_format
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger

from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


# TODO: import
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectron2")


class TrainingModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()

        self.cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        self.storage: EventStorage = None
        self.model = build_model(self.cfg)

        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["iteration"] = self.storage.iter

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]) -> None:
        self.start_iter = checkpointed_state["iteration"]
        self.storage.iter = self.start_iter

    def setup(self, stage: str):
        if self.cfg.MODEL.WEIGHTS:
            self.checkpointer = DetectionCheckpointer(
                # Assume you want to save checkpoints together with logs/statistics
                self.model,
                self.cfg.OUTPUT_DIR,
            )
            logger.info(f"Load model weights from checkpoint: {self.cfg.MODEL.WEIGHTS}.")
            # Only load weights, use lightning checkpointing if you want to resume
            self.checkpointer.load(self.cfg.MODEL.WEIGHTS)

        self.iteration_timer = hooks.IterationTimer()
        self.iteration_timer.before_train()
        self.data_start = time.perf_counter()
        self.writers = None

    def training_step(self, batch, batch_idx):
        data_time = time.perf_counter() - self.data_start
        # Need to manually enter/exit since trainer may launch processes
        # This ideally belongs in setup, but setup seems to run before processes are spawned
        if self.storage is None:
            self.storage = EventStorage(0)
            self.storage.__enter__()
            self.iteration_timer.trainer = weakref.proxy(self)
            self.iteration_timer.before_step()
            self.writers = (
                default_writers(self.cfg.OUTPUT_DIR, self.max_iter)
                if comm.is_main_process()
                else {}
            )

        loss_dict = self.model(batch)
        SimpleTrainer.write_metrics(loss_dict, data_time)

        opt = self.optimizers()
        self.storage.put_scalar(
            "lr", opt.param_groups[self._best_param_group_id]["lr"], smoothing_hint=False
        )
        self.iteration_timer.after_step()
        self.storage.step()
        # A little odd to put before step here, but it's the best way to get a proper timing
        self.iteration_timer.before_step()

        if self.storage.iter % 20 == 0:
            for writer in self.writers:
                writer.write()
        return sum(loss_dict.values())

    def training_step_end(self, training_step_outpus):
        self.data_start = time.perf_counter()
        return training_step_outpus

    def training_epoch_end(self, training_step_outputs):
        self.iteration_timer.after_train()
        if comm.is_main_process():
            self.checkpointer.save("model_final")
        for writer in self.writers:
            writer.write()
            writer.close()
        self.storage.__exit__(None, None, None)

    def _process_dataset_evaluation_results(self, _outputs) -> OrderedDict:
        results = OrderedDict()
        for idx, dataset_name in enumerate(self.cfg.DATASETS.TEST):
            results[dataset_name] = self._evaluators[idx].evaluate()
            if comm.is_main_process():
                print_csv_format(results[dataset_name])

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def _reset_dataset_evaluators(self):
        self._evaluators = []
        for dataset_name in self.cfg.DATASETS.TEST:
            evaluator = build_evaluator(self.cfg, dataset_name)
            evaluator.reset()
            self._evaluators.append(evaluator)

    def on_validation_epoch_start(self):
        self._reset_dataset_evaluators()

    def validation_epoch_end(self, _outputs):
        results = self._process_dataset_evaluation_results(_outputs)

        flattened_results = flatten_results_dict(results)
        for k, v in flattened_results.items():
            try:
                v = float(v)
            except Exception as e:
                raise ValueError(
                    "[EvalHook] eval_function should return a nested dict of float. "
                    "Got '{}: {}' instead.".format(k, v)
                ) from e
        self.storage.put_scalars(**flattened_results, smoothing_hint=False)

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not isinstance(batch, List):
            batch = [batch]
        outputs = self.model(batch)
        self._evaluators[dataloader_idx].process(batch, outputs)

    def configure_optimizers(self):
        optimizer = build_optimizer(self.cfg, self.model)
        self._best_param_group_id = hooks.LRScheduler.get_best_param_group_id(optimizer)
        scheduler = build_lr_scheduler(self.cfg, optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class DataModule(LightningDataModule):
    def __init__(self, cfg, args):
        super().__init__()
        self.cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        self.args = args

    def train_dataloader(self):
        try:
            data = DatasetCatalog.get("ego4d_train")
        except:
            args = self.args
            register_coco_instances("ego4d_train", {}, args.path_to_train_json, args.path_to_images)
            register_coco_instances("ego4d_val", {}, args.path_to_val_json, args.path_to_images)
        return build_detection_train_loader(self.cfg)

    def val_dataloader(self):
        try:
            data = DatasetCatalog.get("ego4d_val")
        except:
            args = self.args
            register_coco_instances("ego4d_val", {}, args.path_to_val_json, args.path_to_images)

        dataloaders = []
        for dataset_name in self.cfg.DATASETS.TEST:
            dataloaders.append(build_detection_test_loader(self.cfg, dataset_name))
        return dataloaders



def train(cfg, args):
    trainer_params = {
        # training loop is bounded by max steps, use a large max_epochs to make
        # sure max_steps is met first
        "max_epochs": 10**8,
        "max_steps": cfg.SOLVER.MAX_ITER,
        "val_check_interval": cfg.TEST.EVAL_PERIOD if cfg.TEST.EVAL_PERIOD > 0 else 10**8,
        "num_nodes": args.num_machines,
        "gpus": args.num_gpus,
        "num_sanity_val_steps": 0,
    }
    if cfg.SOLVER.AMP.ENABLED:
        trainer_params["precision"] = 16

    last_checkpoint = os.path.join(cfg.OUTPUT_DIR, "last.ckpt")
    if args.resume:
        # resume training from checkpoint
        trainer_params["resume_from_checkpoint"] = last_checkpoint
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}.")

    trainer = pl.Trainer(**trainer_params, replace_sampler_ddp=False)
    logger.info(f"start to train with {args.num_machines} nodes and {args.num_gpus} GPUs")

    module = TrainingModule(cfg, )
    data_module = DataModule(cfg, args)
    if args.eval_only:
        logger.info("Running inference")
        trainer.validate(module, data_module)
    else:
        logger.info("Running training")
        trainer.fit(module, data_module)



def main(args):
    register_coco_instances("ego4d_train", {}, args.path_to_train_json, args.path_to_images)
    register_coco_instances("ego4d_val", {}, args.path_to_val_json, args.path_to_images)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(bases[args.arch][args.base]))
    cfg.merge_from_file(args.cfg)
    if args.batch_size is not None:
        cfg.SOLVER.IMS_PER_BATCH = args.batch_size


    data = DatasetCatalog.get("ego4d_train")
    metadata = MetadataCatalog.get("ego4d_train")

    base_len = 94433  # the length of the COCO dataset

    current_len = len(data) # the length of the loaded dataset
    num_classes = len(metadata.thing_classes)

    # TODO: load from disk
    if args.arch == "faster_rcnn":
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    else:
        cfg.MODEL.RETINANET.NUM_CLASSES = num_classes

    scale_factor = current_len/base_len # scale factor to adapt the learning schedule
    cfg.OUTPUT_DIR = str(args.output_dir)
    cfg.SOLVER.MAX_ITER = int(scale_factor*cfg.SOLVER.MAX_ITER) #adapt max iter
    cfg.SOLVER.STEPS = [int(scale_factor*x) for x in cfg.SOLVER.STEPS] #adapt steps
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.TEST.EVAL_PERIOD = args.eval_period

    train(cfg, args)

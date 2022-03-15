import os
import pickle
import pprint

import ego4d.utils.logging as logging
import numpy as np
import torch
from ego4d.tasks.short_term_anticipation import ShortTermAnticipationTask
from ego4d.utils.c2_model_loading import get_name_convert_func
from ego4d.utils.parser import load_config, parse_args
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from scripts.slurm import copy_and_run_with_config
from pytorch_lightning.plugins import DDPPlugin


logger = logging.get_logger(__name__)


def main(cfg):
    seed_everything(cfg.RNG_SEED)

    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("Run with config:")
    logger.info(pprint.pformat(cfg))

    # Choose task type based on config.
    if cfg.DATA.TASK == "short_term_anticipation":
        TaskType = ShortTermAnticipationTask

    task = TaskType(cfg)

    # Load model from checkpoint if checkpoint file path is given.
    ckp_path = cfg.CHECKPOINT_FILE_PATH
    if len(ckp_path) > 0:
        if cfg.CHECKPOINT_VERSION == "caffe2":
            with open(ckp_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            state_dict = data["blobs"]
            fun = get_name_convert_func()
            state_dict = {
                fun(k): torch.from_numpy(np.array(v))
                for k, v in state_dict.items()
                if "momentum" not in k and "lr" not in k and "model_iter" not in k
            }

            if not cfg.CHECKPOINT_LOAD_MODEL_HEAD:
                state_dict = {k: v for k, v in state_dict.items() if "head" not in k}
            print(task.model.load_state_dict(state_dict, strict=False))
            print(f"Checkpoint {ckp_path} loaded")
        elif cfg.DATA.CHECKPOINT_MODULE_FILE_PATH != "":

            # Load slowfast weights into backbone submodule
            ckpt = torch.load(
                cfg.DATA.CHECKPOINT_MODULE_FILE_PATH,
                map_location=lambda storage, loc: storage,
            )

            def remove_first_module(key):
                return ".".join(key.split(".")[1:])

            state_dict = {
                remove_first_module(k): v
                for k, v in ckpt["state_dict"].items()
                if "head" not in k
            }
            missing_keys, unexpected_keys = task.model.backbone.load_state_dict(
                state_dict, strict=False
            )

            # Ensure only head key is missing.
            assert len(unexpected_keys) == 0
            assert all(["head" in x for x in missing_keys])
        else:
            # Load all child modules except for "head" if CHECKPOINT_LOAD_MODEL_HEAD is
            # False.
            pretrained = TaskType.load_from_checkpoint(ckp_path)
            state_dict_for_child_module = {
                child_name: child_state_dict.state_dict()
                for child_name, child_state_dict in pretrained.model.named_children()
            }
            for child_name, child_module in task.model.named_children():
                if not cfg.CHECKPOINT_LOAD_MODEL_HEAD and "head" in child_name:
                    continue

                state_dict = state_dict_for_child_module[child_name]
                child_module.load_state_dict(state_dict)

    checkpoint_callback = ModelCheckpoint(
        monitor=task.checkpoint_metric, mode="min", save_last=True, save_top_k=1
    )
    if cfg.ENABLE_LOGGING:
        args = {"callbacks": [LearningRateMonitor(), checkpoint_callback]}
    else:
        args = {"logger": False, "callbacks": checkpoint_callback}

    trainer = Trainer(
        gpus=cfg.NUM_GPUS,
        num_nodes=cfg.NUM_SHARDS,
        accelerator=cfg.SOLVER.ACCELERATOR,
        max_epochs=cfg.SOLVER.MAX_EPOCH,
        num_sanity_val_steps=3,
        benchmark=True,
        log_gpu_memory="min_max",
        replace_sampler_ddp=False,
        fast_dev_run=cfg.FAST_DEV_RUN,
        default_root_dir=cfg.OUTPUT_DIR,
        plugins=DDPPlugin(find_unused_parameters=False),
        **args,
    )

    if cfg.TRAIN.ENABLE and cfg.TEST.ENABLE:
        trainer.fit(task)

        # Calling test without the lightning module arg automatically selects the best
        # model during training.
        return trainer.test()

    elif cfg.TRAIN.ENABLE:
        return trainer.fit(task)

    elif cfg.TEST.ENABLE:
        return trainer.test(task)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)
    if args.on_cluster:
        copy_and_run_with_config(
            main,
            cfg,
            args.working_directory,
            job_name=args.job_name,
            time="72:00:00",
            partition="learnfair",
            gpus_per_node=cfg.NUM_GPUS,
            ntasks_per_node=cfg.NUM_GPUS,
            cpus_per_task=10,
            mem="470GB",
            nodes=cfg.NUM_SHARDS,
            constraint="volta32gb",
        )
    else:  # local
        main(cfg)

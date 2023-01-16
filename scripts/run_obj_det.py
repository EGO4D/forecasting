import sys
import os

from argparse import ArgumentParser
from detectron2 import model_zoo
from pathlib import Path
from tools.short_term_anticipation.train_object_detector import run_main, bases
from scripts.slurm import copy_and_run_with_config
from detectron2.config import get_cfg

from detectron2.data.datasets import register_coco_instances


def construct_arg_parser():
    parser = ArgumentParser()

    parser.add_argument('path_to_train_json', type=Path)
    parser.add_argument('path_to_val_json', type=Path)
    parser.add_argument('path_to_images', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--num_machines', type=int, default=1)
    parser.add_argument('--checkpoint_period', type=int, default=5000)
    parser.add_argument('--eval_period', type=int, default=5000)
    # parser.add_argument('--base', type=str, default='R101_FPN_3x')
    parser.add_argument('--base', type=str, default='R50_FPN_3x')
    parser.add_argument('--arch', type=str, default='faster_rcnn')
    parser.add_argument('--base_lr', type=float, default=0.002)
    parser.add_argument('--slurm', type=int, default=0)
    parser.add_argument('--working_directory', type=str, default="")
    parser.add_argument('--job_name', type=str, default="sta_obj_det")
    parser.add_argument('--resume', action="store_true", default=False)
    parser.add_argument('--eval_only', action="store_true", default=False)
    parser.add_argument('--on_cluster', action="store_true", default=False)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--cfg', type=str, default="configs/Ego4dShortTermAnticipation/ObjDet.yaml")
    return parser


if __name__ == "__main__":
    parser = construct_arg_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14

    if args.on_cluster:
        copy_and_run_with_config(
            run_main,
            args,
            args.working_directory,
            job_name=args.job_name,
            time="72:00:00",
            partition="pixar",
            gpus_per_node=args.num_gpus,
            ntasks_per_node=args.num_gpus,
            cpus_per_task=10,
            mem="470GB",
            nodes=1,
            constraint="volta32gb",
        )
    else:
        run_main(args)

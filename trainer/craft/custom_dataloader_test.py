"""
time: 10 sec

"""

import sys
import psutil
import time
import yaml
import argparse

import torch

from config.load_config import load_yaml, DotDict
from model.craft import CRAFT
from data.dataset import CustomDataset, CustomDataset2
from data.imgproc import denormalizeMeanVariance
from utils.util import saveInput, saveImage, copyStateDict
# from decorator_wraps import memory_printer

if __name__ == "__main__":
    # print(sys.path)

    parser = argparse.ArgumentParser(description="CRAFT IC15 Train")
    parser.add_argument("--yaml",
                        "--yaml_file_name",
                        default="syn_train2",
                        type=str,
                        help="Load configuration")
    args = parser.parse_args()
    config = load_yaml(args.yaml)
    config = DotDict(config)

    start_time = time.time()
    print(yaml.dump(config))

    craft = CRAFT(pretrained=True, amp=config.train.amp)
    if config.train.ckpt_path is not None:
        net_param = torch.load(config.train.ckpt_path)
        craft.load_state_dict(copyStateDict(net_param["craft"]))
    craft = craft.cuda()

    # custom_dataset = CustomDataset(
    #     output_size=config.train.data.output_size,
    #     data_dir=config.data_root_dir,
    #     saved_gt_dir=None,
    #     mean=config.train.data.mean,
    #     variance=config.train.data.variance,
    #     gauss_init_size=config.train.data.gauss_init_size,
    #     gauss_sigma=config.train.data.gauss_sigma,
    #     enlarge_region=config.train.data.enlarge_region,
    #     enlarge_affinity=config.train.data.enlarge_affinity,
    #     watershed_param=config.train.data.watershed,
    #     aug=config.train.data.custom_aug,
    #     vis_test_dir=config.vis_test_dir,
    #     sample=config.train.data.custom_sample,
    #     vis_opt=config.train.data.vis_opt,
    #     pseudo_vis_opt=config.train.data.pseudo_vis_opt,
    #     do_not_care_label=config.train.data.do_not_care_label,
    # )

    custom_dataset = CustomDataset2(
        output_size=config.train.data.output_size,
        data_dir=config.data_root_dir,
        saved_gt_dir=None,
        mean=config.train.data.mean,
        variance=config.train.data.variance,
        gauss_init_size=config.train.data.gauss_init_size,
        gauss_sigma=config.train.data.gauss_sigma,
        enlarge_region=config.train.data.enlarge_region,
        enlarge_affinity=config.train.data.enlarge_affinity,
        aug=config.train.data.custom_aug,
        vis_test_dir=config.vis_test_dir,
        sample=config.train.data.custom_sample,
        vis_opt=config.train.data.vis_opt,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # custom_dataset.update_model(craft)
    # custom_dataset.update_device(device)

    ic15_train_loader = torch.utils.data.DataLoader(
        custom_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=False,
    )

    batch_ic = iter(ic15_train_loader)
    ic15_images, ic15_region_label, ic15_affi_label, ic15_confidence_mask = next(
        batch_ic
    )
    print(ic15_images.numpy()[0].transpose(1, 2, 0).shape, ic15_images.numpy()[0].transpose(1, 2, 0).dtype,
          ic15_images.numpy()[0].transpose(1, 2, 0).max(), ic15_images.numpy()[0].transpose(1, 2, 0).min())
    print(ic15_region_label.numpy()[0].shape, ic15_region_label.numpy()[0].dtype, ic15_region_label.numpy()[0].max(),
          ic15_region_label.numpy()[0].min())
    print(ic15_affi_label.numpy()[0].shape, ic15_affi_label.numpy()[0].dtype, ic15_affi_label.numpy()[0].max(),
          ic15_affi_label.numpy()[0].min())
    print(ic15_confidence_mask.numpy()[0].shape, ic15_confidence_mask.numpy()[0].dtype,
          ic15_confidence_mask.numpy()[0].max(), ic15_confidence_mask.numpy()[0].min())

    if config.train.data.vis_opt:
        for i in range(config.train.batch_size):
            saveInput(
                f'test_img_{i}',
                config.vis_test_dir,
                denormalizeMeanVariance(ic15_images.numpy()[i].transpose(1, 2, 0)),
                ic15_region_label.numpy()[i],
                ic15_affi_label.numpy()[i],
                ic15_confidence_mask.numpy()[i]
            )

    print(f"elapsed time : {time.time() - start_time}")

import json
from torchvision.ops import box_convert
import torch
import shutil
# result format : 377,117,463,117,465,130,378,130,Genaxis Theatre

img_dir = '/data/artlab/20220718_SooickBot+TroubleDetectionAcne466_cor/train/JPEGImages/'
result_root_dir = 'data_root_dir'
type = 'training' # training | test

with open('/data/artlab/20220718_SooickBot+TroubleDetectionAcne466_cor/train/annotations.json', "r") as j:
    full_json = json.load(j)

    i = 0
    img_path = full_json['images'][i]['file_name']
    img_name = img_path.split('/')[-1]
    img_id = full_json['annotations'][i]['image_id']


    filenames = [file_attributes['file_name'].split('/')[1] for file_attributes in full_json['images'] if file_attributes['file_name'].split('/')[1].startswith("newdeal")]

    img_idx = 0
    total_text = ''
    for j in range(len(full_json['annotations'])):
        if img_idx != full_json['annotations'][j]['image_id']:
            shutil.copy(f'{img_dir}/{filenames[img_idx]}',
                        f'./{result_root_dir}/ch4_{type}_images/{filenames[img_idx]}')
            with open(f"./{result_root_dir}/ch4_{type}_localization_transcription_gt/{filenames[img_idx][:-4]}.txt", "a") as f:
                f.write(total_text)
                f.close()
            img_idx += 1
            total_text = ''
        bbox_points = full_json['annotations'][j]['bbox']  # format : cx, cy, w, h
        bbox_points = box_convert(torch.Tensor(bbox_points), 'xywh', 'xyxy').tolist()
        bbox_points = list(map(int, bbox_points))
        top_left = f'{bbox_points[0]},{bbox_points[1]}'
        top_right = f'{bbox_points[2]},{bbox_points[1]}'
        bottom_right = f'{bbox_points[2]},{bbox_points[3]}'
        bottom_left = f'{bbox_points[0]},{bbox_points[3]}'
        result_line = f'{top_left},{top_right},{bottom_right},{bottom_left},N'
        total_text = total_text + result_line + '\n'





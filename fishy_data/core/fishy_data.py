import fiftyone as fo
import fiftyone.utils.yolo as fouy
from os.path import join

name = 'CichlidDetection_v1'
dataset_dir = join('../data', name)
splits = ['train', 'test', 'val']
classes = ['f', 'm']
dataset = fo.Dataset(name)
for split in splits:
    dataset.add_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        split=split,
        tags=split,
        label_field='ground_truth'
    )
for split in splits:
    fouy.add_yolo_labels(sample_collection=dataset,
                         label_field='predictions',
                         labels_path=f'../data/CichlidDetection_v1_Detect/labels/{split}',
                         classes=classes)





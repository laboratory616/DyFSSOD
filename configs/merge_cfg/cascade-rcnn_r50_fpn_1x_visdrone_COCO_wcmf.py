_base_ = [
    '../models/cascade-rcnn_r50_visdrone_wcmf.py',
    '../datasets/coco_detection_visdrone.py',
    '../schedules/schedule_1x_visdrone.py', '../../configs/_base_/default_runtime.py'
]
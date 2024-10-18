# For single image inference
from mmdet.apis import init_detector, inference_detector
import mmcv

# Paths to the config file and the model checkpoint
config_file = './inference_gt_app/configs/fast_rcnn_bdd100/faster_rcnn_r50_fpn_1x_det_bdd100k.py'
checkpoint_file = './inference_gt_app/configs/fast_rcnn_bdd100/faster_rcnn_r50_fpn_1x_det_bdd100k.pth'

model = init_detector(config_file, checkpoint_file, device='cpu')  # Set device to 'cpu' or 'cuda:0'
print(model.CLASSES)

img = mmcv.imread("./inference_gt_app/test_image.jpg") 
output_image_path = "./inference_gt_app/test_image_output.jpg"

result = inference_detector(model, img)
print("Model's Result: ", result,)
model.show_result(img, result, out_file=output_image_path)
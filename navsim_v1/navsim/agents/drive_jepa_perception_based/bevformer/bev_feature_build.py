import numpy as np
from .transform3d import PhotoMetricDistortionMultiViewImage ,NormalizeMultiviewImage \
    ,RandomScaleImageMultiViewImage ,PadMultiViewImage
import torch

PhotoMetricDistortionMultiViewImage = PhotoMetricDistortionMultiViewImage(
    brightness_delta=32,
    contrast_range=(0.5, 1.5),
    saturation_range=(0.5, 1.5),
    hue_delta=18)
NormalizeMultiviewImage = NormalizeMultiviewImage(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
RandomScaleImageMultiViewImage = RandomScaleImageMultiViewImage(scales=[0.4])
PadMultiViewImage = PadMultiViewImage(size=None, size_divisor=32, pad_val=0)


def LoadMultiViewImageFromFiles(agent_input):
    image_result = {}

    lidar2img_rts = []
    lidar2cam_rts = []
    cam_intrinsics = []
   # image_result["camera2ego"] = []
    image_result["camera_intrinsics"] = []
    image_result["img"] = []

    for cameras in agent_input.cameras[-1:]:
        for cam in [cameras.cam_b0, cameras.cam_f0, cameras.cam_l0,cameras.cam_r0]:#, cameras.cam_l1, cameras.cam_r1 cameras.cam_l1, cameras.cam_l2, , cameras.cam_r1, cameras.cam_r2
            if cam.image is None:
                continue
            img = cam.image.astype(np.float32)

            image_result["img"].append(img)

            # obtain lidar to image transformation matrix
            lidar2cam_r = np.linalg.inv(cam.sensor2lidar_rotation)
            lidar2cam_t = cam.sensor2lidar_translation @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            intrinsic = cam.intrinsics
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt.T)
            lidar2img_rts.append(lidar2img_rt)

            cam_intrinsics.append(viewpad)
            lidar2cam_rts.append(lidar2cam_rt.T)

            # camera intrinsics
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = cam.intrinsics
            image_result["camera_intrinsics"].append(camera_intrinsics)


    image_result.update(
        dict(
            lidar2img=lidar2img_rts,
            cam_intrinsic=cam_intrinsics,
            lidar2cam=lidar2cam_rts,
        ))

    return image_result



def _get_bev_feature( agent_input, training: bool=False):
    image_result=LoadMultiViewImageFromFiles(agent_input)
    # if training:
    #     image_result = PhotoMetricDistortionMultiViewImage(image_result)
    image_result = NormalizeMultiviewImage(image_result)
    image_result = RandomScaleImageMultiViewImage(image_result)  # 432,768
    image_result = PadMultiViewImage(image_result)  # 448,768
    imgs = [img.transpose(2, 0, 1) for img in image_result['img']]

    camera_feature = torch.tensor(np.ascontiguousarray(np.stack(imgs, axis=0)))

    features = {
                "img_shape": torch.FloatTensor(np.stack(image_result["img_shape"])),#8,3
                "lidar2img": torch.FloatTensor(np.stack(image_result["lidar2img"]))#8,4,4
                }

    return features

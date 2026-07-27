from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers.image_transforms import convert_to_rgb, resize, to_channel_dimension_format
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, smart_resize
from transformers.utils import logging
from transformers.video_utils import VideoInput


logger = logging.get_logger(__name__)


class Qwen2VLImageProcessorExport(Qwen2VLImageProcessor):
    """
    Image preprocessor for ONNX export path.
    It changes output layout to [t, seq, tpp, c], so it can be converted to [t, c, seq, tpp].
    """

    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        do_resize: bool = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "input looks already scaled to [0, 1], set `do_rescale=False` to avoid double rescaling."
            )

        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []

        # transformers<=4.x exposed min_pixels/max_pixels directly, while
        # newer image processors keep them under size. Support both layouts
        # because this export helper intentionally subclasses the legacy
        # Qwen2-VL processor implementation.
        min_pixels = getattr(self, "min_pixels", None)
        max_pixels = getattr(self, "max_pixels", None)
        if min_pixels is None:
            min_pixels = getattr(self.size, "shortest_edge", None)
        if max_pixels is None:
            max_pixels = getattr(self.size, "longest_edge", None)
        if min_pixels is None:
            min_pixels = self.size["shortest_edge"]
        if max_pixels is None:
            max_pixels = self.size["longest_edge"]

        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=self.patch_size * self.merge_size,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                )
                image = resize(
                    image,
                    size=(resized_height, resized_width),
                    resample=resample,
                    input_data_format=input_data_format,
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(
                    image=image,
                    mean=image_mean,
                    std=image_std,
                    input_data_format=input_data_format,
                )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)

        remainder = patches.shape[0] % self.temporal_patch_size
        if remainder != 0:
            repeat_count = self.temporal_patch_size - remainder
            repeats = np.repeat(patches[-1][np.newaxis], repeat_count, axis=0)
            patches = np.concatenate([patches, repeats], axis=0)

        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size

        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )

        patches = patches.transpose(0, 3, 6, 4, 7, 1, 5, 8, 2)
        flatten_patches = patches.reshape(
            grid_t,
            grid_h * grid_w,
            self.temporal_patch_size * self.patch_size * self.patch_size,
            channel,
        )
        return flatten_patches, (grid_t, grid_h, grid_w)


def preprocess_image_to_nchw(
    image: Image.Image,
    image_processor: Qwen2VLImageProcessorExport,
) -> tuple[torch.Tensor, torch.LongTensor]:
    pixel_values, grid_thw = image_processor._preprocess(
        [image],
        do_resize=True,
        resample=PILImageResampling.BICUBIC,
        do_rescale=False,
        do_normalize=False,
        do_convert_rgb=True,
    )

    pixel_values = torch.from_numpy(pixel_values).to(torch.float32)
    mean = torch.tensor(image_processor.image_mean, dtype=torch.float32).view(1, 1, 1, 3) * 255.0
    std = torch.tensor(image_processor.image_std, dtype=torch.float32).view(1, 1, 1, 3) * 255.0
    pixel_values = (pixel_values - mean) / std

    pixel_values_nchw = pixel_values.permute(0, 3, 1, 2).contiguous()
    image_grid_thw = torch.tensor(grid_thw, dtype=torch.long).reshape(1, 3)
    return pixel_values_nchw, image_grid_thw

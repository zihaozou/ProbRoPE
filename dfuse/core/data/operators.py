import torch, torchvision, imageio, os
import imageio.v3 as iio
from PIL import Image


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = (
            [] if operators is None else operators
        )

    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data

    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")

    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value

    def __call__(self, data):
        if data is None:
            data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True, convert_RGBA=False):
        self.convert_RGB = convert_RGB
        self.convert_RGBA = convert_RGBA

    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB:
            image = image.convert("RGB")
        if self.convert_RGBA:
            image = image.convert("RGBA")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(
        self,
        height=None,
        width=None,
        max_pixels=None,
        height_division_factor=1,
        width_division_factor=1,
    ):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        image = torchvision.transforms.functional.center_crop(
            image, (target_height, target_width)
        )
        return image

    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width

    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class NumpyArrayCropAndResize(DataProcessingOperator):
    def __init__(
        self,
        height=None,
        width=None,
        max_pixels=None,
        height_division_factor=1,
        width_division_factor=1,
        vector_field_x_index=None,
        vector_field_y_index=None,
    ):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.vector_field_x_index = vector_field_x_index
        self.vector_field_y_index = vector_field_y_index

    def crop_and_resize(self, image, target_height, target_width):
        import cv2

        height, width = image.shape[:2]
        scale = max(target_width / width, target_height / height)
        new_size = (round(width * scale), round(height * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        if self.vector_field_x_index is not None:
            image[..., self.vector_field_x_index] *= scale
        if self.vector_field_y_index is not None:
            image[..., self.vector_field_y_index] *= scale


        h, w = image.shape[:2]
        y = (h - target_height) // 2
        x = (w - target_width) // 2
        image = image[y : y + target_height, x : x + target_width]
        return image

    def get_height_width(self, image):
        if self.height is None or self.width is None:
            height, width = image.shape[:2]
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width

    def __call__(self, data):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]


class LoadVideo(DataProcessingOperator):
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder

        self.frame_processor = frame_processor

    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while (
                num_frames > 1
                and num_frames % self.time_division_factor
                != self.time_division_remainder
            ):
                num_frames -= 1
        return num_frames

    def __call__(self, data: str):
        reader = imageio.get_reader(data)
        num_frames = self.get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(frame_id)
            frame = Image.fromarray(frame)
            frame = self.frame_processor(frame)
            frames.append(frame)
        reader.close()
        return frames


class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator

    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    def __init__(
        self,
        num_frames=81,
        time_division_factor=4,
        time_division_remainder=1,
        frame_processor=lambda x: x,
    ):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder

        self.frame_processor = frame_processor

    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while (
                num_frames > 1
                and num_frames % self.time_division_factor
                != self.time_division_remainder
            ):
                num_frames -= 1
        return num_frames

    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map

    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")


class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map

    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location

    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path

    def __call__(self, data):
        return os.path.join(self.base_path, data)


class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr

    def __call__(self, data: str):
        import librosa

        input_audio, sample_rate = librosa.load(data, sr=self.sr)
        return input_audio


class LoadNumpyArray(DataProcessingOperator):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, data):
        import numpy as np

        array = np.load(data)
        if self.dtype is not None:
            array = array.astype(self.dtype)
        return array


class TransposeArray(DataProcessingOperator):
    def __init__(self, axes):
        self.axes = axes

    def __call__(self, data):
        import numpy as np

        return np.transpose(data, self.axes)


class SelectFrames(DataProcessingOperator):
    def __init__(self, num_frames, time_division_factor=4, time_division_remainder=1):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder

    def get_num_frames(self, total_frames):
        num_frames = self.num_frames
        if total_frames < num_frames:
            num_frames = total_frames
            while (
                num_frames > 1
                and num_frames % self.time_division_factor
                != self.time_division_remainder
            ):
                num_frames -= 1
        return num_frames

    def __call__(self, data):
        total_frames = len(data)
        num_frames = self.get_num_frames(total_frames)
        if num_frames == total_frames:
            return data
        else:
            return data[:num_frames]


class StackArrays(DataProcessingOperator):
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, data):
        import numpy as np

        return np.stack(data, axis=self.axis)


class AppendDimension(DataProcessingOperator):
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, data):
        import numpy as np

        return np.expand_dims(data, axis=self.axis)


class ToNumpyArray(DataProcessingOperator):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, data):
        import numpy as np

        return np.array(data, dtype=self.dtype)


class LoadIntrinsic(DataProcessingOperator):
    def __init__(
        self,
        dtype=None,
        height=None,
        width=None,
        max_pixels=None,
        height_division_factor=1,
        width_division_factor=1,
    ):
        self.dtype = dtype
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def get_height_width(self, img_height, img_width):
        if self.height is None or self.width is None:
            width, height = img_width, img_height
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width

    def __call__(self, data):
        import numpy as np

        fx, fy, cx, cy, img_width, img_height = (
            data.get("fx"),
            data.get("fy"),
            data.get("cx"),
            data.get("cy"),
            data.get("width"),
            data.get("height"),
        )

        target_height, target_width = self.get_height_width(img_height, img_width)
        scale = max(target_width / img_width, target_height / img_height)
        new_height = round(img_height * scale)
        new_width = round(img_width * scale)

        fx *= scale
        fy *= scale
        cx *= scale
        cy *= scale

        cx -= (new_width - target_width) // 2
        cy -= (new_height - target_height) // 2

        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=self.dtype)


class GlobNatsort(DataProcessingOperator):
    def __init__(self, pattern):
        self.pattern = pattern

    def __call__(self, data):
        import glob
        import natsort

        files = glob.glob(os.path.join(data, self.pattern))
        files = natsort.natsorted(files)
        return files

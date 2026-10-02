from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import random

from .base import BaseProcessor


@dataclass
class StandardProcessorConfig:
    """Standard Processor Config."""

    num_frames_list: List[int] = field(default_factory=lambda: [17])
    load_depths: bool = True
    load_depth_masks: bool = False
    load_forward_flows: bool = False
    load_backward_flows: bool = False
    load_forward_flow_masks: bool = False
    load_backward_flow_masks: bool = False
    load_masks: bool = False
    load_events: bool = False
    camera_ids: Optional[List[str]] = None


class StandardVideoProcessor(BaseProcessor):
    """Standard Video Processor."""

    def __init__(self, config: StandardProcessorConfig):
        self.config = config

    def process(self, context) -> Dict[str, Any]:
        seq_meta = context.seq_meta
        dataset_type = context.dataset_type
        num_frames_total = seq_meta["num_frames"]
        target_num_frames = int(random.choice(self.config.num_frames_list))
        if target_num_frames > num_frames_total:
            raise ValueError(
                f"Sequence {context.seq_id} has {num_frames_total} frames, "
                f"cannot sample {target_num_frames} frames."
            )

        start_idx = int(random.randint(0, num_frames_total - target_num_frames))
        end_idx = start_idx + target_num_frames
        paths = context.get_paths()

        output: Dict[str, Any] = {
            "sequence_id": context.seq_id,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "num_frames": target_num_frames,
            "dataset_type": dataset_type,
        }

        if dataset_type == "monocular":
            self._process_monocular(context, output, paths, start_idx, end_idx)
        elif dataset_type == "stereo":
            self._process_stereo(context, output, paths, start_idx, end_idx)
        elif dataset_type == "multiview":
            self._process_multiview(context, output, paths, start_idx, end_idx)
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        return output


    def _process_monocular(
        self,
        context,
        output: Dict[str, Any],
        paths: Dict,
        start_idx: int,
        end_idx: int,
    ) -> None:
        output["images"] = context.load_images(None, start_idx, end_idx)

        if self.config.load_events:
            events = context.load_events(None, start_idx, end_idx)
            if events is not None:
                output["events"] = events

        self._load_optional_monocular(context, output, paths, start_idx, end_idx)
        self._load_monocular_camera_params(context, output, start_idx, end_idx)

    def _load_optional_monocular(
        self,
        context,
        output: Dict[str, Any],
        paths: Dict,
        start_idx: int,
        end_idx: int,
    ) -> None:
        if self.config.load_depths and "depths" in paths:
            output["depths"] = context.load_hdf5_slice(
                paths["depths"], "data", start_idx, end_idx
            )

        if self.config.load_depth_masks and "depth_masks" in paths:
            output["depth_masks"] = context.load_hdf5_slice(
                paths["depth_masks"], "data", start_idx, end_idx
            )

        if self.config.load_forward_flows and "forward_flows" in paths:
            output["forward_flows"] = context.load_hdf5_slice(
                paths["forward_flows"], "data", start_idx, end_idx - 1
            )

        if self.config.load_backward_flows and "backward_flows" in paths:
            output["backward_flows"] = context.load_hdf5_slice(
                paths["backward_flows"], "data", start_idx + 1, end_idx
            )

        if self.config.load_forward_flow_masks and "forward_flow_masks" in paths:
            output["forward_flow_masks"] = context.load_hdf5_slice(
                paths["forward_flow_masks"], "data", start_idx, end_idx - 1
            )

        if self.config.load_backward_flow_masks and "backward_flow_masks" in paths:
            output["backward_flow_masks"] = context.load_hdf5_slice(
                paths["backward_flow_masks"], "data", start_idx + 1, end_idx
            )

        if self.config.load_masks and "masks" in paths:
            output["masks"] = context.load_hdf5_slice(
                paths["masks"], "data", start_idx, end_idx
            )

    def _load_monocular_camera_params(
        self, context, output: Dict[str, Any], start_idx: int, end_idx: int
    ) -> None:

        try:
            params = context.get_camera_params("data", start_idx, end_idx)
            output["intrinsics"] = params["intrinsics"]
            ext = params["extrinsics"]
            output["extrinsics"] = context.normalize_extrinsics(ext, ext[0])
        except (ValueError, KeyError):

            pass


        prompt = context.get_prompt("data")
        if prompt is not None:
            output["prompt"] = prompt


    def _process_stereo(
        self,
        context,
        output: Dict[str, Any],
        paths: Dict,
        start_idx: int,
        end_idx: int,
    ) -> None:
        output["images_left"] = context.load_images("left", start_idx, end_idx)
        output["images_right"] = context.load_images("right", start_idx, end_idx)

        if self.config.load_events:
            for side in ["left", "right"]:
                events = context.load_events(side, start_idx, end_idx)
                if events is not None:
                    output[f"events_{side}"] = events

        self._load_optional_stereo(context, output, paths, start_idx, end_idx)
        self._load_stereo_camera_params(context, output, start_idx, end_idx)

    def _load_optional_stereo(
        self,
        context,
        output: Dict[str, Any],
        paths: Dict,
        start_idx: int,
        end_idx: int,
    ) -> None:
        for side in ["left", "right"]:
            if self.config.load_depths and "depths" in paths:
                output[f"depths_{side}"] = context.load_hdf5_slice(
                    paths["depths"], side, start_idx, end_idx
                )

            if self.config.load_depth_masks and "depth_masks" in paths:
                output[f"depth_masks_{side}"] = context.load_hdf5_slice(
                    paths["depth_masks"], side, start_idx, end_idx
                )

            if self.config.load_forward_flows and "forward_flows" in paths:
                output[f"forward_flows_{side}"] = context.load_hdf5_slice(
                    paths["forward_flows"], side, start_idx, end_idx - 1
                )

            if self.config.load_backward_flows and "backward_flows" in paths:
                output[f"backward_flows_{side}"] = context.load_hdf5_slice(
                    paths["backward_flows"], side, start_idx + 1, end_idx
                )

            if self.config.load_forward_flow_masks and "forward_flow_masks" in paths:
                output[f"forward_flow_masks_{side}"] = context.load_hdf5_slice(
                    paths["forward_flow_masks"], side, start_idx, end_idx - 1
                )

            if self.config.load_backward_flow_masks and "backward_flow_masks" in paths:
                output[f"backward_flow_masks_{side}"] = context.load_hdf5_slice(
                    paths["backward_flow_masks"], side, start_idx + 1, end_idx
                )

            if self.config.load_masks and "masks" in paths:
                output[f"masks_{side}"] = context.load_hdf5_slice(
                    paths["masks"], side, start_idx, end_idx
                )

    def _load_stereo_camera_params(
        self, context, output: Dict[str, Any], start_idx: int, end_idx: int
    ) -> None:
        seq_meta = context.seq_meta


        try:
            params_left = context.get_camera_params("left", start_idx, end_idx)
            params_right = context.get_camera_params("right", start_idx, end_idx)

            output["intrinsics_left"] = params_left["intrinsics"]
            output["intrinsics_right"] = params_right["intrinsics"]

            ext_l = params_left["extrinsics"]
            ext_r = params_right["extrinsics"]
            output["extrinsics_left"] = context.normalize_extrinsics(ext_l, ext_l[0])
            output["extrinsics_right"] = context.normalize_extrinsics(ext_r, ext_l[0])
        except (ValueError, KeyError):

            pass

        if "baseline" in seq_meta:
            output["baseline"] = seq_meta["baseline"]


        prompt_left = context.get_prompt("left")
        prompt_right = context.get_prompt("right")
        if prompt_left is not None:
            output["prompt_left"] = prompt_left
        if prompt_right is not None:
            output["prompt_right"] = prompt_right


    def _process_multiview(
        self,
        context,
        output: Dict[str, Any],
        paths: Dict,
        start_idx: int,
        end_idx: int,
    ) -> None:
        camera_ids = self.config.camera_ids or context.get_camera_ids()
        if not camera_ids:
            raise ValueError(
                f"No camera_ids provided for multiview sequence {context.seq_id}."
            )

        output["images"] = {
            cam_id: context.load_images(cam_id, start_idx, end_idx)
            for cam_id in camera_ids
        }

        if self.config.load_events:
            events = {}
            for cam_id in camera_ids:
                ev = context.load_events(cam_id, start_idx, end_idx)
                if ev is not None:
                    events[cam_id] = ev
            if events:
                output["events"] = events

        self._load_optional_multiview(
            context, output, paths, camera_ids, start_idx, end_idx
        )
        self._load_multiview_camera_params(
            context, output, camera_ids, start_idx, end_idx
        )
        output["camera_ids"] = camera_ids

    def _load_optional_multiview(
        self,
        context,
        output: Dict[str, Any],
        paths: Dict,
        camera_ids: List[str],
        start_idx: int,
        end_idx: int,
    ) -> None:
        def _load_dict(key: str, flow: bool = False, is_forward: bool = True):
            if key not in paths:
                return None
            h5_path = paths[key]
            result = {}
            for cam_id in camera_ids:
                try:
                    if flow:
                        result[cam_id] = context.load_flow_slice(
                            h5_path, cam_id, start_idx, end_idx, is_forward=is_forward
                        )
                    else:
                        result[cam_id] = context.load_hdf5_slice(
                            h5_path, cam_id, start_idx, end_idx
                        )
                except KeyError:
                    continue
            return result if result else None

        if self.config.load_depths:
            data = _load_dict("depths")
            if data:
                output["depths"] = data

        if self.config.load_depth_masks:
            data = _load_dict("depth_masks")
            if data:
                output["depth_masks"] = data

        if self.config.load_forward_flows:
            data = _load_dict("forward_flows", flow=True, is_forward=True)
            if data:
                output["forward_flows"] = data

        if self.config.load_backward_flows:
            data = _load_dict("backward_flows", flow=True, is_forward=False)
            if data:
                output["backward_flows"] = data

        if self.config.load_forward_flow_masks:
            data = _load_dict("forward_flow_masks", flow=True, is_forward=True)
            if data:
                output["forward_flow_masks"] = data

        if self.config.load_backward_flow_masks:
            data = _load_dict("backward_flow_masks", flow=True, is_forward=False)
            if data:
                output["backward_flow_masks"] = data

        if self.config.load_masks:
            data = _load_dict("masks")
            if data:
                output["masks"] = data

    def _load_multiview_camera_params(
        self,
        context,
        output: Dict[str, Any],
        camera_ids: List[str],
        start_idx: int,
        end_idx: int,
    ) -> None:

        intrinsics_dict = {}
        extrinsics_dict = {}

        for cam_id in camera_ids:
            try:
                params = context.get_camera_params(cam_id, start_idx, end_idx)
                intrinsics_dict[cam_id] = params["intrinsics"]
                extrinsics_dict[cam_id] = params["extrinsics"]
            except (ValueError, KeyError):

                continue

        if intrinsics_dict:
            output["intrinsics"] = intrinsics_dict

        if extrinsics_dict:
            output["extrinsics"] = context.normalize_extrinsics_multiview(
                extrinsics_dict
            )


        prompts_dict = {}
        for cam_id in camera_ids:
            prompt = context.get_prompt(cam_id)
            if prompt is not None:
                prompts_dict[cam_id] = prompt

        if prompts_dict:
            output["prompts"] = prompts_dict

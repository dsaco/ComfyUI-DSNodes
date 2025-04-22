import torch

import comfy.sd

import folder_paths


class FluxCombinationNode:

    @classmethod
    def INPUT_TYPES(s):
        """
        Return a dictionary which contains config for all input fields.
        Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
        Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
        The type can be a list for selection.

        Returns: `dict`:
            - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
            - Value input_fields (`dict`): Contains input fields config:
                * Key field_name (`string`): Name of a entry-point method's argument
                * Value field_config (`tuple`):
                    + First value is a string indicate the type of field or a list for selection.
                    + Second value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                ),
                "clip_name1": (folder_paths.get_filename_list("text_encoders"),),
                "clip_name2": (folder_paths.get_filename_list("text_encoders"),),
            },
            "optional": {
                "type": (
                    ["sdxl", "sd3", "flux", "hunyuan_video", "hidream"],
                    {"advanced": True, "default": "flux"},
                ),
                "device": (["default", "cpu"], {"advanced": True}),
            },
        }

    RETURN_TYPES = (
        "MODEL",
        "CLIP",
    )
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "start"

    # OUTPUT_NODE = False

    CATEGORY = "Dsaco/FluxCombination"

    def load_unet(self, unet_name, weight_dtype):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model,)

    def load_clip(self, clip_name1, clip_name2, type="flux", device="default"):
        clip_type = getattr(
            comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION
        )

        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = (
                torch.device("cpu")
            )

        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path1, clip_path2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options,
        )
        return (clip,)

    def start(
        self,
        unet_name,
        weight_dtype,
        clip_name1,
        clip_name2,
        type="flux",
        device="default",
    ):
        model = self.load_unet(unet_name, weight_dtype)
        clip = self.load_clip(clip_name1, clip_name2, type, device)
        return (
            model,
            clip,
        )

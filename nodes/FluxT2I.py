import torch
import sys
import nodes
import node_helpers
import latent_preview

import comfy.utils
import comfy.sd
import comfy.model_sampling
import comfy.samplers
import comfy.sample
import comfy.model_management
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO

import folder_paths


class Guider_Basic(comfy.samplers.CFGGuider):
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})


class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = (
            input_latent["batch_index"] if "batch_index" in input_latent else None
        )
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)


class FluxT2INode:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @staticmethod
    def vae_list():
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False
        f1_taesd_enc = False
        f1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."):
                sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."):
                sd3_taesd_enc = True
            elif v.startswith("taef1_encoder."):
                f1_taesd_dec = True
            elif v.startswith("taef1_decoder."):
                f1_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        if f1_taesd_dec and f1_taesd_enc:
            vaes.append("taef1")
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(
            filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes)
        )
        decoder = next(
            filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes)
        )

        enc = comfy.utils.load_torch_file(
            folder_paths.get_full_path_or_raise("vae_approx", encoder)
        )
        for k in enc:
            sd["taesd_encoder.{}".format(k)] = enc[k]

        dec = comfy.utils.load_torch_file(
            folder_paths.get_full_path_or_raise("vae_approx", decoder)
        )
        for k in dec:
            sd["taesd_decoder.{}".format(k)] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305)
            sd["vae_shift"] = torch.tensor(0.0609)
        elif name == "taef1":
            sd["vae_scale"] = torch.tensor(0.3611)
            sd["vae_shift"] = torch.tensor(0.1159)
        return sd

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
                ),
                "vae_name": (s.vae_list(),),
                "clip_name1": (folder_paths.get_filename_list("text_encoders"),),
                "clip_name2": (folder_paths.get_filename_list("text_encoders"),),
                "type": (
                    ["sdxl", "sd3", "flux", "hunyuan_video", "hidream"],
                    {"advanced": True, "default": "flux"},
                ),
                "device": (["default", "cpu"], {"advanced": True}),
                "guidance": (
                    "FLOAT",
                    {
                        "advanced": True,
                        "default": 3.5,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                    },
                ),
                "max_shift": (
                    "FLOAT",
                    {
                        "advanced": True,
                        "default": 1.15,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                    },
                ),
                "base_shift": (
                    "FLOAT",
                    {
                        "advanced": True,
                        "default": 0.5,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": nodes.MAX_RESOLUTION,
                        "step": 8,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": nodes.MAX_RESOLUTION,
                        "step": 8,
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "advanced": True,
                        "default": 1,
                        "min": 1,
                        "max": 4096,
                        "tooltip": "The number of latent images in the batch.",
                    },
                ),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "noise_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "text": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The text to be encoded.",
                    },
                ),
            },
            "optional": {},
        }

    # RETURN_TYPES = ("MODEL", "CLIP", "SIGMAS", "LATENT")
    # RETURN_NAMES = ("unet", "clip", "sigmas", "latent")

    RETURN_TYPES = ("LATENT", "VAE", "LATENT")
    RETURN_NAMES = ("output", "vae", "denoised_output")

    FUNCTION = "start"

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
        return model

    def load_vae(self, vae_name):
        if vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
            sd = self.load_taesd(vae_name)
        else:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=sd)
        return vae

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
        return clip

    def patch(self, model, max_shift, base_shift, width, height):
        m = model.clone()

        x1 = 256
        x2 = 4096
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        shift = (width * height / (8 * 8 * 2 * 2)) * mm + b

        sampling_base = comfy.model_sampling.ModelSamplingFlux
        sampling_type = comfy.model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift)
        m.add_object_patch("model_sampling", model_sampling)
        return m

    def get_sigmas(self, model, scheduler, steps, denoise):
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            total_steps = int(steps / denoise)

        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), scheduler, total_steps
        ).cpu()
        sigmas = sigmas[-(steps + 1) :]
        return sigmas

    def generateLatent(self, width, height, batch_size=1):
        latent = torch.zeros(
            [batch_size, 4, height // 8, width // 8], device=self.device
        )
        return {"samples": latent}

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = (
            input_latent["batch_index"] if "batch_index" in input_latent else None
        )
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)

    def get_sampler(self, sampler_name):
        sampler = comfy.samplers.sampler_object(sampler_name)
        return sampler

    def encode(self, clip, text):
        if clip is None:
            raise RuntimeError(
                "ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model."
            )
        tokens = clip.tokenize(text)
        return clip.encode_from_tokens_scheduled(tokens)

    def append(self, conditioning, guidance):
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        return c

    def get_guider(self, model, conditioning):
        guider = Guider_Basic(model)
        guider.set_conds(conditioning)
        return guider

    def sample(self, noise, guider, sampler, sigmas, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher, latent_image
        )
        latent["samples"] = latent_image

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(
            guider.model_patcher, sigmas.shape[-1] - 1, x0_output
        )

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(
            noise.generate_noise(latent),
            latent_image,
            sampler,
            sigmas,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=noise.seed,
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = guider.model_patcher.model.process_latent_out(
                x0_output["x0"].cpu()
            )
        else:
            out_denoised = out
        return (out, out_denoised)

    def start(
        self,
        unet_name,
        weight_dtype,
        vae_name,
        clip_name1,
        clip_name2,
        type,
        device,
        max_shift,
        base_shift,
        width,
        height,
        scheduler,
        steps,
        denoise,
        batch_size,
        noise_seed,
        sampler_name,
        text,
        guidance,
    ):
        vae = self.load_vae(vae_name)
        model = self.load_unet(unet_name, weight_dtype)
        clip = self.load_clip(clip_name1, clip_name2, type, device)
        model = self.patch(model, max_shift, base_shift, width, height)
        sigmas = self.get_sigmas(model, scheduler, steps, denoise)
        latent = self.generateLatent(width, height, batch_size)

        noise = Noise_RandomNoise(noise_seed)

        sampler = self.get_sampler(sampler_name)

        conditioning = self.encode(clip, text)
        conditioning = self.append(conditioning, guidance)

        guider = self.get_guider(model, conditioning)

        out, out_denoised = self.sample(noise, guider, sampler, sigmas, latent)
        return (out, vae, out_denoised)

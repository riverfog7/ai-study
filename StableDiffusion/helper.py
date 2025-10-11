import torch
import os

from collections.abc import Callable
from tqdm import tqdm
from dataclasses import dataclass
from PIL import Image
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import PNDMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from typing import List, Tuple, Optional, Union, Literal

dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DiffusionResult:
    image: Image.Image
    latents: torch.Tensor
    image_list: List[Image.Image]
    latent_list: List[torch.Tensor]


class CustomDiffusion:
    def __init__(self, model_id: str = "CompVis/stable-diffusion-v1-4", device: torch.device = dev,
                 latent_scale_factor: int = 8, height: int = 512, width: int = 512, batch_size: int = 1,
                 guidance_scale: float = 7.5):
        # constants
        self.LATENT_SCALE_FACTOR = latent_scale_factor
        self.BATCH_SIZE = batch_size
        self.HEIGHT = height
        self.WIDTH = width
        self.GUIDANCE_SCALE = guidance_scale
        self.SCALING_FACTOR: Optional[float] = None

        self.model_id: str = model_id
        self.device: torch.device = device

        # model components
        self.pipe: Optional[StableDiffusionPipeline] = None
        self.vae: Optional[AutoencoderKL] = None
        self.tokenizer: Optional[CLIPTokenizer] = None
        self.text_encoder: Optional[CLIPTextModel] = None
        self.unet: Optional[UNet2DConditionModel] = None
        self.scheduler: Optional[PNDMScheduler] = None

        # runtime variables
        self.latents: Optional[torch.Tensor] = None
        self.image: Optional[Image.Image] = None
        self.latent_list: List[torch.Tensor] = []
        self.image_list: List[Image.Image] = []

        self.__load_model()

    def __load_model(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16,
                                                            cache_dir=os.environ['HF_HOME'], safety_checker=None)
        self.vae = self.pipe.vae.to(self.device)
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder.to(self.device)
        self.unet = self.pipe.unet.to(self.device)
        self.scheduler = self.pipe.scheduler

        self.SCALING_FACTOR = self.vae.config.scaling_factor

    def encode_text(self, txt: str) -> torch.Tensor:
        assert self.tokenizer is not None, "Tokenizer is not loaded"
        assert self.text_encoder is not None, "Text encoder is not loaded"

        with torch.no_grad():
            tokenized = self.tokenizer(txt, return_tensors='pt', padding="max_length",
                                       max_length=self.tokenizer.model_max_length, truncation=True).input_ids.to(self.device)
            result = self.text_encoder(tokenized)[0].to(self.device)
            del tokenized
        return result

    def __sanitize_input(self, input_positive: Union[str, torch.Tensor], input_negative: Optional[Union[str, torch.Tensor]]) -> torch.Tensor:
        if isinstance(input_positive, str):
            text_embeddings = self.encode_text(input_positive)
        elif isinstance(input_positive, torch.Tensor):
            text_embeddings = input_positive.to(self.device)
        else:
            raise ValueError("Invalid type for input_positive")

        if input_negative is None:
            input_negative = ""
        if isinstance(input_negative, str):
            negative_embeddings = self.encode_text(input_negative)
        elif isinstance(input_negative, torch.Tensor):
            negative_embeddings = input_negative.to(self.device)
        else:
            raise ValueError("Invalid type for input_negative")
        if text_embeddings.shape != negative_embeddings.shape:
            raise ValueError("Positive and negative embeddings must have the same shape")
        return torch.cat([negative_embeddings, text_embeddings])

    def __prepare_diffusion(self, steps: int = 50, seed: int = 42):
        generator = torch.Generator(self.device).manual_seed(seed)
        self.latents = torch.randn(
            (
                self.BATCH_SIZE,
                self.unet.config.in_channels,
                self.HEIGHT // self.LATENT_SCALE_FACTOR,
                self.WIDTH // self.LATENT_SCALE_FACTOR,
            ),
            generator=generator,
            device=self.device,
            dtype=torch.float16,
        ).to(self.device)
        self.scheduler.set_timesteps(steps)
        self.latents = self.latents * self.scheduler.init_noise_sigma

    def __latent_diffusion(self, text_embeddings: torch.Tensor, print_steps: bool = True, decode_every_step: bool = False, callback_fn: Callable = None, callback_args: Tuple[Literal["image", "latent", "total_steps", "current_step"]] = None) -> torch.Tensor:
        assert self.unet is not None, "UNet model is not loaded"
        assert self.scheduler is not None, "Scheduler is not loaded"
        assert self.latents is not None, "Latents are not initialized"

        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Diffusion", disable=not print_steps)):
            latent_model_input = torch.cat([self.latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep=t)

            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

            self.latents = self.scheduler.step(
                noise_pred, t, self.latents
            ).prev_sample
            self.latent_list.append(self.latents.cpu().clone())

            if decode_every_step:
                self.image_list.append(self.__vae_decode(self.latents))
                self.image = self.image_list[-1]

            if callback_fn is not None and callback_args is not None:
                args = []
                for arg in callback_args:
                    if arg == "image":
                        args.append(self.image)
                    elif arg == "latent":
                        args.append(self.latents)
                    elif arg == "total_steps":
                        args.append(len(self.scheduler.timesteps))
                    elif arg == "current_step":
                        args.append(i + 1)
                    else:
                        raise ValueError("Invalid callback argument")
                callback_fn(*args)
        return self.latents

    def __vae_decode(self, latents: torch.Tensor) -> Image.Image:
        assert self.vae is not None, "VAE model is not loaded"
        assert self.SCALING_FACTOR is not None, "Scaling factor is not set"

        latents = 1 / self.SCALING_FACTOR * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(image)
        return image

    def __reset_state(self):
        self.latents = None
        self.image = None
        self.latent_list = []
        self.image_list = []

    def generate(self, prompt: Union[str, torch.Tensor], negative_prompt: Optional[Union[str, torch.Tensor]], steps: int = 50, seed: int = 42, print_steps: bool = True, decode_every_step: bool = False, callback_fn: Callable = None, callback_args: Tuple[Literal["image", "latent", "total_steps", "current_step"]] = None) -> DiffusionResult:
        self.__reset_state()
        text_embeddings = self.__sanitize_input(prompt, negative_prompt)
        self.__prepare_diffusion(steps, seed)
        final_latents = self.__latent_diffusion(text_embeddings, print_steps, decode_every_step, callback_fn, callback_args)
        self.image = self.__vae_decode(final_latents)

        result = DiffusionResult(
            image=self.image,
            latents=final_latents,
            image_list=self.image_list,
            latent_list=self.latent_list
        )
        return result

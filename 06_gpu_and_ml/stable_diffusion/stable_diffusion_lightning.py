from pathlib import Path

import modal

stub = modal.Stub("sd-demo")

image = modal.Image.debian_slim().pip_install(
    "diffusers", "transformers", "accelerate"
)

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = (
    "sdxl_lightning_1step_unet_x0.safetensors"
)  # Use the correct ckpt for your step setting!


with image.imports():
    import io

    import torch
    from diffusers import (
        EulerDiscreteScheduler,
        StableDiffusionXLPipeline,
        UNet2DConditionModel,
    )
    from fastapi import Response
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

# Ensure using the same inference steps as the loaded model and CFG set to 0.


@stub.cls(
    image=image,
    gpu="h100",
    _experimental_boost=True,
    timeout=60,
    container_idle_timeout=300,
    concurrency_limit=15,
)
class Model:
    @modal.build()
    @modal.enter()
    def load_weights(self):
        # Load model.
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(
            "cuda", torch.float16
        )
        unet.load_state_dict(
            load_file(hf_hub_download(repo, ckpt), device="cuda")
        )
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base, unet=unet, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")

        # Ensure sampler uses "trailing" timesteps and "sample" prediction type.
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config,
            timestep_spacing="trailing",
            prediction_type="sample",
        )

    @modal.web_endpoint()
    def inference(
        self,
        prompt="A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
    ):
        print("Received", prompt)
        image = self.pipe(
            prompt, num_inference_steps=1, guidance_scale=0
        ).images[0]

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        return Response(content=buffer.getvalue(), media_type="image/jpeg")


frontend_path = Path(__file__).parent / "frontend"

web_image = modal.Image.debian_slim().pip_install("jinja2")


@stub.function(
    mounts=[modal.Mount.from_local_dir(frontend_path, remote_path="/assets")],
    image=web_image,
    allow_concurrent_inputs=20,
    keep_warm=1,
)
@modal.asgi_app(custom_domains=["potatoes.ai"])
def app():
    import fastapi.staticfiles
    from fastapi import FastAPI
    from jinja2 import Template

    web_app = FastAPI()

    with open("/assets/index.html", "r") as f:
        template_html = f.read()

    template = Template(template_html)

    with open("/assets/index.html", "w") as f:
        html = template.render(inference_url=Model.inference.web_url)
        f.write(html)

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )

    return web_app

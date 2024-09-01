import io

import modal

app = modal.App("flux-subapp")

flux_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "libglib2.0-0",
        "libsm6",
        "libxrender1",
        "libxext6",
        "ffmpeg",
        "libgl1",
    )
    .pip_install(
        "invisible_watermark==0.2.0",
        "transformers==4.44.0",
        "accelerate==0.33.0",
        "safetensors==0.4.4",
        "sentencepiece==0.2.0",
        "diffusers==0.30.1",
        "slack-sdk",
    )
)


with flux_image.imports():
    import torch
    from diffusers import FluxPipeline


@app.cls(
    gpu="A100",
    image=flux_image,
    secrets=[
        modal.Secret.from_name("flux-bot-secret"),
        modal.Secret.from_name("huggingface"),
    ],
    container_idle_timeout=300,
)
class Flux:
    @modal.build()
    @modal.enter()
    def enter(self):
        from huggingface_hub import snapshot_download
        from transformers.utils import move_cache

        snapshot_download("black-forest-labs/FLUX.1-dev")

        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
        )
        self.pipe.to("cuda")
        move_cache()

    @modal.method()
    async def inference(self, prompt: str):
        image = self.pipe(
            prompt,
            output_type="pil",
            width=1024,
            height=1024,
            max_sequence_length=512,
            num_inference_steps=28,  # use a larger number if you are using [dev], smaller for [schnell]
        ).images[0]

        with io.BytesIO() as buf:
            image.save(buf, format="JPEG")
            return buf.getvalue()

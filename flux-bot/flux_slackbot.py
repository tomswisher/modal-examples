import io
import logging
import os

import fastapi
import modal

app = modal.App("flux-bot")

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

slackbot_image = modal.Image.debian_slim().pip_install(
    "slack-sdk", "slack-bolt", "openai", "langchain"
)

with flux_image.imports():
    import slack_sdk
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
            width=512,
            height=512,
            num_inference_steps=4,  # use a larger number if you are using [dev], smaller for [schnell]
        ).images[0]

        with io.BytesIO() as buf:
            image.save(buf, format="JPEG")
            return buf.getvalue()


@app.function(
    keep_warm=1,
    image=slackbot_image,
    secrets=[modal.Secret.from_name("flux-bot-secret")],
    allow_concurrent_inputs=100,
)
@modal.asgi_app(label="flux")
def entrypoint():
    import slack_bolt
    from slack_bolt.adapter.fastapi import SlackRequestHandler

    slack_app = slack_bolt.App(
        signing_secret=os.environ["SLACK_SIGNING_SECRET"],
        token=os.environ["SLACK_BOT_TOKEN"],
    )

    fastapi_app = fastapi.FastAPI()
    handler = SlackRequestHandler(slack_app)

    @slack_app.event("url_verification")
    def handle_url_verification(body, logger):
        challenge = body.get("challenge")
        return {"challenge": challenge}

    @slack_app.shortcut("open_flux_modal")
    def open_modal(ack, shortcut, client):
        ack()

        thread_ts = (
            shortcut["message"]["thread_ts"]
            if "thread_ts" in shortcut["message"]
            else shortcut["message"]["ts"]
        )

        initial_value = (
            f'a cat holdin a sign that says "{shortcut["message"]["text"]}"'
        )

        client.views_open(
            trigger_id=shortcut["trigger_id"],
            view={
                "type": "modal",
                "callback_id": "modal_submission",
                "title": {"type": "plain_text", "text": "Flux!"},
                "submit": {"type": "plain_text", "text": "Submit"},
                "blocks": [
                    {
                        "type": "input",
                        "block_id": "input_block",
                        "label": {
                            "type": "plain_text",
                            "text": "Enter some text",
                        },
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "text_input",
                            "initial_value": initial_value,
                        },
                    }
                ],
                "private_metadata": f"{thread_ts},{shortcut['channel']['id']},{shortcut['user']['id']}",
            },
        )

    @slack_app.view("modal_submission")
    def handle_submission(ack, body, client, view):
        ack()

        thread_ts, channel_id, user_id = view["private_metadata"].split(",")

        user_input = view["state"]["values"]["input_block"]["text_input"][
            "value"
        ]
        img_bytes = Flux().inference.remote(user_input)

        try:
            client.files_upload(
                channels=channel_id,
                thread_ts=thread_ts,
                title=user_input,
                content=img_bytes,
            )
        except slack_sdk.errors.SlackApiError as e:
            if e.response["error"] == "not_in_channel":
                client.conversations_join(channel=channel_id)
                client.files_upload(
                    channels=channel_id,
                    thread_ts=thread_ts,
                    title=user_input,
                    content=img_bytes,
                )
            elif e.response["error"] == "channel_not_found":
                client.chat_postMessage(
                    channel=user_id,
                    text="Channel not found, please invite the bot to the channel first.",
                )
            else:
                raise e

    @slack_app.function("generate")
    def handle_sample_function_event(
        inputs: dict,
        say: slack_bolt.Say,
        fail: slack_bolt.Fail,
        logger: logging.Logger,
    ):
        user_id = inputs["user_id"]

        try:
            say(
                channel=user_id,  # sending a DM to this user
                text="Click the button to signal the function has completed",
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "Click the button to signal the function has completed",
                        },
                        "accessory": {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "Complete function",
                            },
                            "action_id": "sample_click",
                        },
                    }
                ],
            )
        except Exception as e:
            logger.exception(e)
            fail(f"Failed to handle a function request (error: {e})")

    @fastapi_app.post("/")
    async def root(request: fastapi.Request):
        return await handler.handle(request)

    return fastapi_app

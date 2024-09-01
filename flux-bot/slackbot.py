import logging
import os

import fastapi
import modal

from .flux import Flux, app as model_app

app = modal.App("flux-bot")
app.include(model_app)

slackbot_image = modal.Image.debian_slim().pip_install(
    "slack-sdk", "slack-bolt", "langchain", "langchain-anthropic"
)

with slackbot_image.imports():
    import slack_sdk

    from .prompter import generate_image_prompt, generate_meme_idea


@app.function(
    keep_warm=1,
    image=slackbot_image,
    secrets=[
        modal.Secret.from_name("flux-bot-secret"),
        modal.Secret.from_name("anthropic-secret"),
    ],
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
        try:
            conversation_history = client.conversations_replies(
                channel=shortcut["channel"]["id"],
                ts=thread_ts,
            )
        except slack_sdk.errors.SlackApiError as e:
            if e.response["error"] == "not_in_channel":
                client.conversations_join(channel=shortcut["channel"]["id"])
                conversation_history = client.conversations_replies(
                    channel=shortcut["channel"]["id"],
                    ts=thread_ts,
                )
            elif e.response["error"] == "channel_not_found":
                client.chat_postMessage(
                    channel=shortcut["user"]["id"],
                    text="Channel not found, please invite the bot to the channel first.",
                )
            else:
                raise e

        messages = conversation_history["messages"]
        root_message = messages[0]["text"] if messages else ""

        if len(messages) > 1:
            replies = [msg["text"] for msg in messages[1:]]
        else:
            replies = []

        thread_context = f"Root message: {root_message}\n"
        if replies:
            thread_context += "Replies:\n" + "\n".join(replies)

        print(f"Thread context: {thread_context}")

        initial_value = (
            generate_meme_idea(thread_context).replace("'", "").replace('"', "")
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
                            "text": "What's funny about this?",
                        },
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "text_input",
                            "initial_value": initial_value,
                            "multiline": True,
                        },
                    }
                ],
                "private_metadata": f"{thread_ts},{shortcut['channel']['id']}",
            },
        )

    @slack_app.view("modal_submission")
    def handle_submission(ack, body, client, view):
        ack()

        thread_ts, channel_id = view["private_metadata"].split(",")

        user_input = view["state"]["values"]["input_block"]["text_input"][
            "value"
        ]

        prompt = generate_image_prompt(user_input)

        print(f"Input: {user_input}\n\nPrompt: {prompt}")

        img_bytes = Flux().inference.remote(prompt)

        client.files_upload(
            channels=channel_id,
            thread_ts=thread_ts,
            title=user_input,
            content=img_bytes,
        )

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


@app.function(
    image=slackbot_image, secrets=[modal.Secret.from_name("anthropic-secret")]
)
def test_prompt():
    print(generate_image_prompt("Modal makes things too easy"))

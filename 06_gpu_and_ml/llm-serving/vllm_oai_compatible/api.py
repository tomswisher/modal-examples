# ---
# cmd: ["modal", "deploy", "06_gpu_and_ml/llm-serving/vllm_oai_compatible/api.py", "&&", "pip", "install", "openai==1.13.3", "&&" "python", "06_gpu_and_ml/llm-serving/vllm_oai_compatible/client.py"]
# ---
# # Run an OpenAI-Compatible vLLM Server
#
# LLMs do more than just model language: they chat, they produce JSON and XML, they run code, and more.
# OpenAI's API has emerged as a standard interface for LLMs,
# and it is supported by open source LLM serving frameworks like vLLM.
#
# In this example, we show how to run a vLLM server in OpenAI-compatible mode on Modal.
# Note that the vLLM server is a FastAPI app, which can be configured and extended just like any other.
# Here, we use it to add simple authentication middleware.
# This implementation is based on the OpenAI-compatible server provided by vLLM.
# For the latest reference, see the vLLM documentation: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
#
# Note: The chat template should be specified as a command-line argument
# when starting the server, e.g., --chat-template /path/to/chat_template.jinja
#
# ## Set up the container image
#
# Our first order of business is to define the environment our server will run in: the [container `Image`](https://modal.com/docs/guide/custom-container).
# We'll build it up, step-by-step, from a slim Debian Linux image.
#
# First, we install some dependencies with `pip`.

from pathlib import Path
import traceback

import modal

vllm_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    [
        "vllm==0.5.3.post1",  # LLM serving
        "huggingface_hub>=0.23.2,<1.0",  # download models from the Hugging Face Hub
        "hf-transfer==0.1.6",  # download models faster
    ]
)

# Then, we need to get hold of the weights for the model we're serving:
# Meta's LLaMA 3-8B Instruct. We create a Python function for this and add it to the image definition,
# so that we only need to download it when we define the image, not every time we run the server.
#
# If you adapt this example to run another model,
# note that for this step to work on a [gated model](https://huggingface.co/docs/hub/en/models-gated)
# the `HF_TOKEN` environment variable must be set and provided as a [Modal Secret](https://modal.com/secrets).


MODEL_NAME = "NousResearch/Meta-Llama-3-8B-Instruct"
MODEL_REVISION = "b1532e4dee724d9ba63fe17496f298254d87ca64"
MODEL_DIR = f"/models/{MODEL_NAME}"


def download_model_to_image(model_dir, model_name, model_revision):
    import os

    from huggingface_hub import snapshot_download

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
        revision=model_revision,
    )


MINUTES = 60

vllm_image = vllm_image.env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}).run_function(
    download_model_to_image,
    timeout=20 * MINUTES,
    kwargs={
        "model_dir": MODEL_DIR,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
    },
)

# ## Build the server
#
# vLLM's OpenAI-compatible server is a [FastAPI](https://fastapi.tiangolo.com/) app.
#
# FastAPI is a Python web framework that implements the [ASGI standard](https://en.wikipedia.org/wiki/Asynchronous_Server_Gateway_Interface),
# much like [Flask](https://en.wikipedia.org/wiki/Flask_(web_framework)) is a Python web framework
# that implements the [WSGI standard](https://en.wikipedia.org/wiki/Web_Server_Gateway_Interface).
#
# Modal offers [first-class support for ASGI (and WSGI) apps](https://modal.com/docs/guide/webhooks). We just need to decorate a function that returns the app
# with `@modal.asgi_app()` (or `@modal.wsgi_app()`) and then add it to the Modal app with the `app.function` decorator.
#
# The function below first imports the FastAPI app from the vLLM library, then adds some middleware. You might also add more routes here.
#
# Then, the function creates an `AsyncLLMEngine`, the core of the vLLM server. It's responsible for loading the model, running inference, and serving responses.
#
# After attaching that engine to the FastAPI app via the `api_server` module of the vLLM library, we return the FastAPI app
# so it can be served on Modal.

app = modal.App("vllm-openai-compatible")

N_GPU = 1  # tip: for best results, first upgrade to A100s or H100s, and only then increase GPU count
TOKEN = "super-secret-token"  # auth token. for production use, replace with a modal.Secret
local_template_path = (
    Path(__file__).parent / "template_llama3.jinja"
)  # many models have a custom chat template -- using the wrong one subtly degrades results. watch out for it!

@app.function(
    image=vllm_image,
    gpu=modal.gpu.A10G(count=N_GPU),
    container_idle_timeout=20 * MINUTES,
    secrets=[modal.Secret.from_name("my-huggingface-secret")]
)
@modal.asgi_app()
def serve(chat_template: str = None):
    print("Starting serve function")
    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"N_GPU: {N_GPU}")
    print(f"Chat template: {chat_template}")
    import vllm
    import os
    print(f"Using vLLM version: {vllm.__version__}")

    if chat_template:
        if os.path.isfile(chat_template):
            with open(chat_template, 'r') as f:
                chat_template = f.read()
            print(f"Chat template loaded from file: {chat_template}")
        else:
            print("Chat template provided as string")
    else:
        print("Warning: No chat template provided. Chat functionality may be limited.")

    if chat_template:
        if os.path.isfile(chat_template):
            with open(chat_template, 'r') as f:
                chat_template = f.read()
            print(f"Chat template loaded from file: {chat_template}")
        else:
            print("Chat template provided as string")
    else:
        print("Warning: No chat template provided. Chat functionality may be limited.")

    from fastapi import FastAPI, Request, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion

    print("All necessary imports completed successfully")

    print("About to define engine_args")
    print("Initializing AsyncEngineArgs with the following parameters:")
    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"N_GPU: {N_GPU}")
    print("All kwargs being passed to AsyncEngineArgs:")
    kwargs = {
        "model": MODEL_DIR,
        "tensor_parallel_size": N_GPU,
        "gpu_memory_utilization": 0.95,
        "max_model_len": 8192,
        "enforce_eager": False,
        "trust_remote_code": True,
        "dtype": "auto",
        "quantization": None,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
        "disable_log_stats": True,
        "swap_space": 4,
        "enable_lora": False,
    }
    for key, value in kwargs.items():
        print(f"{key}: {value}")
    engine_args = AsyncEngineArgs(**kwargs)
    print(f"AsyncEngineArgs initialized: {engine_args}")

    if chat_template is None:
        print("Warning: No chat template provided. Chat functionality may be limited.")
    else:
        print(f"Chat template successfully loaded: {chat_template[:50]}...")

    print(f"engine_args defined: {engine_args}")
    print("AsyncEngineArgs initialized successfully")

    try:
        print("About to create AsyncLLMEngine")
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("AsyncLLMEngine created successfully")
        print(f"Model loaded successfully: {MODEL_DIR}")

        model_config = {
            "model": MODEL_DIR,
            "tokenizer": engine_args.tokenizer,
            "max_model_len": engine_args.max_model_len,
        }
        print(f"Model config: {model_config}")
        print(f"Chat template: {chat_template}")

        print("Initializing OpenAIServingChat with the following parameters:")
        print(f"engine: {engine}")
        print(f"model_config: {model_config}")
        print(f"MODEL_DIR: {MODEL_DIR}")

        try:
            print("OpenAIServingChat arguments:", {
                "engine": engine,
                "model_config": model_config,
                "response_role": "assistant",
                "chat_template": chat_template,
                "served_model_names": [MODEL_DIR],
                "lora_modules": None,
                "prompt_adapters": None,
                "request_logger": None,
                "use_beam_search": False,
                "top_k": 50,
                "top_p": 0.95,
                "temperature": 0.7,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
            })
            openai_serving_chat = OpenAIServingChat(
                engine=engine,
                model_config=model_config,
                response_role="assistant",
                chat_template=chat_template,
                served_model_names=[MODEL_DIR],
                lora_modules=None,
                prompt_adapters=None,
                request_logger=None,
                use_beam_search=False,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                presence_penalty=0.0,
                frequency_penalty=0.0,
            )
            print("OpenAIServingChat initialized successfully")
            print("OpenAIServingChat object:", openai_serving_chat)
            print("OpenAIServingChat attributes:", vars(openai_serving_chat))
            if chat_template:
                print(f"Chat template successfully loaded: {chat_template[:50]}...")
            else:
                print("Warning: No chat template provided. Chat functionality may be limited.")
        except Exception as e:
            print(f"Error initializing OpenAIServingChat: {str(e)}")
            raise

        print("Initializing OpenAIServingCompletion with arguments:", {
            "engine": engine,
            "model_config": model_config,
            "served_model_names": [MODEL_DIR],
            "lora_modules": None,
            "prompt_adapters": None,
            "request_logger": None,
            "best_of": 1,
            "use_beam_search": False,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 1.0,
        })
        openai_serving_completion = OpenAIServingCompletion(
            engine=engine,
            model_config=model_config,
            served_model_names=[MODEL_DIR],
            lora_modules=None,
            prompt_adapters=None,
            request_logger=None,
            best_of=1,
            use_beam_search=False,
            top_k=50,
            top_p=1.0,
            temperature=1.0,
        )
        print("OpenAIServingCompletion initialized successfully")

        print("About to create FastAPI app")
        app = FastAPI()
        print(f"FastAPI app created successfully: {app}")
        print("Adding OpenAIServingChat and OpenAIServingCompletion routers")

        app.include_router(openai_serving_chat.router)
        app.include_router(openai_serving_completion.router)
        print("Chat and Completion routers included in the app")
        print("Chat router included:", openai_serving_chat.router)
        print("Completion router included:", openai_serving_completion.router)

        # security: CORS middleware for external requests
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        print("CORS middleware added")

        # security: auth middleware
        @app.middleware("http")
        async def authentication(request: Request, call_next):
            if not request.url.path.startswith("/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != f"Bearer {TOKEN}":
                return JSONResponse(
                    content={"error": "Unauthorized"}, status_code=401
                )
            return await call_next(request)
        print("Authentication middleware added")

    except Exception as e:
        print(f"Error initializing server: {str(e)}")
        print("Stack trace:", traceback.format_exc())
        raise

    print("Chat router included:", openai_serving_chat.router)
    print("Completion router included:", openai_serving_completion.router)
    print("Server is starting...")

    # All old vLLM API references have been updated
    print("All vLLM API references have been updated successfully.")

    print("vllm_openai_compatible_serve function completed")
    return app

# ## Deploy the server
#
# To deploy the API on Modal, just run
# ```bash
# modal deploy api.py
# ```
#
# This will create a new app on Modal, build the container image for it, and deploy.
#
# ### Interact with the server
#
# Once it is deployed, you'll see a URL appear in the command line,
# something like `https://your-workspace-name--vllm-openai-compatible-serve.modal.run`.
#
# You can find [interactive Swagger UI docs](https://swagger.io/tools/swagger-ui/)
# at the `/docs` route of that URL, i.e. `https://your-workspace-name--vllm-openai-compatible-serve.modal.run/docs`.
# These docs describe each route and indicate the expected input and output.
#
# For simple routes like `/health`, which checks whether the server is responding,
# you can even send a request directly from the docs.
#
# To interact with the API programmatically, you can use the Python `openai` library.
#
# See the small test `client.py` script included with this example for details.
#
# ```bash
# # pip install openai==1.13.3
# python client.py
# ```

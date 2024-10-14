# ---
# cmd: ["modal", "run", "06_gpu_and_ml.sam.sam_example"]
# deploy: true
# ---

# # Run Facebook's Segment Anything Model 2 (SAM2) on Modal

# This example demonstrates how to deploy Facebook's [SAM2](https://github.com/facebookresearch/sam2)
# on Modal. SAM2 is a powerful, flexible image and video segmentation model that can be used
# for various computer vision tasks like object detection, instance segmentation,
# and even as a foundation for more complex computer vision applications.
# SAM2 extends the capabilities of the original SAM to include video segmentation.

# In particular, this example segments [this video](https://www.youtube.com/watch?v=WAz1406SjVw) of a man jumping off the cliff.
# This is the output:
#
# | | | |
# |---|---|---|
# | ![Figure 1](./assets/video_output_0.png) | ![Figure 2](./assets/video_output_1.png) | ![Figure 3](./assets/video_output_2.png) |
# | ![Figure 4](./assets/video_output_3.png) | ![Figure 5](./assets/video_output_4.png) | ![Figure 6](./assets/video_output_5.png) |
# | ![Figure 7](./assets/video_output_6.png) | ![Figure 8](./assets/video_output_7.png) | ![Figure 9](./assets/video_output_8.png) |


# # Setup

# First, we set up the Modal image with the necessary dependencies, including PyTorch,
# OpenCV, `huggingFace_hub`, and Torchvision. We also install the SAM2 library.


import modal

from .helper import show_mask, show_points

MODEL_TYPE = "facebook/sam2-hiera-large"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "python3-opencv")
    .pip_install(
        "torch",
        "torchvision",
        "opencv-python",
        "pycocotools",
        "matplotlib",
        "onnxruntime",
        "onnx",
        "huggingface_hub",
    )
    .run_commands("git clone https://github.com/facebookresearch/sam2.git")
    .run_commands("cd sam2 && pip install -e .")
)

app = modal.App("sam2-app", image=image)

# # Model definition

# Next, we define the `Model` class that will handle SAM2 operations for both image and video.


# We use `@modal.build()` and `@modal.enter() decorators here for optimization:
# @modal.build() ensures this method runs during the container build process,
# downloading the model only once and caching it in the container image.
# `@modal.enter()` makes sure the method runs only once when a new container starts,
# initializing the model and moving it to GPU.
# The upshot is that model downloading and initialization only happen once upon container startup.
# This significantly reduces cold start times and improves performance.
@app.cls(gpu="any", timeout=600)
class Model:
    # Model initialization
    @modal.build()
    @modal.enter()
    def download_model_to_folder(self):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        self.image_predictor = SAM2ImagePredictor.from_pretrained(MODEL_TYPE)
        self.video_predictor = SAM2VideoPredictor.from_pretrained(MODEL_TYPE)

    # Prompt-based mask generation for images
    @modal.method()
    def generate_image_masks(self, image):
        import io

        import cv2
        import matplotlib.pyplot as plt
        import numpy as np
        import torch

        # Convert image bytes to numpy array
        image = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # We are hardcoding the input point and label here
        # In a real-world scenario, you would want to display the image
        # and allow the user to click on the image to select the point
        input_point = np.array([[300, 250]])
        input_label = np.array([1])

        # Want to run the model on GPU
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            self.image_predictor.set_image(image)
            masks, scores, logits = self.image_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

            # # Visualize the results
            frame_images = []
            for i, (mask, score) in enumerate(zip(masks, scores)):
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                show_mask(mask, plt.gca())
                show_points(input_point, input_label, plt.gca())
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
                plt.axis("off")

                # Convert plot to PNG bytes
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                frame_images.append(buf.getvalue())
                plt.close()

        return frame_images

    @modal.method()
    def generate_video_masks(self, frame_names, video_dir):
        import io
        import os

        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        from PIL import Image

        # We are hardcoding the input point and label here
        # In a real-world scenario, you would want to display the video
        # and allow the user to click on the video to select the point
        points = np.array([[250, 200]], dtype=np.float32)
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1], np.int32)

        # Want to run the model on GPU
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16
        ):
            self.inference_state = self.video_predictor.init_state(
                video_path=video_dir
            )

            # add new prompts and instantly get the output on the same frame
            (
                frame_idx,
                object_ids,
                masks,
            ) = self.video_predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
            )

            print(
                f"frame_idx: {frame_idx}, object_ids: {object_ids}, masks: {masks}"
            )

            # run propagation throughout the video and collect the results in a dict
            video_segments = {}  # video_segments contains the per-frame segmentation results
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in self.video_predictor.propagate_in_video(self.inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            # render the segmentation results every few frames
            vis_frame_stride = 30
            frame_images = []
            for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.set_title(f"frame {out_frame_idx}")
                frame = Image.open(
                    os.path.join(video_dir, frame_names[out_frame_idx])
                )
                ax.imshow(frame)
                for out_obj_id, out_mask in video_segments[
                    out_frame_idx
                ].items():
                    show_mask(out_mask, ax, obj_id=out_obj_id)

                # Convert plot to PNG bytes
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                frame_images.append(buf.getvalue())
                plt.close(fig)

        return frame_images


# # Local entrypoint


# Finally, we define a [`local_entrypoint`](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps)
# to run the segmentation. This local entrypoint invokes the `generate_image_masks` and `generate_video_masks` methods.
# Please note that there are several ways to pass the image from the local machine to the container running the Modal function.
# One way is to conver the image to bytes and pass the bytes to the function.
# Another way is to mount the image (or images) to the container.
# In the image segmentation example below, the image is passed as bytes to the function.
# In the video segmentation example on the other hand, the directory with all the frames is [auto-mounted](https://modal.com/docs/guide/mounting) to the container
@app.local_entrypoint()
def main():
    import os

    # Instantiate the model
    model = Model()

    # Image segmentation

    # Read the image as bytes
    with open("06_gpu_and_ml/sam/assets/dog.jpg", "rb") as f:
        image_bytes = f.read()

    # Pass image bytes to the function
    frame_images = model.generate_image_masks.remote(image_bytes)

    # # Save the output image bytes to assets/ folder
    for i, image_bytes in enumerate(frame_images):
        output_path = f"06_gpu_and_ml/sam/assets/image_output_{i}.png"
        print(f"Saving it to {output_path}")
        with open(output_path, "wb") as f:
            f.write(image_bytes)

    # # Video segmentation

    # Directory that will be automounted to the container
    video_dir = "06_gpu_and_ml/sam/assets/video_frames"

    # scan all the JPEG frame names in this directory
    frame_names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    frame_images = model.generate_video_masks.remote(
        frame_names=frame_names, video_dir=video_dir
    )

    for i, image_bytes in enumerate(frame_images):
        output_path = f"06_gpu_and_ml/sam/assets/video_output_{i}.png"
        print(f"Saving it to {output_path}")
        with open(output_path, "wb") as f:
            f.write(image_bytes)

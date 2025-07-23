import os
import gradio as gr
from PIL import Image
import numpy as np
from src.utils import *
from scripts.grounding_sam import *
from diffusers.models.attention_processor import FluxAttnProcessor2_0
from src.attn_processor import (
    PersonalizeAnythingAttnProcessor,
    MultiPersonalizeAnythingAttnProcessor,
    set_flux_transformer_attn_processor,
)
from src.pipeline import RFInversionParallelFluxPipeline
from diffusers import FluxTransformer2DModel
from diffusers.hooks import apply_group_offloading

FLUX_DEV = "black-forest-labs/FLUX.1-dev"

MAX_SEED = np.iinfo(np.int32).max

device = torch.device("cuda")
torch_type = torch.bfloat16

# use a smaller transformer
transformer = FluxTransformer2DModel.from_single_file(
    "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors",
    torch_dtype=torch_type,
)

pipe = RFInversionParallelFluxPipeline.from_pretrained(
    FLUX_DEV, torch_dtype=torch_type, transformer=transformer
)

apply_group_offloading(
    pipe.transformer,
    offload_device=torch.device("cpu"),
    onload_device=torch.device("cuda"),
    offload_type="leaf_level",
    use_stream=True,
)
apply_group_offloading(
    pipe.text_encoder, 
    offload_device=torch.device("cpu"),
    onload_device=torch.device("cuda"),
    offload_type="leaf_level",
    use_stream=True,
)
apply_group_offloading(
    pipe.text_encoder_2, 
    offload_device=torch.device("cpu"),
    onload_device=torch.device("cuda"),
    offload_type="leaf_level",
    use_stream=True,
)
apply_group_offloading(
    pipe.vae, 
    offload_device=torch.device("cpu"),
    onload_device=torch.device("cuda"),
    offload_type="leaf_level",
    use_stream=True,
)


IMAGE_MAX_SIZE=512

os.environ["GRADIO_TEMP_DIR"] = "tmp"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

root_dir = "result"
save_dir = os.path.join(root_dir, "gradio")
os.makedirs(save_dir, exist_ok=True)

object_detector, processor, segmentator = prepare_model(device=device)


def predict_outpainting_mask(
    pil_image,
    r_x,
    r_y,
    resize_width,
    resize_height,
    full_width,
    full_height,
):
    background = Image.new("RGB", (full_width, full_height), (0, 0, 0))
    mask = Image.new("L", (full_width, full_height), 0)

    # resize
    resized = pil_image.resize((resize_width, resize_height), Image.LANCZOS)
    paste_x = max(r_x, 0)
    paste_y = max(r_y, 0)
    paste_max_x = min(r_x + resize_width, full_width)
    paste_max_y = min(r_y + resize_height, full_height)

    src_left = max(-r_x, 0)
    src_top = max(-r_y, 0)
    src_right = min(resize_width, full_width - r_x)
    src_bottom = min(resize_height, full_height - r_y)

    if (src_right > src_left) and (src_bottom > src_top):
        valid_region = resized.crop((src_left, src_top, src_right, src_bottom))
        background.paste(valid_region, (paste_x, paste_y))
        mask.paste(255, (paste_x, paste_y, paste_max_x, paste_max_y))

    background.save(os.path.join(save_dir, "background.png"))
    mask.save(os.path.join(save_dir, "mask.png"))

    return background


def predict_inpainting_mask(im, save_dir=save_dir):
    mask_img = os.path.join(save_dir, "mask.png")
    bg_img = os.path.join(save_dir, "background.png")
    save_array_as_png(im["background"], bg_img)
    convert_to_mask_inpainting(im["layers"][0], mask_img)


def generate_from_latents(
    prompt, init_image, mask_tensor, seed, timestep, tau, random_seed
):
    if random_seed:
        seed = random.randint(0, MAX_SEED)

    width, height = init_image.size
    latent_h = height // (pipe.vae_scale_factor * 2)
    latent_w = width // (pipe.vae_scale_factor * 2)
    img_dims = latent_h * latent_w
    generator = torch.Generator(device=device).manual_seed(seed)

    set_flux_transformer_attn_processor(
        pipe.transformer, lambda *args, **kwargs: FluxAttnProcessor2_0()
    )
    inverted_latents, image_latents, latent_image_ids = pipe.invert(
        source_prompt="",
        image=init_image,
        height=height,
        width=width,
        num_inversion_steps=timestep,
        gamma=1.0,
    )

    set_flux_transformer_attn_processor(
        pipe.transformer,
        set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
            name=name, tau=tau / 100, mask=mask_tensor, device=device, img_dims=img_dims
        ),
    )

    image = pipe(
        ["", prompt],
        inverted_latents=inverted_latents,
        image_latents=image_latents,
        latent_image_ids=latent_image_ids,
        height=height,
        width=width,
        start_timestep=0.0,
        stop_timestep=0.99,
        num_inference_steps=timestep,
        eta=1.0,
        generator=generator,
    ).images[-1]

    return image


def generate_image_with_perturbation(
    tb_label, image_dict, mask_dict, prompt, seed, timestep, tau, random_seed, shift=0
):

    if isinstance(image_dict, dict) and "background" in image_dict:
        image_np = image_dict["background"]
    init_image = Image.fromarray(image_np.astype(np.uint8)).convert("RGB")

    if isinstance(mask_dict, dict) and "composite" in mask_dict:
        drawn_mask = mask_dict["composite"]

    # resize
    init_image = init_image.resize((IMAGE_MAX_SIZE, IMAGE_MAX_SIZE), Image.LANCZOS)
    width, height = init_image.size
    latent_h = height // (pipe.vae_scale_factor * 2)
    latent_w = width // (pipe.vae_scale_factor * 2)

    print(f"Image size: {width}x{height}, latent size: {latent_w}x{latent_h}")
    
    mask_resized = Image.fromarray(drawn_mask).resize(
        (latent_w, latent_h), Image.NEAREST
    )
    mask_tensor = (
        torch.tensor(np.array(mask_resized), dtype=torch.float32, device=device) / 255.0
    )

    # Apply shift to mask if shift is not 0
    if shift != 0:
        shift_mask = shift_tensor(mask_tensor, shift)
    else:
        shift_mask = None

    return generate_from_latents_with_shift(
        tb_label, prompt, init_image, mask_tensor, shift_mask, seed, timestep, tau, random_seed
    )


def generate_from_latents_with_shift(
    tb_label, prompt, init_image, mask_tensor, shift_mask, seed, timestep, tau, random_seed
):
    if random_seed:
        seed = random.randint(0, MAX_SEED)

    width, height = init_image.size
    latent_h = height // (pipe.vae_scale_factor * 2)
    latent_w = width // (pipe.vae_scale_factor * 2)
    img_dims = latent_h * latent_w
    generator = torch.Generator(device=device).manual_seed(seed)

    set_flux_transformer_attn_processor(
        pipe.transformer, lambda *args, **kwargs: FluxAttnProcessor2_0()
    )
    inverted_latents, image_latents, latent_image_ids = pipe.invert(
        source_prompt="",
        image=init_image,
        height=height,
        width=width,
        num_inversion_steps=timestep,
        gamma=1.0,
    )

    set_flux_transformer_attn_processor(
        pipe.transformer,
        set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
            name=name, 
            tau=tau / 100, 
            mask=mask_tensor, 
            shift_mask=shift_mask,
            device=device, 
            img_dims=img_dims
        ),
    )

    image = pipe(
        [tb_label, prompt],
        inverted_latents=inverted_latents,
        image_latents=image_latents,
        latent_image_ids=latent_image_ids,
        height=height,
        width=width,
        start_timestep=0.0,
        stop_timestep=0.99,
        num_inference_steps=timestep,
        eta=1.0,
        generator=generator,
    ).images[-1]

    return image

def generate_image(tb_painting_label, prompt, seed, timestep, tau, random_seed, height=None, width=None):
    with torch.no_grad():
        if random_seed:
            seed = random.randint(0, MAX_SEED)

        init_image_path = os.path.join(save_dir, "background.png")
        init_image = Image.open(init_image_path).convert("RGB")

        if height is None or width is None:
            width, height = init_image.size

        latent_h = height // (pipe.vae_scale_factor * 2)
        latent_w = width // (pipe.vae_scale_factor * 2)
        img_dims = latent_h * latent_w
        height, width = latent_h * (pipe.vae_scale_factor * 2), latent_w * (
            pipe.vae_scale_factor * 2
        )
        init_image = init_image.resize((width, height))
        print(f"Image size: {width}x{height}, Latent size: {latent_w}x{latent_h}")

        generator = torch.Generator(device=device).manual_seed(seed)

        mask_path = os.path.join(save_dir, "mask.png")
        mask = create_mask(mask_path, latent_w, latent_h)
        mask = (
            torch.tensor(np.array(mask), dtype=torch.float32, device=device) / 255.0
        )

        inverted_latents, image_latents, latent_image_ids = pipe.invert(
            source_prompt="",
            image=init_image,
            height=height,
            width=width,
            num_inversion_steps=timestep,
            gamma=1.0,
        )
        set_flux_transformer_attn_processor(
            pipe.transformer,
            set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
                name=name, tau=tau / 100, mask=mask, device=device, img_dims=img_dims
            ),
        )

        image = pipe(
            [tb_painting_label, prompt],
            inverted_latents=inverted_latents,
            image_latents=image_latents,
            latent_image_ids=latent_image_ids,
            height=height,
            width=width,
            start_timestep=0.0,
            stop_timestep=0.99,
            num_inference_steps=timestep,
            eta=1.0,
            generator=generator,
        ).images[-1]

        set_flux_transformer_attn_processor(
            pipe.transformer,
            set_attn_proc_func=lambda name, dh, nh, ap: FluxAttnProcessor2_0(),
        )

    return image


def generate_mask_from_sam(input_dict, label_text):
    if isinstance(input_dict, dict) and "background" in input_dict:
        image_np = input_dict["background"]
    image = Image.fromarray(image_np.astype(np.uint8)).convert("RGB")
    width, height = image.size
    _, detections, _ = grounded_segmentation(
        object_detector, processor, segmentator, image=image, labels=label_text
    )

    if len(detections) == 0 or detections[0].mask is None:
        raise gr.Error("No object detected. Try changing the label.")

    mask_np = detections[0].mask

    bin_mask = (mask_np > 0).astype(np.uint8) * 255

    return Image.fromarray(bin_mask, mode="L").convert("RGB")

def generate_image_with_scene(image_dict, bg_dict, mask_dict, fg_prompt, bg_prompt, new_prompt, seed, timestep, tau, random_seed):
    if random_seed:
        seed = random.randint(0, MAX_SEED)

    if isinstance(image_dict, dict) and "background" in image_dict:
        image_np = image_dict["background"]
    init_image = Image.fromarray(image_np.astype(np.uint8)).convert("RGB")

    if isinstance(bg_dict, dict) and "background" in bg_dict:
        bg_image_np = bg_dict["background"]
    bg_image = Image.fromarray(bg_image_np.astype(np.uint8)).convert("RGB")

    if isinstance(mask_dict, dict) and "composite" in mask_dict:
        drawn_mask = mask_dict["composite"]

    # resize
    init_image = init_image.resize((IMAGE_MAX_SIZE, IMAGE_MAX_SIZE), Image.LANCZOS)
    bg_image = bg_image.resize((IMAGE_MAX_SIZE, IMAGE_MAX_SIZE), Image.LANCZOS)
    width, height = init_image.size
    latent_h = height // (pipe.vae_scale_factor * 2)
    latent_w = width // (pipe.vae_scale_factor * 2)
    img_dims = latent_h * latent_w

    mask_resized = Image.fromarray(drawn_mask).resize(
        (latent_w, latent_h), Image.NEAREST
    )
    bg_mask = 1 - np.array(mask_resized)
    mask_tensor = (
        torch.tensor(np.array(mask_resized), dtype=torch.float32, device=device) / 255.0
    )
    bg_mask_tensor = (
        torch.tensor(bg_mask, dtype=torch.float32, device=device) / 255.0
    )

    generator = torch.Generator(device=device).manual_seed(seed)
    
    set_flux_transformer_attn_processor(pipe.transformer, set_attn_proc_func=lambda name, dh, nh, ap:FluxAttnProcessor2_0())
    inverted_latents_fg, image_latents_fg, latent_image_ids = pipe.invert(
        source_prompt="",
        image=init_image,
        height=height,
        width=width,
        num_inversion_steps=timestep,
        gamma=1.0,
    )

    inverted_latents_bg, image_latents_bg, latent_image_ids = pipe.invert(
        source_prompt="",
        image=bg_image,
        height=height,
        width=width,
        num_inversion_steps=timestep,
        gamma=1.0,
    )

    inverted_latents = torch.cat([inverted_latents_fg, inverted_latents_bg], dim=0)
    image_latents = torch.cat([image_latents_fg, image_latents_bg], dim=0)
    masks = [mask_tensor, bg_mask_tensor]

    set_flux_transformer_attn_processor(
        pipe.transformer,
        set_attn_proc_func=lambda name, dh, nh, ap: MultiPersonalizeAnythingAttnProcessor(
            name=name, tau=tau/100, masks=masks, shift_masks=None, device=device, img_dims=img_dims),
    )

    image = pipe(
        [fg_prompt, bg_prompt, new_prompt],
        inverted_latents=inverted_latents,
        image_latents=image_latents,
        latent_image_ids=latent_image_ids,
        height = height,
        width = width,
        start_timestep=0.0,
        stop_timestep=0.99,
        num_inference_steps=timestep,
        eta=1.0,
        generator=generator,
    ).images[-1]
    
    return image



with gr.Blocks() as demo:
    with gr.Tab("Layout-guided Subject Personalization"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    im_layout = gr.Sketchpad(
                        label="Original image", sources="upload", type="numpy"
                    )
                gr.Markdown(
                    "Provide a specific label for the object you want to segment, such as 'teddy bear' or 'robot'. "
                )
                with gr.Row():
                    tb_label = gr.Textbox(
                        label="Label", lines=1, placeholder="e.g.,A teddy bear"
                    )

                button_generate_mask = gr.Button(
                    value="Generate Mask", variant="primary"
                )
            with gr.Column():
                with gr.Row():
                    im_layout_mask = gr.ImageEditor(
                        image_mode="1",
                        brush=gr.Brush(colors=["#000000", "#FFFFFF"]),
                        label="Mask",
                        type="numpy",
                        interactive=True,
                    )
                gr.Markdown(
                    "Use the black and white brushes to adjust the mask. Once ready, enter a prompt to generate the image. "
                )
                with gr.Row():
                    tb_layout = gr.Textbox(label="Prompt", lines=1)
                with gr.Row():
                    shift_layout = gr.Slider(
                        label="Mask Shift",
                        minimum=-10,
                        maximum=10,
                        step=1,
                        value=0,
                        info="Shift mask position: negative values shift left, positive values shift right"
                    )
                with gr.Row():
                    tau_layout = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        label="Tau",
                        info="Smaller tau enhances local consistency at the cost of overall harmony.",
                    )
                with gr.Row():
                    seed_layout = gr.Slider(
                        label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=18
                    )
                    random_seed_layout = gr.Checkbox(
                        label="Random seed", value=False
                    )
                    timestep_layout = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=28,
                    )
                button_layout = gr.Button(value="Run Generation", variant="primary")
            with gr.Column():
                result_layout = gr.Image(label="Generated image")
    with gr.Tab("Multi-subject Personalization"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    im_background = gr.Sketchpad(
                        label="Background Subject Image", sources="upload", type="numpy"
                    )
                gr.Markdown(
                    "Provide a label for the background subject, such as 'mountain' or 'city'. "
                )
                with gr.Row():
                    tb_background_label = gr.Textbox(
                        label="Label", lines=1, placeholder="e.g., A mountain"
                    )
            with gr.Column():
                with gr.Row():
                    im_subject = gr.Sketchpad(
                        label="Foregound Subject Image", sources="upload", type="numpy"
                    )
                gr.Markdown(
                    "Provide a label for the foreground subject you want to segment, such as 'teddy bear' or 'robot'. "
                )
                with gr.Row():
                    tb_subject_label = gr.Textbox(
                        label="Label", lines=1, placeholder="e.g., A teddy bear"
                    )
                button_generate_subject_mask = gr.Button(
                    value="Generate Mask", variant="primary"
                )
            
            with gr.Column():
                with gr.Row():
                    im_subject_mask = gr.ImageEditor(
                        image_mode="1",
                        brush=gr.Brush(colors=["#000000", "#FFFFFF"]),
                        label="Mask",
                        type="numpy",
                        interactive=True,
                    )
                gr.Markdown(
                    "Use the black and white brushes to adjust the mask. Once ready, enter a prompt to generate the image. "
                )
                with gr.Row():
                    tb_subject_prompt = gr.Textbox(
                        label="Prompt", lines=1
                    )
                with gr.Row():
                    tau_subject = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        label="Tau",
                        info="Smaller tau enhances local consistency at the cost of overall harmony.",
                    )
                with gr.Row():
                    seed_subject = gr.Slider(
                        label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=18
                    )
                    random_seed_subject = gr.Checkbox(
                        label="Random seed", value=False
                    )
                    timestep_subject = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=28,
                    )
                button_subject = gr.Button(value="Run Generation", variant="primary")
            with gr.Column():
                result_subject = gr.Image(label="Generated image")
    with gr.Tab("Inpainting"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    im_inpainting = gr.Sketchpad(
                        label="Original image", sources="upload", type="numpy"
                    )
                gr.Markdown(
                    "Provide a specific label for the object you want to segment, such as 'A moutain'. "
                )
                with gr.Row():
                    tb_painting_label = gr.Textbox(
                        label="Label", lines=1, placeholder="e.g., A moutain"
                    )
                with gr.Row():
                    tb_inpainting = gr.Textbox(label="Prompt (Optional)", lines=1)
                with gr.Row():
                    tau_inpainting = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        label="Tau",
                        info="Smaller tau enhances local consistency at the cost of overall harmony.",
                    )
                with gr.Row():
                    seed_inpainting = gr.Slider(
                        label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=18
                    )
                    random_seed_inpainting = gr.Checkbox(
                        label="Random seed", value=False
                    )
                    timestep_inpainting = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=28,
                    )

                button_inpainting = gr.Button(value="Run Generation", variant="primary")

            with gr.Column():
                result_im_inpainting = gr.Image(label="Generated image")

    with gr.Tab("Outpainting"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    im_outpainting = gr.Image(type="pil", label="Input Image")
                    process_im_outpainting = gr.Image(type="pil", label="Process Image")
                gr.Markdown(
                    "Provide a specific label for the object you want to segment, such as 'A cyberpunk-style girl'. "
                )
                with gr.Row():
                    tb_painting_label = gr.Textbox(
                        label="Label", lines=1, placeholder="e.g., A cyberpunk-style girl."
                    )
                with gr.Row():
                    resize_height = gr.Slider(
                        label="Resize height", minimum=256, maximum=1024, value=768
                    )
                    resize_width = gr.Slider(
                        label="Resize width", minimum=256, maximum=1024, value=768
                    )
                with gr.Row():
                    height_outpainting = gr.Slider(
                        label="Target height", minimum=256, maximum=1024, value=1024
                    )
                    width_outpainting = gr.Slider(
                        label="Target width", minimum=256, maximum=1024, value=1024
                    )
                with gr.Row():
                    r_x = gr.Slider(
                        label="The X coordinate of the top-left corner pasted on the canvas.",
                        minimum=0,
                        maximum=512,
                        value=128,
                        step=1,
                    )
                    r_y = gr.Slider(
                        label="The Y coordinate of the top-left corner pasted on the canvas.",
                        minimum=0,
                        maximum=512,
                        value=0,
                        step=1,
                    )
                with gr.Row():
                    preview_button = gr.Button("Preview alignment and mask")

                with gr.Row():
                    tb_outpainting = gr.Textbox(label="Prompt (Optional)", lines=1)
                with gr.Row():
                    tau = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=70,
                        label="Tau",
                        info="Smaller tau enhances local consistency at the cost of overall harmony.",
                    )
                with gr.Row():
                    seed = gr.Slider(
                        label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=18
                    )
                    random_seed = gr.Checkbox(label="Random seed", value=False)
                    timestep = gr.Slider(
                        label="Number of inference steps",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=28,
                    )

                button_outpainting = gr.Button(
                    value="Run Generation", variant="primary"
                )

            with gr.Column():
                result_im_outpainting = gr.Image(label="Generated image")

    # layout-guided subject personalization (single subject personalization with manual perturbation)
    button_generate_mask.click(
        generate_mask_from_sam, inputs=[im_layout, tb_label], outputs=im_layout_mask
    )

    button_layout.click(
        generate_image_with_perturbation,
        inputs=[
            tb_label,
            im_layout,
            im_layout_mask,
            tb_layout,
            seed_layout,
            timestep_layout,
            tau_layout,
            random_seed_layout,
            shift_layout,
        ],
        outputs=result_layout,
    )

    # subject scene composition
    button_generate_subject_mask.click(
        generate_mask_from_sam,
        inputs=[im_subject, tb_subject_label],
        outputs=im_subject_mask,
    )
    button_subject.click(
        generate_image_with_scene,
        inputs=[
            im_subject,
            im_background,
            im_subject_mask,
            tb_subject_label,
            tb_background_label,
            tb_subject_prompt,
            seed_subject,
            timestep_subject,
            tau_subject,
            random_seed_subject,
        ],
        outputs=result_subject,
    )

    # inpainting
    im_inpainting.change(predict_inpainting_mask, inputs=[im_inpainting])
    button_inpainting.click(
        generate_image,
        inputs=[
            tb_painting_label,
            tb_inpainting,
            seed_inpainting,
            timestep_inpainting,
            tau_inpainting,
            random_seed_inpainting,
        ],
        outputs=result_im_inpainting,
    )

    # outpainting
    preview_button.click(
        predict_outpainting_mask,
        inputs=[
            im_outpainting,
            r_x,
            r_y,
            resize_width,
            resize_height,
            width_outpainting,
            height_outpainting,
        ],
        outputs=process_im_outpainting,
    )
    button_outpainting.click(
        predict_outpainting_mask,
        inputs=[
            im_outpainting,
            r_x,
            r_y,
            resize_width,
            resize_height,
            width_outpainting,
            height_outpainting,
        ],
        outputs=process_im_outpainting,
    ).then(
        generate_image,
        inputs=[
            tb_painting_label,
            tb_outpainting,
            seed,
            timestep,
            tau,
            random_seed,
            height_outpainting,
            width_outpainting,
        ],
        outputs=result_im_outpainting,
    )

if __name__ == "__main__":

    demo.launch(share=True)

# TORCH_CUDA_ARCH_LIST="8.6,8.9" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=4 python gradio_demo.py

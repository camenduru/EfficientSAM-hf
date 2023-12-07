import copy
import os  # noqa

import gradio as gr
import numpy as np
import torch
from PIL import ImageDraw
from torchvision.transforms import ToTensor

from utils.tools import format_results, point_prompt
from utils.tools_gradio import fast_process

# Most of our demo code is from [FastSAM Demo](https://huggingface.co/spaces/An-619/FastSAM). Thanks for AN-619.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpu_checkpoint_path = "efficientsam_s_gpu.jit"
cpu_checkpoint_path = "efficientsam_s_cpu.jit"

if torch.cuda.is_available():
    model = torch.jit.load(gpu_checkpoint_path)
else:
    model = torch.jit.load(cpu_checkpoint_path)
model.eval()

# Description
title = "<center><strong><font size='8'>Efficient Segment Anything(EfficientSAM)<font></strong></center>"

description_e = """This is a demo of [Efficient Segment Anything(EfficientSAM) Model](https://github.com/yformer/EfficientSAM).
              """

description_p = """# Interactive Instance Segmentation
                - Point-prompt instruction
                <ol>
                <li> Click on the left image (point input), visualizing the point on the right image </li>
                <li> Click the button of Segment with Point Prompt </li>
                </ol>
                - Box-prompt instruction
                <ol>
                <li> Click on the left image (one point input), visualizing the point on the right image </li>
                <li> Click on the left image (another point input), visualizing the point and the box on the right image</li>
                <li> Click the button of Segment with Box Prompt </li>
                </ol>
                - Github [link](https://github.com/yformer/EfficientSAM)
              """

# examples
examples = [
    ["examples/image1.jpg"],
    ["examples/image2.jpg"],
    ["examples/image3.jpg"],
    ["examples/image4.jpg"],
    ["examples/image5.jpg"],
    ["examples/image6.jpg"],
    ["examples/image7.jpg"],
    ["examples/image8.jpg"],
    ["examples/image9.jpg"],
    ["examples/image10.jpg"],
    ["examples/image11.jpg"],
    ["examples/image12.jpg"],
    ["examples/image13.jpg"],
    ["examples/image14.jpg"],
]

default_example = examples[0]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


def segment_with_boxs(
    image,
    seg_image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global global_points
    global global_point_label
    if len(global_points) < 2:
        return seg_image
    print("Original Image : ", image.size)

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    print("Scaled Image : ", image.size)
    print("Scale : ", scale)

    scaled_points = np.array(
        [[int(x * scale) for x in point] for point in global_points]
    )
    scaled_points = scaled_points[:2]
    scaled_point_label = np.array(global_point_label)[:2]

    print(scaled_points, scaled_points is not None)
    print(scaled_point_label, scaled_point_label is not None)

    if scaled_points.size == 0 and scaled_point_label.size == 0:
        print("No points selected")
        return image

    nd_image = np.array(image)
    img_tensor = ToTensor()(nd_image)

    print(img_tensor.shape)
    pts_sampled = torch.reshape(torch.tensor(scaled_points), [1, 1, -1, 2])
    pts_sampled = pts_sampled[:, :, :2, :]
    pts_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2])

    predicted_logits, predicted_iou = model(
        img_tensor[None, ...].to(device),
        pts_sampled.to(device),
        pts_labels.to(device),
    )
    predicted_logits = predicted_logits.cpu()
    all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
    predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()


    max_predicted_iou = -1
    selected_mask_using_predicted_iou = None
    selected_predicted_iou = None

    for m in range(all_masks.shape[0]):
        curr_predicted_iou = predicted_iou[m]
        if (
            curr_predicted_iou > max_predicted_iou
            or selected_mask_using_predicted_iou is None
        ):
            max_predicted_iou = curr_predicted_iou
            selected_mask_using_predicted_iou = all_masks[m:m+1]
            selected_predicted_iou = predicted_iou[m:m+1]

    results = format_results(selected_mask_using_predicted_iou, selected_predicted_iou, predicted_logits, 0)

    annotations = results[0]["segmentation"]
    annotations = np.array([annotations])
    print(scaled_points.shape)
    fig = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        use_retina=use_retina,
        bbox = scaled_points.reshape([4]),
        withContours=withContours,
    )

    global_points = []
    global_point_label = []
    # return fig, None
    return fig


def segment_with_points(
    image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global global_points
    global global_point_label

    print("Original Image : ", image.size)

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    print("Scaled Image : ", image.size)
    print("Scale : ", scale)

    if global_points is None:
        return image
    if len(global_points) < 1:
        return image
    scaled_points = np.array(
        [[int(x * scale) for x in point] for point in global_points]
    )
    scaled_point_label = np.array(global_point_label)

    print(scaled_points, scaled_points is not None)
    print(scaled_point_label, scaled_point_label is not None)

    if scaled_points.size == 0 and scaled_point_label.size == 0:
        print("No points selected")
        return image

    nd_image = np.array(image)
    img_tensor = ToTensor()(nd_image)

    print(img_tensor.shape)
    pts_sampled = torch.reshape(torch.tensor(scaled_points), [1, 1, -1, 2])
    pts_labels = torch.reshape(torch.tensor(global_point_label), [1, 1, -1])

    predicted_logits, predicted_iou = model(
        img_tensor[None, ...].to(device),
        pts_sampled.to(device),
        pts_labels.to(device),
    )
    predicted_logits = predicted_logits.cpu()
    all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
    predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()

    results = format_results(all_masks, predicted_iou, predicted_logits, 0)

    annotations, _ = point_prompt(
        results, scaled_points, scaled_point_label, new_h, new_w
    )
    annotations = np.array([annotations])

    fig = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        points = scaled_points,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )

    global_points = []
    global_point_label = []
    # return fig, None
    return fig


def get_points_with_draw(image, cond_image, evt: gr.SelectData):
    global global_points
    global global_point_label
    if len(global_points) == 0:
        image = copy.deepcopy(cond_image)
    x, y = evt.index[0], evt.index[1]
    label = "Add Mask"
    point_radius, point_color = 15, (255, 255, 0) if label == "Add Mask" else (
        255,
        0,
        255,
    )
    global_points.append([x, y])
    global_point_label.append(1 if label == "Add Mask" else 0)

    print(x, y, label == "Add Mask")

    if image is not None:
        draw = ImageDraw.Draw(image)

        draw.ellipse(
            [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
            fill=point_color,
        )

    return image

def get_points_with_draw_(image, cond_image, evt: gr.SelectData):
    global global_points
    global global_point_label
    if len(global_points) == 0:
        image = copy.deepcopy(cond_image)
    if len(global_points) > 2:
        return image
    x, y = evt.index[0], evt.index[1]
    label = "Add Mask"
    point_radius, point_color = 15, (255, 255, 0) if label == "Add Mask" else (
        255,
        0,
        255,
    )
    global_points.append([x, y])
    global_point_label.append(1 if label == "Add Mask" else 0)

    print(x, y, label == "Add Mask")

    if image is not None:
        draw = ImageDraw.Draw(image)

        draw.ellipse(
            [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
            fill=point_color,
        )

    if len(global_points) == 2:
        x1, y1 = global_points[0]
        x2, y2 = global_points[1]
        if x1 < x2 and y1 < y2:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
        elif x1 < x2 and y1 >= y2:
            draw.rectangle([x1, y2, x2, y1], outline="red", width=5)
        elif x1 >= x2 and y1 < y2:
            draw.rectangle([x2, y1, x1, y2], outline="red", width=5)
        elif x1 >= x2 and y1 >= y2:
            draw.rectangle([x2, y2, x1, y1], outline="red", width=5)

    return image


cond_img_p = gr.Image(label="Input with Point", value=default_example[0], type="pil")
cond_img_b = gr.Image(label="Input with Box", value=default_example[0], type="pil")

segm_img_p = gr.Image(
    label="Segmented Image with Point-Prompt", interactive=False, type="pil"
)
segm_img_b = gr.Image(
    label="Segmented Image with Box-Prompt", interactive=False, type="pil"
)

global_points = []
global_point_label = []

input_size_slider = gr.components.Slider(
    minimum=512,
    maximum=1024,
    value=1024,
    step=64,
    label="Input_size",
    info="Our model was trained on a size of 1024",
)

with gr.Blocks(css=css, title="Efficient SAM") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)

    with gr.Tab("Point mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_p.render()

            with gr.Column(scale=1):
                segm_img_p.render()

        # Submit & Clear
        # ###
        with gr.Row():
            with gr.Column():

                with gr.Column():
                    segment_btn_p = gr.Button(
                        "Segment with Point Prompt", variant="primary"
                    )
                    clear_btn_p = gr.Button("Clear", variant="secondary")

                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[cond_img_p],
                    examples_per_page=4,
                )

            with gr.Column():
                # Description
                gr.Markdown(description_p)

    with gr.Tab("Box mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_b.render()

            with gr.Column(scale=1):
                segm_img_b.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():

                with gr.Column():
                    segment_btn_b = gr.Button(
                        "Segment with Box Prompt", variant="primary"
                    )
                    clear_btn_b = gr.Button("Clear", variant="secondary")

                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[cond_img_b],

                    examples_per_page=4,
                )

            with gr.Column():
                # Description
                gr.Markdown(description_p)

    cond_img_p.select(get_points_with_draw, [segm_img_p, cond_img_p], segm_img_p)

    cond_img_b.select(get_points_with_draw_, [segm_img_b, cond_img_b], segm_img_b)

    segment_btn_p.click(
        segment_with_points, inputs=[cond_img_p], outputs=segm_img_p
    )

    segment_btn_b.click(
        segment_with_boxs, inputs=[cond_img_b, segm_img_b], outputs=segm_img_b
    )

    def clear():
        return None, None

    def clear_text():
        return None, None, None

    clear_btn_p.click(clear, outputs=[cond_img_p, segm_img_p])
    clear_btn_b.click(clear, outputs=[cond_img_b, segm_img_b])

demo.queue()
demo.launch(share=True)

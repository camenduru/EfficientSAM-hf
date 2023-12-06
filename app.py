import gradio as gr
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image

# loading EfficientSAM model
model_path = "efficientsam_s_cpu.jit"
with open(model_path, "rb") as f:
    model = torch.jit.load(f)

# getting mask using points
def get_sam_mask_using_points(img_tensor, pts_sampled, model):
    pts_sampled = torch.reshape(torch.tensor(pts_sampled), [1, 1, -1, 2])
    max_num_pts = pts_sampled.shape[2]
    pts_labels = torch.ones(1, 1, max_num_pts)

    predicted_logits, predicted_iou = model(
        img_tensor[None, ...],
        pts_sampled,
        pts_labels,
    )
    predicted_logits = predicted_logits.cpu()
    all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
    predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()

    max_predicted_iou = -1
    selected_mask_using_predicted_iou = None
    for m in range(all_masks.shape[0]):
        curr_predicted_iou = predicted_iou[m]
        if (
            curr_predicted_iou > max_predicted_iou
            or selected_mask_using_predicted_iou is None
        ):
            max_predicted_iou = curr_predicted_iou
            selected_mask_using_predicted_iou = all_masks[m]
    return selected_mask_using_predicted_iou

# examples
examples = [["examples/image1.jpg"], ["examples/image2.jpg"], ["examples/image3.jpg"], ["examples/image4.jpg"],
            ["examples/image5.jpg"], ["examples/image6.jpg"], ["examples/image7.jpg"], ["examples/image8.jpg"],
            ["examples/image9.jpg"], ["examples/image10.jpg"], ["examples/image11.jpg"], ["examples/image12.jpg"]
            ["examples/image13.jpg"], ["examples/image14.jpg"]]


with gr.Blocks() as demo:
    with gr.Row():
        input_img = gr.Image(label="Input",height=512)
        output_img = gr.Image(label="Selected Segment",height=512)

    with gr.Row():
        gr.Markdown("Try some of the examples below ⬇️")
        gr.Examples(examples=examples,
                    inputs=[input_img])

    def get_select_coords(img, evt: gr.SelectData):
        img_tensor = ToTensor()(img)
        _, H, W = img_tensor.shape

        visited_pixels = set()
        pixels_in_queue = set()
        pixels_in_segment = set()

        mask = get_sam_mask_using_points(img_tensor, [[evt.index[0], evt.index[1]]], model)

        out = img.copy()

        out = out.astype(np.uint8)
        out *= mask[:,:,None]
        for pixel in pixels_in_segment:
            out[pixel[0], pixel[1]] = img[pixel[0], pixel[1]]
        print(out)
        return out

    input_img.select(get_select_coords, [input_img], output_img)

if __name__ == "__main__":
    demo.launch()

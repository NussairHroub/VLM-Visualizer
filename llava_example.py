import os
import sys
sys.path.append("./models")
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from utils import (
    load_image, 
    aggregate_llm_attention, aggregate_vit_attention,
    heterogenous_stack,
    show_mask_on_image
)

# ===> specify the model path
model_path = "liuhaotian/llava-v1.5-7b"

# load the model
load_8bit = False
load_4bit = False
device = "cuda" if torch.cuda.is_available() else "cpu"

disable_torch_init()

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, 
    None,  # model_base
    model_name, 
    load_8bit, 
    load_4bit, 
    device=device
)

# ===> specify the image path or url and the prompt text
image_path_or_url = "https://github.com/open-compass/MMBench/blob/main/samples/MMBench/1.jpg?raw=true"
prompt_text = "What python code can be used to generate the output in the image?"

# ===> Create output directory
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

# Load and process image
image = load_image(image_path_or_url)
image_tensor, images = process_images([image], image_processor, model.config)
image = images[0]
image_size = image.size

if type(image_tensor) is list:
    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
else:
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

# Prepare prompt
conv_mode = "llava_v1"
conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + prompt_text)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt().replace(
    "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. ",
    ""
)

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

# ===> Generate response
with torch.inference_mode():
    outputs = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image_size],
        do_sample=False,
        max_new_tokens=512,
        use_cache=True,
        return_dict_in_generate=True,
        output_attentions=True,
    )

text = tokenizer.decode(outputs["sequences"][0]).strip()
print(text)

# Constructing the LLM attention matrix
aggregated_prompt_attention = []
for layer in outputs["attentions"][0]:
    layer_attns = layer.squeeze(0)
    attns_per_head = layer_attns.mean(dim=0)
    cur = attns_per_head[:-1].cpu().clone()
    cur[1:, 0] = 0.
    cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
    aggregated_prompt_attention.append(cur)
aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)

llm_attn_matrix = heterogenous_stack(
    [torch.tensor([1])]
    + list(aggregated_prompt_attention) 
    + list(map(aggregate_llm_attention, outputs["attentions"]))
)

# ===> Save the LLM attention visualization
gamma_factor = 1
enhanced_attn_m = np.power(llm_attn_matrix.numpy(), 1 / gamma_factor)

fig, ax = plt.subplots(figsize=(10, 20), dpi=150)
ax.imshow(enhanced_attn_m, vmin=enhanced_attn_m.min(), vmax=enhanced_attn_m.max(), interpolation="nearest")
plt.savefig(os.path.join(output_dir, "llm_attention_matrix.png"))
plt.close()

# ===> Save the trend of attention weights over vision tokens
overall_attn_weights_over_vis_tokens = [
    row[
        len(tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0]):
        len(tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0]) + model.get_vision_tower().num_patches
    ].sum().item()
    for row in llm_attn_matrix[len(input_ids[0]) - 1:]
]

fig, ax = plt.subplots(figsize=(20, 5))
ax.plot(overall_attn_weights_over_vis_tokens)

# ===> Fixed Dynamic X-Tick Labeling (Prevents IndexError)
num_labels = len(overall_attn_weights_over_vis_tokens)
num_tokens = len(outputs["sequences"][0])  # Total number of generated tokens

# Ensure we do not exceed the number of available tokens
num_labels = min(num_labels, num_tokens)

step = max(1, num_labels // 20)  # Adjust step size dynamically
tick_indices = list(range(0, num_labels, step))

ax.set_xticks(tick_indices)
ax.set_xticklabels(
    [tokenizer.decode(outputs["sequences"][0][i], add_special_tokens=False).strip() for i in tick_indices],
    rotation=75
)

ax.set_title("Attention Weights Over Vision Tokens")
plt.savefig(os.path.join(output_dir, "attention_trend.png"))
plt.close()

# ===> Save the vision encoder attention
vis_attn_matrix = aggregate_vit_attention(
    model.get_vision_tower().image_attentions,
    select_layer=model.get_vision_tower().select_layer,
    all_prev_layers=True
)
grid_size = model.get_vision_tower().num_patches_per_side

num_image_per_row = 8
num_rows = len(outputs["sequences"][0]) // num_image_per_row + (1 if len(outputs["sequences"][0]) % num_image_per_row != 0 else 0)
fig, axes = plt.subplots(
    num_rows, num_image_per_row, 
    figsize=(10, (10 / num_image_per_row) * (image_size[0] / image_size[1]) * num_rows), 
    dpi=150
)
plt.subplots_adjust(wspace=0.05, hspace=0.2)

output_token_inds = list(range(len(input_ids[0]) - 1, len(input_ids[0]) - 1 + len(outputs["sequences"][0])))
for i, ax in enumerate(axes.flatten()):
    if i >= len(outputs["sequences"][0]):
        ax.axis("off")
        continue

    attn_weights_over_vis_tokens = llm_attn_matrix[output_token_inds[i]][
        len(tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0]):
        len(tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0]) + model.get_vision_tower().num_patches
    ]
    attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum()

    attn_over_image = sum(
        vis_attn.reshape(grid_size, grid_size) * weight 
        for weight, vis_attn in zip(attn_weights_over_vis_tokens, vis_attn_matrix)
    )
    attn_over_image = attn_over_image / attn_over_image.max()

    np_img = np.array(image)[:, :, ::-1]
    img_with_attn, heatmap = show_mask_on_image(np_img, attn_over_image.numpy())

    ax.imshow(img_with_attn)
    ax.axis("off")

plt.savefig(os.path.join(output_dir, "vision_attention_overlay.png"))
plt.close()

print(f"All images saved in '{output_dir}'")

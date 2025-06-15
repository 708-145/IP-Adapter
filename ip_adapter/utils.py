import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

attn_maps = {}
def hook_fn(name):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            attn_maps[name] = module.processor.attn_map
            del module.processor.attn_map

    return forward_hook

def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        if name.split('.')[-1].startswith('attn2'):
            module.register_forward_hook(hook_fn(name))

    return unet

def upscale(attn_map, target_size):
    attn_map = torch.mean(attn_map, dim=0)
    attn_map = attn_map.permute(1,0)
    temp_size = None

    for i in range(0,5):
        scale = 2 ** i
        if ( target_size[0] // scale ) * ( target_size[1] // scale) == attn_map.shape[1]*64:
            temp_size = (target_size[0]//(scale*8), target_size[1]//(scale*8))
            break

    assert temp_size is not None, "temp_size cannot is None"

    attn_map = attn_map.view(attn_map.shape[0], *temp_size)

    attn_map = F.interpolate(
        attn_map.unsqueeze(0).to(dtype=torch.float32),
        size=target_size,
        mode='bilinear',
        align_corners=False
    )[0]

    attn_map = torch.softmax(attn_map, dim=0)
    return attn_map
def get_net_attn_map(image_size, batch_size=2, instance_or_negative=False, detach=True):

    idx = 0 if instance_or_negative else 1
    net_attn_maps = []

    for name, attn_map in attn_maps.items():
        attn_map = attn_map.cpu() if detach else attn_map
        attn_map = torch.chunk(attn_map, batch_size)[idx].squeeze()
        attn_map = upscale(attn_map, image_size) 
        net_attn_maps.append(attn_map) 

    net_attn_maps = torch.mean(torch.stack(net_attn_maps,dim=0),dim=0)

    return net_attn_maps

def attnmaps2images(net_attn_maps):

    #total_attn_scores = 0
    images = []

    for attn_map in net_attn_maps:
        attn_map = attn_map.cpu().numpy()
        #total_attn_scores += attn_map.mean().item()

        normalized_attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
        normalized_attn_map = normalized_attn_map.astype(np.uint8)
        #print("norm: ", normalized_attn_map.shape)
        image = Image.fromarray(normalized_attn_map)

        #image = fix_save_attn_map(attn_map)
        images.append(image)

    #print(total_attn_scores)
    return images
def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator


def get_compel_prompt_embeddings(pipe, prompt, negative_prompt, device):
    """
    Encodes prompts using Compel to obtain embeddings.

    Args:
        pipe: The diffusion pipeline (must have .tokenizer, .text_encoder, and .compel).
        prompt (str or list[str]): The prompt or a list of prompts to encode.
        negative_prompt (str or list[str]): The negative prompt or a list of negative prompts to encode.
        device (str or torch.device): The device to move tensors to.

    Returns:
        tuple: A tuple containing (prompt_embeds, negative_prompt_embeds).
               Returns (None, None) if Compel is not available or an error occurs.
    """
    if not hasattr(pipe, 'compel'):
        print("Pipeline does not have Compel. Skipping Compel-based prompt encoding.")
        return None, None

    try:
        # Ensure prompts are lists for Compel
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        prompt_embeds = pipe.compel.build_conditioning_tensor(prompt).to(device)
        negative_prompt_embeds = pipe.compel.build_conditioning_tensor(negative_prompt).to(device)

        # Pad conditioning tensors if necessary (Compel specific for long prompts)
        if hasattr(pipe.compel, 'pad_conditioning_tensors_to_same_length'):
            prompt_embeds, negative_prompt_embeds = pipe.compel.pad_conditioning_tensors_to_same_length(
                [prompt_embeds, negative_prompt_embeds]
            )

        return prompt_embeds, negative_prompt_embeds
    except Exception as e:
        print(f"Error during Compel prompt encoding: {e}")
        # Fallback to pipeline's default encoder if Compel fails
        try:
            print("Falling back to default pipeline encoder.")
            prompt_embeds = pipe._encode_prompt(
                prompt,
                device=device,
                num_images_per_prompt=1, # Assuming 1 image per prompt for default
                do_classifier_free_guidance=True, # Assuming CFG for default
                negative_prompt=None # Handled separately by default encoder
            )
            negative_prompt_embeds = pipe._encode_prompt(
                negative_prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=None
            )
            # The default _encode_prompt might return a tuple if text_encoder_2 exists (e.g. SDXL)
            # We are interested in the first element (prompt_embeds)
            if isinstance(prompt_embeds, tuple):
                prompt_embeds = prompt_embeds[0]
            if isinstance(negative_prompt_embeds, tuple):
                negative_prompt_embeds = negative_prompt_embeds[0]

            return prompt_embeds, negative_prompt_embeds
        except Exception as fallback_e:
            print(f"Error during fallback prompt encoding: {fallback_e}")
            return None, None
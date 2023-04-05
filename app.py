import gradio as gr
import time
import tqdm
import torch
from omegaconf import OmegaConf
from gligen.task_grounded_generation import grounded_generation_box, load_ckpt, load_common_ckpt
import yaml
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from functools import partial
from collections import Counter
import math
import gc
from collections import defaultdict
import subprocess
from gradio import processing_utils
from typing import Optional
import ast
import os
import warnings
from datetime import datetime
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import pandas as pd
import random
hf_hub_download = partial(hf_hub_download, library_name="gligen_demo")
import cv2 # returns one box
from PIL import Image
# res_plotted = test[0].plot()
import numpy as np
from matplotlib import cm
import sys
sys.tracebacklimit = 0

def make_yaml(name, *cat):
    subprocess.run(['mkdir', f'{name}'])
    lst = list(set([*cat]))
    for i in lst:
        if i == '':
            lst.remove(i)
    nc = len(lst)
    dict1 = {'names': lst,
             'nc': nc, 
             'test': f'test/images',
             'train': f'train/images',
             'val': f'valid/images'}
    with open(f'{name}/data.yaml', 'w') as file:
        documents = yaml.dump(dict1, file)
    return dict1

class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        if x is None:
            return x
        if self.tool == "sketch" and self.source in ["upload", "webcam"] and type(x) != dict:
            decode_image = processing_utils.decode_base64_to_image(x)
            width, height = decode_image.size
            mask = np.zeros((height, width, 4), dtype=np.uint8)
            mask[..., -1] = 255
            mask = self.postprocess(mask)
            x = {'image': x, 'mask': mask}
        return super().preprocess(x)


class Blocks(gr.Blocks):

    def __init__(
        self,
        theme: str = "default",
        analytics_enabled: Optional[bool] = None,
        mode: str = "blocks",
        title: str = "Gradio",
        css: Optional[str] = None,
        **kwargs,
    ):

        self.extra_configs = {
            'thumbnail': kwargs.pop('thumbnail', ''),
            'url': kwargs.pop('url', 'https://gradio.app/'),
            'creator': kwargs.pop('creator', '@teamGradio'),
        }

        super(Blocks, self).__init__(theme, analytics_enabled, mode, title, css, **kwargs)
        warnings.filterwarnings("ignore")

    def get_config_file(self):
        config = super(Blocks, self).get_config_file()

        for k, v in self.extra_configs.items():
            config[k] = v
        
        return config


def draw_box(boxes=[], texts=[], img=None):
    if len(boxes) == 0 and img is None:
        return None

    if img is None:
        img = Image.new('RGB', (512, 512), (255, 255, 255))
    colors = ["red", "olive", "blue", "green", "orange", "brown", "cyan", "purple"]
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("DejaVuSansMono.ttf", size=18)
    for bid, box in enumerate(boxes):
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=colors[bid % len(colors)], width=4)
        anno_text = texts[bid]
        draw.rectangle([box[0], box[3] - int(font.size * 1.2), box[0] + int((len(anno_text) + 0.8) * font.size * 0.6), box[3]], outline=colors[bid % len(colors)], fill=colors[bid % len(colors)], width=4)
        draw.text([box[0] + int(font.size * 0.2), box[3] - int(font.size*1.2)], anno_text, font=font, fill=(255,255,255))
    return img

def get_concat(ims):
    if len(ims) == 1:
        n_col = 1
    else:
        n_col = 2
    n_row = math.ceil(len(ims) / 2)
    dst = Image.new('RGB', (ims[0].width * n_col, ims[0].height * n_row), color="white")
    for i, im in enumerate(ims):
        row_id = i // n_col
        col_id = i % n_col
        dst.paste(im, (im.width * col_id, im.height * row_id))
    return dst


def auto_append_grounding(language_instruction, grounding_texts):
    for grounding_text in grounding_texts:
        if grounding_text not in language_instruction and grounding_text != 'auto':
            language_instruction += "; " + grounding_text
    return language_instruction


def slice_per(source, step):
    return [source[i::step] for i in range(step)]

def generate(task, name, name2, split, grounding_texts, sketch_pad,
             alpha_sample, guidance_scale, batch_size,
             fix_seed, rand_seed, use_actual_mask, append_grounding, style_cond_image,
             state):
    print(task, name, name2, split, grounding_texts)
    name = name2
    if os.path.isdir(name) == False:
        try:
            subprocess.run(['mkdir',f'{name}'])
        except:
            None
        subprocess.run(['mkdir',f'datasets/{name}/'])
        subprocess.run(['mkdir',f'datasets/{name}/test'])
        subprocess.run(['mkdir',f'datasets/{name}/test/labels'])
        subprocess.run(['mkdir',f'datasets/{name}/test/images'])
        subprocess.run(['mkdir',f'datasets/{name}/valid'])
        subprocess.run(['mkdir',f'datasets/{name}/valid/labels'])
        subprocess.run(['mkdir',f'datasets/{name}/valid/images'])
        subprocess.run(['mkdir',f'datasets/{name}/train'])
        subprocess.run(['mkdir',f'datasets/{name}/train/labels'])
        subprocess.run(['mkdir',f'datasets/{name}/train/images'])
        subprocess.run(['touch', f'datasets/{name}/data.yaml'])
        # subprocess.run(['cp','-r',f'datasets/{name}/', 'datasets/datasets/'])


        make_yaml(name, *grounding_texts.split(';'))
    num = len(os.listdir(f'datasets/{name}/{split}/images'))
    image = state.get('original_image', sketch_pad['image']).copy()
    image = center_crop(image)
    image = Image.fromarray(image)
    image.save(f'datasets/{name}/{split}/images/{name}-{num}.png')
    if 'boxes' not in state:
        state['boxes'] = []

    boxes = state['boxes']
    grounding_texts = [x.strip() for x in grounding_texts.split(';')]
    # assert len(boxes) == len(grounding_texts)
    if len(boxes) != len(grounding_texts):
        if len(boxes) < len(grounding_texts):
            raise ValueError("""The number of boxes should be equal to the number of grounding objects.
Number of boxes drawn: {}, number of grounding tokens: {}.
Please draw boxes accordingly on the sketch pad.""".format(len(boxes), len(grounding_texts)))
        grounding_texts = grounding_texts + [""] * (len(boxes) - len(grounding_texts))

    boxes = (np.asarray(boxes) / 512).tolist()
    grounding_instruction = {}

    grounding_instruction = defaultdict(list)
    for obj,box in zip(grounding_texts, boxes):
        
        grounding_instruction[obj].append(box)
    g_i = dict(grounding_instruction)
    with open(f'{name}/data.yaml', 'r') as file:
        confi = yaml.safe_load(file)
        with open(f'datasets/{name}/{split}/labels/{name}-{num}.txt', 'w') as f:
            for i in list(g_i.keys()):
                if len(g_i[i])>1:
                    for box in g_i[i]:
                        f.write(f'{confi["names"].index(i)} {" ".join(map(str, box))}')
                        f.write('\n')
                else:
                    f.write(f'{confi["names"].index(i)} {" ".join(map(str, g_i[i][0]))}')
                    f.write('\n')
            
    return image, g_i, state



def train(name, name2, epochs, model_type, model_type2, prog = gr.Progress()):
    # Load a model
    if name == '':
        name = name2
    if model_type == '':
        model_type = model_type2
    model = YOLO(f"{model_type}.yaml")  # build a new model from scratch
    model = YOLO(f"{model_type}.pt")  # load a pretrained model (recommended for training)
    
    model.train(data=f"{name}/data.yaml", epochs=epochs, verbose = True)
    
    #     yield pd.read_csv('runs/detect/train28.csv')
    metrics = model.val()  # evaluate model performance on the validation set
    success = model.export(format="onnx")  # export the model to ONNX format
    return pd.DataFrame.from_dict([metrics.results_dict])

# ###test = model.predict('/notebooks/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvcHUyMzMxNzg4LWltYWdlLXJtNTAzLTAxXzEtbDBqOXFyYzMucG5n copy.png')

# ###im = Image.fromarray(np.uint8((test[0].plot())*255))

def infer(model_path, model_type, img):
    model = YOLO(f"{model_type}.yaml")  # build a new model from scratch
    model = YOLO(f"{model_path}.pt")  #load your pretrained model (recommended for training)
    test = model.predict(img)
    
    return test[0].plot(), test

def binarize(x):
    return (x != 0).astype('uint8') * 255

def sized_center_crop(img, cropx, cropy):
    y, x = img.shape[:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    return img[starty:starty+cropy, startx:startx+cropx]

def sized_center_fill(img, fill, cropx, cropy):
    y, x = img.shape[:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    img[starty:starty+cropy, startx:startx+cropx] = fill
    return img

def sized_center_mask(img, cropx, cropy):
    y, x = img.shape[:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    center_region = img[starty:starty+cropy, startx:startx+cropx].copy()
    img = (img * 0.2).astype('uint8')
    img[starty:starty+cropy, startx:startx+cropx] = center_region
    return img

def center_crop(img, HW=None, tgt_size=(512, 512)):
    if HW is None:
        H, W = img.shape[:2]
        HW = min(H, W)
    img = sized_center_crop(img, HW, HW)
    img = Image.fromarray(img)
    img = img.resize(tgt_size)
    return np.array(img)

def draw(task, input, grounding_texts, new_image_trigger, state):
    if type(input) == dict:
        image = input['image']
        mask = input['mask']
    else:
        mask = input

    if mask.ndim == 3:
        mask = mask[..., 0]

    image_scale = 1.0

    # resize trigger
    if task == "Grounded Inpainting":
        mask_cond = mask.sum() == 0
        # size_cond = mask.shape != (512, 512)
        if mask_cond and 'original_image' not in state:
            image = Image.fromarray(image)
            width, height = image.size
            scale = 600 / min(width, height)
            image = image.resize((int(width * scale), int(height * scale)))
            state['original_image'] = np.array(image).copy()
            image_scale = float(height / width)
            return [None, new_image_trigger + 1, image_scale, state]
        else:
            original_image = state['original_image']
            H, W = original_image.shape[:2]
            image_scale = float(H / W)

    mask = binarize(mask)
    if mask.shape != (512, 512):
        # assert False, "should not receive any non- 512x512 masks."
        if 'original_image' in state and state['original_image'].shape[:2] == mask.shape:
            mask = center_crop(mask, state['inpaint_hw'])
            image = center_crop(state['original_image'], state['inpaint_hw'])
        else:
            mask = np.zeros((512, 512), dtype=np.uint8)
    # mask = center_crop(mask)
    mask = binarize(mask)

    if type(mask) != np.ndarray:
        mask = np.array(mask)

    if mask.sum() == 0 and task != "Grounded Inpainting":
        state = {}

    if task != 'Grounded Inpainting':
        image = None
    else:
        image = Image.fromarray(image)

    if 'boxes' not in state:
        state['boxes'] = []

    if 'masks' not in state or len(state['masks']) == 0:
        state['masks'] = []
        last_mask = np.zeros_like(mask)
    else:
        last_mask = state['masks'][-1]

    if type(mask) == np.ndarray and mask.size > 1:
        diff_mask = mask - last_mask
    else:
        diff_mask = np.zeros([])

    if diff_mask.sum() > 0:
        x1x2 = np.where(diff_mask.max(0) != 0)[0]
        y1y2 = np.where(diff_mask.max(1) != 0)[0]
        y1, y2 = y1y2.min(), y1y2.max()
        x1, x2 = x1x2.min(), x1x2.max()

        if (x2 - x1 > 5) and (y2 - y1 > 5):
            state['masks'].append(mask.copy())
            state['boxes'].append((x1, y1, x2, y2))

    grounding_texts = [x.strip() for x in grounding_texts.split(';')]
    grounding_texts = [x for x in grounding_texts if len(x) > 0]
    if len(grounding_texts) < len(state['boxes']):
        grounding_texts += [f'Obj. {bid+1}' for bid in range(len(grounding_texts), len(state['boxes']))]

    box_image = draw_box(state['boxes'], grounding_texts, image)

    if box_image is not None and state.get('inpaint_hw', None):
        inpaint_hw = state['inpaint_hw']
        box_image_resize = np.array(box_image.resize((inpaint_hw, inpaint_hw)))
        original_image = state['original_image'].copy()
        box_image = sized_center_fill(original_image, box_image_resize, inpaint_hw, inpaint_hw)

    return [box_image, new_image_trigger, image_scale, state]

def clear(task, sketch_pad_trigger, batch_size, state, switch_task=False):
    if task != 'Grounded Inpainting':
        sketch_pad_trigger = sketch_pad_trigger + 1
    blank_samples = batch_size % 2 if batch_size > 1 else 0
    out_images = [gr.Image.update(value=None, visible=True) for i in range(batch_size)] \
                    + [gr.Image.update(value=None, visible=True) for _ in range(blank_samples)] \
                    + [gr.Image.update(value=None, visible=False) for _ in range(4 - batch_size - blank_samples)]
    state = {}
    return [None, sketch_pad_trigger, None, 1.0] + out_images + [state]

css = """
#img2img_image, #img2img_image > .fixed-height, #img2img_image > .fixed-height > div, #img2img_image > .fixed-height > div > img
{
    height: var(--height) !important;
    max-height: var(--height) !important;
    min-height: var(--height) !important;
}
#paper-info a {
    color:#008AD7;
    text-decoration: none;
}
#paper-info a:hover {
    cursor: pointer;
    text-decoration: none;
}
"""

rescale_js = """
function(x) {
    const root = document.querySelector('gradio-app').shadowRoot || document.querySelector('gradio-app');
    let image_scale = parseFloat(root.querySelector('#image_scale input').value) || 1.0;
    const image_width = root.querySelector('#img2img_image').clientWidth;
    const target_height = parseInt(image_width * image_scale);
    document.body.style.setProperty('--height', `${target_height}px`);
    root.querySelectorAll('button.justify-center.rounded')[0].style.display='none';
    root.querySelectorAll('button.justify-center.rounded')[1].style.display='none';
    return x;
}
"""

with Blocks(
    css=css,
    analytics_enabled=False,
    title="YOLOv8 Gradio demo",
) as main:
    description = """
    <p style="text-align: center; font-weight: bold;">
        <span style="font-size: 28px">YOLOv8 Gradio demo</span>
        <br>
    </p>
    """
    gr.HTML(description)
    with gr.Tab("Label Images"):
        with gr.Row():
            with gr.Column(scale=4):
                sketch_pad_trigger = gr.Number(value=0, visible=False)
                sketch_pad_resize_trigger = gr.Number(value=0, visible=False)
                init_white_trigger = gr.Number(value=0, visible=False)
                image_scale = gr.Number(value=0, elem_id="image_scale", visible=False)
                new_image_trigger = gr.Number(value=0, visible=False)

                task = gr.Radio(
                    choices=["Grounded Generation", 'Grounded Inpainting'],
                    type="value",
                    value="Grounded Inpainting",
                    label="Task", visible = False
                )
                name = gr.Textbox(
                    label = 'name of directory holding files', visible = False
                )
                name2 = gr.Textbox(
                    label = 'Name of directory holding files'
                )
                split = gr.Radio(label='Which image split does this image fall into?', choices = ['train','test','valid'], value = 'train')
                grounding_instruction = gr.Textbox(
                    label="Annotations (seperated by semicolon)",
                )
                with gr.Row():
                    sketch_pad = ImageMask(label="Input image", elem_id="img2img_image")
                    out_imagebox = gr.Image(type="pil", label="Annotated image")
                with gr.Row():
                    clear_btn = gr.Button(value='Clear')
                    gen_btn = gr.Button(value='Generate')

                with gr.Accordion("Advanced Options", open=False, visible = False):
                    with gr.Column():
                        alpha_sample = gr.Slider(minimum=0, maximum=1.0, step=0.1, value=0.3, label="Scheduled Sampling (τ)")
                        guidance_scale = gr.Slider(minimum=0, maximum=50, step=0.5, value=7.5, label="Guidance Scale")
                        batch_size = gr.Slider(minimum=1, maximum=4, step=1, value=2, label="Number of Samples")
                        append_grounding = gr.Checkbox(value=True, label="Append grounding instructions to the caption")
                        use_actual_mask = gr.Checkbox(value=False, label="Use actual mask for inpainting", visible=False)
                        with gr.Row():
                            fix_seed = gr.Checkbox(value=True, label="Fixed seed")
                            rand_seed = gr.Slider(minimum=0, maximum=1000, step=1, value=0, label="Seed")
                        with gr.Row():
                            use_style_cond = gr.Checkbox(value=False, label="Enable Style Condition")
                            style_cond_image = gr.Image(type="pil", label="Style Condition", visible=False, interactive=True)
            with gr.Column(scale=4):
                gr.HTML('<span style="font-size: 20px; font-weight: bold">Generated Images</span>')
                with gr.Row():
                    out_gen_1 = gr.Image(type="pil", visible=True, show_label=False)
                    out_gen_2 = gr.Textbox(visible = True, label = 'YAML Config in dictionary format')
                with gr.Row():
                    out_gen_3 = gr.Image(type="pil", visible=False, show_label=False)
                    out_gen_4 = gr.Image(type="pil", visible=False, show_label=False)

            state = gr.State({})
        
            

            
    with gr.Tab('Train'):
        with gr.Row():
            name = gr.Textbox(label = 'Directory name', visible = False)
            name2 = gr.Textbox(label = 'Directory name')
            epochs = gr.Slider(label = "Number of epochs", value = 1, max = 1000)
            model_type = gr.Radio(label = "Model type", choices = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'], visible = False)
            model_type2 = gr.Radio(label = "Model type", choices = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'])
        with gr.Row():
            train_btn = gr.Button(value = 'Train')
        with gr.Row():
            prog = gr.Textbox(label = 'Training progress', visible = False)
            df = gr.Dataframe(label = 'Final training metrics')
                         
    with gr.Tab('Inference'):
        with gr.Row():
            model_path = gr.Textbox(label = 'Path to the pretrained model')
            model_type = gr.Radio(label = "Model type", choices = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'])
            img = gr.Image(label = 'Input image', interactive = True)
            inf_btn = gr.Button(label = 'Run inference on image')
        with gr.Row():
            out_inf_img = gr.Image(label = 'Output image', type = 'pil') 
            outybox = gr.Textbox(label = 'Full output')
                      
        

    class Controller:
        def __init__(self):
            self.calls = 0
            self.tracks = 0
            self.resizes = 0
            self.scales = 0

        def init_white(self, init_white_trigger):
            self.calls += 1
            return np.ones((512, 512), dtype='uint8') * 255, 1.0, init_white_trigger+1

        def change_n_samples(self, n_samples):
            blank_samples = n_samples % 2 if n_samples > 1 else 0
            return [gr.Image.update(visible=True) for _ in range(n_samples + blank_samples)] \
                + [gr.Image.update(visible=False) for _ in range(4 - n_samples - blank_samples)]

        def resize_centercrop(self, state):
            self.resizes += 1
            image = state['original_image'].copy()
            inpaint_hw = int(0.9 * min(*image.shape[:2]))
            state['inpaint_hw'] = inpaint_hw
            image_cc = center_crop(image, inpaint_hw)
            # print(f'resize triggered {self.resizes}', image.shape, '->', image_cc.shape)
            return image_cc, state

        def resize_masked(self, state):
            self.resizes += 1
            image = state['original_image'].copy()
            inpaint_hw = int(0.9 * min(*image.shape[:2]))
            state['inpaint_hw'] = inpaint_hw
            image_mask = sized_center_mask(image, inpaint_hw, inpaint_hw)
            state['masked_image'] = image_mask.copy()
            # print(f'mask triggered {self.resizes}')
            return image_mask, state

        def switch_task_hide_cond(self, task):
            cond = True
            return gr.Checkbox.update(visible=cond, value=False), gr.Image.update(value=None, visible=False), gr.Slider.update(visible=cond), gr.Checkbox.update(visible=(not cond), value=False)

    controller = Controller()
    main.load(
        lambda x:x,
        inputs=sketch_pad_trigger,
        outputs=sketch_pad_trigger,
        queue=False)
    sketch_pad.edit(
        draw,
        inputs=[task, sketch_pad, grounding_instruction, sketch_pad_resize_trigger, state],
        outputs=[out_imagebox, sketch_pad_resize_trigger, image_scale, state],
        queue=False,
    )
    grounding_instruction.change(
        draw,
        inputs=[task, sketch_pad, grounding_instruction, sketch_pad_resize_trigger, state],
        outputs=[out_imagebox, sketch_pad_resize_trigger, image_scale, state],
        queue=False,
    )
    clear_btn.click(
        clear,
        inputs=[task, sketch_pad_trigger, batch_size, state],
        outputs=[sketch_pad, sketch_pad_trigger, out_imagebox, image_scale, out_gen_1, out_gen_2, out_gen_3, out_gen_4, state],
        queue=False)
    task.change(
        partial(clear, switch_task=True),
        inputs=[task, sketch_pad_trigger, batch_size, state],
        outputs=[sketch_pad, sketch_pad_trigger, out_imagebox, image_scale, out_gen_1, out_gen_2, out_gen_3, out_gen_4, state],
        queue=False)
    sketch_pad_trigger.change(
        controller.init_white,
        inputs=[init_white_trigger],
        outputs=[sketch_pad, image_scale, init_white_trigger],
        queue=False)
    sketch_pad_resize_trigger.change(
        controller.resize_masked,
        inputs=[state],
        outputs=[sketch_pad, state],
        queue=False)
    batch_size.change(
        controller.change_n_samples,
        inputs=[batch_size],
        outputs=[out_gen_1, out_gen_2, out_gen_3, out_gen_4],
        queue=False)
    gen_btn.click(
        generate,
        inputs=[
            task, name, name2, split, grounding_instruction, sketch_pad,
            alpha_sample, guidance_scale, batch_size,
            fix_seed, rand_seed,
            use_actual_mask,
            append_grounding, style_cond_image,
            state
        ],
        outputs=[out_gen_1, out_gen_2, state],
        queue=True
    )
    sketch_pad_resize_trigger.change(
        None,
        None,
        sketch_pad_resize_trigger,
        _js=rescale_js,
        queue=False)
    init_white_trigger.change(
        None,
        None,
        init_white_trigger,
        _js=rescale_js,
        queue=False)
    use_style_cond.change(
        lambda cond: gr.Image.update(visible=cond),
        use_style_cond,
        style_cond_image,
        queue=False)
    task.change(
        controller.switch_task_hide_cond,
        inputs=task,
        outputs=[use_style_cond, style_cond_image, alpha_sample, use_actual_mask],
        queue=False)
    train_btn.click(
        train,
        inputs=[name, name2, epochs, model_type, model_type2],
        outputs=[df],
        queue=True)
    inf_btn.click(
        infer,
        inputs = [model_path, model_type, img],
        outputs = [out_inf_img, outybox],
        queue = True)

main.queue(concurrency_count=5, max_size=20).launch(share=True, show_api=False, show_error=True)



import os
import json
import torch
import base64
import pathlib
import numpy as np
from PIL import Image
from io import BytesIO
from PIL import Image, ImageOps
from diffusers import StableDiffusionInpaintPipeline
import tarfile
from transformers import pipeline

def img2img_preprocessor(org_img, mask_img):
    '''
    Computes a binary mask org img based on mask img
    # Two approaches to resisizing:
        1) Resize whilst keeping aspect ratio to the nearest 64
        2) Resize with padding and cropping to 512x512 then remove padding after
    '''
    def _resize_pad_crop(img, return_shape=False):
        img = ImageOps.contain(img, (512, 512))
        img_pad = Image.new(img.mode, (512, 512), 0)
        img_pad.paste(img, (512-img.width, 512-img.height))
        if return_shape:
            return img_pad, np.array(img).shape
        else:
            return img_pad

    # org_img = Image.open(BytesIO(base64.b64decode(org_img)))
    # mask_img = Image.open(BytesIO(base64.b64decode(mask_img)))
    org_img = Image.fromarray(np.array(org_img, dtype=np.uint8))
    mask_img = Image.fromarray(np.array(mask_img, dtype=np.uint8))
    org_img, org_shape = _resize_pad_crop(org_img, True)
    mask_img = _resize_pad_crop(mask_img).convert('L')
    mask_img = np.array(mask_img)
    mask_img[mask_img != 0] = 255
    mask_img = Image.fromarray(mask_img)
    return org_img, mask_img, org_shape


def img2img_postprocessor(img: Image, org_shape: list):
    '''
    Should return a json serializable object
    '''
    img = np.array(img)
    new_shape = img.shape
    img = img[new_shape[0]-org_shape[0]:, new_shape[1]-org_shape[1]:, :]
    # img = Image.fromarray(img)
    # buffered = BytesIO()
    # img.save(buffered, format="JPEG")
    # img_str = base64.b64encode(buffered.getvalue())
    img = np.array(img).tolist()
    return img


def model_fn(model_dir):
    # Check if weights unpacked
    if os.path.exists(f'/var/meadowrun/machine_cache/weights') == False:
        print('Extracting Files..')
        tar = tarfile.open('/var/meadowrun/machine_cache/stable-diffusion-v2-inpainting.tar.gz', "r:gz")
        tar.extractall(f'/var/meadowrun/machine_cache/')
        tar.close()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        f'/var/meadowrun/machine_cache/weights/stable-diffusion-2-inpainting',
        torch_dtype=torch.float16,
        # cache_dir='deployment/aws/models/stable-diffusion-2-inpainting/weights',
        local_files_only=True,
    )
    pipe.enable_attention_slicing()
    pipe = pipe.to(device)
    return pipe


def input_fn(request_body, request_content_type):
    data = json.loads(request_body)
    prompt = data['prompt']
    image = np.array(data['image'])
    mask_image = np.array(data['mask'])
    image, mask_image, original_shape = img2img_preprocessor(image, mask_image)
    return {
        'prompt': prompt,
        'image': image,
        'mask': mask_image,
        'original_shape': original_shape,
    }


def predict_fn(inputs, model):
    with torch.no_grad():
        prediction = model(
            inputs['prompt'], inputs['image'], inputs['mask']).images[0]

    return {'prediction': np.array(prediction), 'original_shape': inputs['original_shape']}


def output_fn(outputs, content_type):
    output_image = img2img_postprocessor(
        outputs['prediction'], outputs['original_shape'])
    return {'output_image': output_image}




def end2end_function(inputs):
    import time
    root = pathlib.Path(__file__).absolute().parent
    inputs = input_fn(inputs, None)
    start = time.perf_counter()
    print('Loading Model')
    model = model_fn(root.parent)
    print(f'Time to load model {time.perf_counter() - start}')
    print('Running Prediction')
    start = time.perf_counter()
    outputs = predict_fn(inputs, model)
    print(f'Time to perform inference {time.perf_counter() - start}')
    outputs = output_fn(outputs, None)

    image = np.array(outputs['output_image'], dtype=np.uint8)
    return image
#     image.save(f'./sample_io/out.png')



if __name__ == '__main__':
    inputs = json.dumps(
        {'prompt': 'Photo realistic image of a cat, high definition',
        'image': np.array(Image.open(f'./sample_io/test_img.png'))[:, :, :3].tolist(),
        'mask': np.array(Image.open(f'./sample_io/test_mask.png'))[:, :, :3].tolist()})

    end2end_function(inputs)

#     generator = pipeline('text-generation', model='distilgpt2')
#     output = generator('hello how are you doing!', max_length=80, num_return_sequences=2)
#     print(output[0]['generated_text'])

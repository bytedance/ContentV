## Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
## 
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
## 
##     http://www.apache.org/licenses/LICENSE-2.0
## 
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import os
import torch
USE_ASCEND_NPU = int(os.getenv('USE_ASCEND_NPU', '0'))
if USE_ASCEND_NPU:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan
from contentv_transformer import SD3Transformer3DModel
from contentv_pipeline import ContentVPipeline
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--model_id', type=str, default='ByteDance/ContentV-8B')
args = parser.parse_args()

model_id = args.model_id
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
transformer = SD3Transformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = ContentVPipeline.from_pretrained(model_id, vae=vae, transformer=transformer, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = 'A young musician sits on a rustic wooden stool in a cozy, dimly lit room, strumming an acoustic guitar with a worn, sunburst finish.'
negative_prompt = "overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"

video = pipe(
    num_frames=125,
    prompt=prompt,
    negative_prompt=negative_prompt,
    generator=torch.Generator().manual_seed(42),
).frames[0]
export_to_video(video, "video.mp4", fps=24)

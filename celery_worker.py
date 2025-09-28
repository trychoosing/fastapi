import time
from config import celery_app
import os
def load_qwen_VLM_model():

  import torch
  from PIL import Image
  from transformers import AutoProcessor, AutoModelForImageTextToText

  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  # Initialize processor and model
  processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
  model = AutoModelForImageTextToText.from_pretrained(
      "Qwen/Qwen2.5-VL-7B-Instruct",
      dtype=torch.bfloat16,
  ).to(DEVICE)
  return model,processor,DEVICE


def load_image_for_qwen(nowfile:os.PathLike):

  from transformers.image_utils import load_image
  image1 = load_image(nowfile )
  return image1
#define tasks

model1, processor1, DEVICE1 = load_qwen_VLM_model() 
def generate_text_from_image_VLM(model,
                                 processor,
                                 prompt,
                             image1,
                             DEVICE,):
  inputs = processor(text=prompt, images=[image1 ], return_tensors="pt")
  inputs = inputs.to(DEVICE)

  # Generate outputs
  generated_ids = model.generate(**inputs, max_new_tokens=5000)
  generated_texts = processor.batch_decode(
      generated_ids,
      skip_special_tokens=True,
  )
  text_gen =  generated_texts[0]
  return text_gen  
  
  
@celery_app.task
def long_running_task(prompt_1,
                      image  ):
    """Simulates a long-running task."""
     
  print('received')
  try:
    print('trying')
    with open(f"/fastapi/uf/{image.filename}", "wb") as buffer:
      shutil.copyfileobj(image.file, buffer)
  finally: 
    print('copy and close')
    image.file.close()
  image1 = load_image_for_qwen(f"/fastapi/uf/{image.filename}")
  prompt_11= [item.strip() for item in prompt_1[0].split('__**__')] 
  #print(prompt_11) 
  prompt_2 = def_prompt_with_task(prompt_11[0],processor1) 
  text_gen1 = generate_text_from_image_VLM(model1,
                                           processor1,
                                          prompt_2,
                                          image1,
                                          DEVICE1,)    
  return text_gen1
import time 

from transformers import AutoProcessor
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

        
def def_prompt_with_task(promptit_for_run:str,
                     processor:AutoProcessor):
  
  messages = [
      {
          "role": "user",
          "content": [
              {"type": "image"},
              {"type": "text", "text": promptit_for_run }
          ]
      },
  ]

  # Prepare inputs
  prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

  return prompt
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
  
   
def long_running_task(prompt_1,
                      image_p  ):
    """Simulates a long-running task.""" 
    image1 = load_image_for_qwen(image_p)  
    prompt_2 = def_prompt_with_task(prompt_1 ,processor1) 
    text_gen1 = generate_text_from_image_VLM(model1,
                                           processor1,
                                          prompt_2,
                                          image1,
                                          DEVICE1,)    
    return text_gen1
if __name__=="__main__":
    import os
    import time 
    mainfilepath = "/fastapi/uf/"
    model1, processor1, DEVICE1 = load_qwen_VLM_model() 
    while 1:
        nowfilepaths =  os.listdir(mainfilepath )
        full_paths = []
        nowfilepaths1 = [x for x in nowfilepaths if '.txt' not in x]
        try:
            item = nowfilepaths1[0] 
            
            with open(  os.path.join(mainfilepath,item+'.txt') ,'r') as ff:
                prompt1 = ff.read()
            image_path =   os.path.join(mainfilepath,item  )
            text_gen1 = long_running_task(prompt1,  image_path  )
            with open(  os.path.join(mainfilepath,item+'text_gen.txt'),'w') as ff:
                ff.write(text_gen1)
            os.system('rm '+mainfilepath+item+'.txt')
            os.system('rm '+mainfilepath+item )
            
            time.sleep(0.1)
         
        except Exception :
            
            time.sleep(1)
         
        
        

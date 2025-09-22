from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
import os
from transformers import AutoProcessor
import uvicorn 
from pydantic import BaseModel
import PIL.Image 
import requests
from fastapi import  File, UploadFile
from typing import Annotated 
from typing import List
# Define API endpoints
import shutil
import json
 

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

# Register routes using LangChain's utility function which integrates the chat model into the API.

# Load the model
app = FastAPI()
# Configure CORS middleware to allow all origins, enabling cross-origin requests.
# details: https://fastapi.tiangolo.com/tutorial/cors/
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

model1, processor1, DEVICE1 = load_qwen_VLM_model()


class task_definition():
      def __init__(self,
                   type_of_calendar_event,
                   do_or_note,
                   recurrence_type,
                   to_do_list,
                   additional_note):
        self.type_of_calendar_event = type_of_calendar_event
        self.do_or_note = do_or_note
        self.recurrence_type = recurrence_type
        self.to_do_list = to_do_list
        self.additional_note = additional_note
        
def def_prompt_with_task(task:task_definition,
                     processor:AutoProcessor):
  type_of_calendar_event = task.type_of_calendar_event
  do_or_note = task.do_or_note
  recurrence_type = task.recurrence_type
  to_do_list = task.to_do_list
  additional_note = task.additional_note
  
  messages = [
      {
          "role": "user",
          "content": [
              {"type": "image"},
              {"type": "text", "text": f"""Describe the {type_of_calendar_event} in the image. 
                                        
                                      
                                      Use date information in the image. i.e., especially where day, month and year are mentioned, ensure these are recorded and used to interpret the schedules/timetables in the image. The majority of the dates in a timetable will be labelled by the title month. If the first day seen is between 25-31 of the month, then it  refers to  the previous month if the rest of the timetable is the whole month of consecutive numbers.
                                      
                                      Write a table where each row is one event/period/routine/activity in the schedule/timetable :
                                      1. Column 1: Title of the event/period/routine/activity
                                      2. Column 2: Date and Time in DD-MMM-YYYY HH:MM format where HH is 24-hour format.
                                      3. Column 3: Estimated location: Both place, and specific sub-location 
                                      4. Column 4: Any additional notes relevant to the person 
                                      
                                      Use additional notes {additional_note} for interpreting the image.  
                                      

                                      """}
          ]
      },
  ]

  # Prepare inputs
  prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

  return prompt
 





@app.post("/submit")
async def submit(prompt_1:  List[str]   , image: UploadFile = File(...)):
  print('received')
  try:
    print('trying')
    with open(f"/fastapi/uf/{image.filename}", "wb") as buffer:
      shutil.copyfileobj(image.file, buffer)
  finally:
      
    print('copy and close')
    image.file.close()
  image1 = load_image_for_qwen(f"/fastapi/uf/{image.filename}")
  prompt_11= [item.strip() for item in prompt_1[0].split(',')]
  task_1 = task_definition(prompt_11[0],prompt_11[1],prompt_11[2],prompt_11[3],prompt_11[4] )
  prompt_2 = def_prompt_with_task(task_1,processor1)
  text_gen1 = generate_text_from_image_VLM(model1,
                                           processor1,
                                          prompt_2,
                                          image1,
                                          DEVICE1,)
  if os.path.exists( '/fastapi/uf/text_gen.txt')==True :
      return 'Wait'
  with open('/fastapi/uf/text_gen.txt','w') as f:
      f.write(text_gen1)
  return {'text_gen1':text_gen1}
  
@app.get("/liveness")
async def liveness():
    """
    Define a liveness check endpoint.

    This route is used to verify that the API is operational and responding to requests.

    Returns:
        A simple string message indicating the API is working.
    """
    import os
    if os.path.exists('/fastapi/uf/text_gen.txt')==True:
        with open('/fastapi/uf/text_gen.txt','r') as f:
          text_gen1 = f.read()
        os.system("rm /fastapi/uf/*")
        return  {'text_gen1':text_gen1}
    else:
        return 'liveness'
        
 
@app.get("/")
async def root():
    return {"message": "Welcome to GPU Worker FastAPI!"}

@app.get("/health")
async def health():
    return {"message": "ok"}
#if __name__ == "__main__":
#    uvicorn.run(app, host="127.0.0.1", port=8000)

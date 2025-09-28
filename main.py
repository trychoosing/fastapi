from fastapi import FastAPI, Body
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
from celery.result import AsyncResult
from config import celery_app
from celery_worker import long_running_task
 


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
 

def load_image_for_qwen(nowfile:os.PathLike):

  from transformers.image_utils import load_image
  image1 = load_image(nowfile )
  return image1
 

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    mainfilepath = "/fastapi/uf/"
    if os.path.exists(mainfilepath+task_id+"text_gen.txt") ==True:
        with open(mainfilepath+task_id+"text_gen.txt",'r') as ff:
            result = ff.read()
             
        os.system("rm "+mainfilepath+task_id+"text_gen.txt")
        return {"status": "completed", "result":  result}
    else:
        return {"status": "pending" }

import uuid
import datetime

@app.post("/submit")
async def submit(prompt_1:  List[str]   , image: UploadFile = File(...)):
  print('received')
   
  prompt_11= [item.strip() for item in prompt_1[0].split('__**__')] 
  uuig = prompt_11[1]
  try:
    print('trying')
    with open(f"/fastapi/uf/{uuig}", "wb") as buffer:
      shutil.copyfileobj(image.file, buffer)
  finally: 
    print('copy and close')
    image.file.close()
  uuig1=uuig+'.txt'
  with open(f"/fastapi/uf/{uuig1}" ,'w' ) as ff:
    ff.write(prompt_11[0])
   
  taskid =  prompt_11[1]
  return   {"message": "Task enqueued", "task_id": taskid} 
  
   
@app.get("/liveness")
async def liveness( ):
    """data: str = Body(...)
    Define a liveness check endpoint.

    This route is used to verify that the API is operational and responding to requests.

    Returns:
        A simple string message indicating the API is working.
    """
    import os 
  
    if os.path.exists('/fastapi/uf/text_gen.txt')==True:
        with open('/fastapi/uf/text_gen.txt','r') as f:
          text_gen1 = f.read()
        os.system("rm /fastapi/uf/text_gen.txt")
        return  {'text_gen1':text_gen1}
    else:
        return {'liveness':'liveness'}
        
 
@app.get("/")
async def root():
    return {"message": "Welcome to GPU Worker FastAPI!"}

@app.get("/health")
async def health():
    return {"message": "ok"}
#if __name__ == "__main__":
#    uvicorn.run(app, host="127.0.0.1", port=8000)

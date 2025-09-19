from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
import os

# Create an instance of FastAPI to serve as the main application.
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple API server using Langchain's Runnable interfaces",
)

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
from transformers import AutoProcessor
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

import uvicorn
# Register routes using LangChain's utility function which integrates the chat model into the API.

# Load the model
from fastapi import FastAPI
app = FastAPI()
model1, processor1, DEVICE1 = load_qwen_VLM_model()
from pydantic import BaseModel
import PIL.Image


import requests
from fastapi import  File, UploadFile
from typing import Annotated

from typing import List
# Define API endpoints
import shutil

@app.post("/submit")
async def submit(prompt_1:  List[str]   , image: UploadFile = File(...)):
  try:
    with open(f"/content/uploaded_files/{image.filename}", "wb") as buffer:
      shutil.copyfileobj(image.file, buffer)
  finally:
    image.file.close()
  image1 = load_image_for_qwen(f"/content/uploaded_files/{image.filename}")

  text_gen1 = generate_text_from_image_VLM(model1,
                                           processor1,
                                          prompt_1[0],
                                          image1,
                                          DEVICE1,)
  return {'text_gen1':text_gen1}

@app.get("/liveness")
async def liveness():
    """
    Define a liveness check endpoint.

    This route is used to verify that the API is operational and responding to requests.

    Returns:
        A simple string message indicating the API is working.
    """
    return  "message ok"
@app.get("/")
async def root():
    return {"message": "Welcome to GPU Worker FastAPI!"}

@app.get("/health")
async def health():
    return {"message": "ok"}
#if __name__ == "__main__":
#    uvicorn.run(app, host="127.0.0.1", port=8000)

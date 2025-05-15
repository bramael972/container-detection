from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.services.segmenter import process_image,calculate_fill_rate,destruct_cropped_img,PredictorSingleton 
from app.services.keypoints_runner import process_model_kpt_extract,PredictorKeySingleton
from app.services.optimizer_parallelepiped import run_optimization_para
from app.services.optimizer_trap import process_trapezoid_info
import os
from os import path
from azure.storage.blob import BlobServiceClient

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env.local")

BLOB_CONNECTION_STRING = os.getenv('BLOB_CONNECTION_STRING')
CONTAINER_NAME = os.getenv('CONTAINER_NAME')

def _download_blob_from_azure(blob_name, download_path):
    # Replace with your Azure Storage connection string
    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    
    # Download the blob to the specified path
    with open(download_path, "wb") as file:
        blob_client = container_client.get_blob_client(blob_name)
        download_stream = blob_client.download_blob()
        file.write(download_stream.readall())
    print(f"Downloaded {blob_name} to {download_path}")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)
class ImageURL(BaseModel):
    url: str

class ModelToDwnl(BaseModel):
    payload: List[str]

@app.on_event("startup")
def preload_models():
    try:
        print("Chargement du modèle segmenter et key...")
        PredictorSingleton.get_instance()
        PredictorKeySingleton.get_instance()
        print("Modèle segmenter et key prêt")
    except Exception as e:
        print("😅 tu dois charger les modèles proprement")

@app.post("/download-all-models")
def process_all_download_pth(data:ModelToDwnl):
    OUTPUT_DIR = "app/models"
    local_model_path_seg = os.path.join(OUTPUT_DIR, "segmentation/model_final.pth")
    local_model_path_key = os.path.join(OUTPUT_DIR, "keypoints/model_final.pth")
    if "segmentation" in data.payload:
        _download_blob_from_azure("segmentation/model_final.pth", local_model_path_seg)
    if "keypoints" in data.payload:
        _download_blob_from_azure("keypoint/model_final.pth" , local_model_path_key)
    preload_models()




@app.post("/process-image")
def process_image_back(payload: ImageURL):

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not payload.url:
        raise HTTPException(status_code=400, detail="Image URL is required")

    list_segment  =  process_image(payload.url,online=True) 
    result_mthd_fit = destruct_cropped_img(payload.url,online=True)

    data = []
    for el in list_segment:
        img = os.path.join(BASE_DIR,"..",el)
        print(img,"helo you")
        case_model = process_model_kpt_extract(img)
        #case_model_trap = process_trapezoid_info(case_model,img)
        case_opti =None# run_optimization_para(img)
        remplissage = None # calculate_fill_rate(img)
        data.append([case_model,case_opti,remplissage])

    

    return JSONResponse(content=[*data,result_mthd_fit])

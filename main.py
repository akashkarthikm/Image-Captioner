from fastapi import FastAPI, File, UploadFile
from ml_model import predictUsingImage
from pickle import dump
import PIL.Image as Image
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('*')
def home():
    return {'home' : 'home'}

@app.post('/predict_image')
async def predictImage(image : UploadFile = File(...)):
    print(image.filename)
    content = await image.read()
    bytes_content = bytearray(content)
    i = Image.open(io.BytesIO(bytes_content))
    i.save(f'./images/{image.filename}')
    # with open(f'./gui_uploaded_images/{image.filename}', 'wb') as imageFile:
    #     dump(content, imageFile)
    pred_desc = predictUsingImage(f'./images/{image.filename}')
    print(pred_desc)
    return {'predicted_text' : pred_desc}
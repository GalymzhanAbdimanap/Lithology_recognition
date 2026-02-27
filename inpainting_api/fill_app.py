import os
import numpy as np
import cv2
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn


from config import inpaint

# Initialize API
app = FastAPI(title="Fill service")

app_dir = os.path.dirname(os.path.realpath(__file__))

# Setup a checkpoint file to load
checkpoint = os.path.join(app_dir, 'models/hifill.pb')


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/fill", response_description="Fill Images",)
async def detection(image: UploadFile = File(...), mask: UploadFile = File(...), multiple: int = 6):
    """

    :param image: image that need to paint
    :param mask: mask image for painiting the image
    :param multiple: hyper parameter for image inpainting
    :return: return image with inpainiting
    """
    try:
        # Converting the uploaded file into cv image
        image_read = image.file.read()
        np_arr = np.frombuffer(image_read, np.uint8)
        raw_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # Converting the uploaded mask into cv image
        image_read = mask.file.read()
        np_arr = np.frombuffer(image_read, np.uint8)
        raw_mask = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        
        # Start inpanitting 
        inpainted = inpaint(raw_img, raw_mask, multiple, checkpoint)
        cv2.imwrite("results/"+image.filename + '_inpainted.png', inpainted)         
        # Returning the mask image encoded to png
        _, img_png = cv2.imencode(".png", inpainted)
        return StreamingResponse(io.BytesIO(img_png.tobytes()), media_type="image/png")       
    
    except Exception as e:
        raise HTTPException(500, "Internal Server Error: " + str(e))



if __name__ == '__main__':
    # run the app in port 5002
    uvicorn.run(app, host='0.0.0.0', port=5002)
    # from command line
    # uvicorn --host=0.0.0.0 --port=5002 --reload mask_app:app
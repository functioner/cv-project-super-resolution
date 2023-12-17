# python3 sr_image.py --model models/LapSRN_x8.pb --input original/set14 --output lapsrn

import argparse
import time
import cv2
import os

def super_resolve_image(model_path, image_path, output_folder):
    modelName = model_path.split(os.path.sep)[-1].split("_")[0].lower()
    modelScale = model_path.split("_x")[-1]
    modelScale = int(modelScale[:modelScale.find(".")])

    print("INFO: loading super resolution model: {}".format(model_path))
    print("INFO: model name {}".format(modelName))
    print("INFO: model scale {}".format(modelScale))
    
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel(modelName, modelScale)

    for filename in os.listdir(image_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            input_image_path = os.path.join(image_path, filename)
            output_image_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_lapsrn.png")
            
            image = cv2.imread(input_image_path)
            print("INFO: Processing", filename)
            
            start = time.time()
            upscaled = sr.upsample(image)
            end = time.time()
            
            print("INFO: Super resolution took {:.6f} seconds".format(end - start))
            cv2.imwrite(output_image_path, upscaled)
            print("INFO: Super-resolved image saved as", output_image_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="path to super resolution model")
    ap.add_argument("-i", "--input", required=True, help="path to the folder containing input images")
    ap.add_argument("-o", "--output", required=True, help="path to the folder to save super-resolved images")
    args = vars(ap.parse_args())

    super_resolve_image(args["model"], args["input"], args["output"])
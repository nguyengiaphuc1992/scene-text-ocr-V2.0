import os
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import io
import base64
import uvicorn
from paddleocr import PaddleOCR
from PIL import Image
import cv2
import numpy as np
import time
from utils import translate, order_boxes4nom,crop_img_boxes
import traceback
import PIL.Image
import PIL.ImageOps
import uuid
import json

# Specify the folder where your files are located
FILES_DIRECTORY = "uploads"


def remove_single_character_transcriptions(result_convert):
    # Tạo một danh sách mới để lưu kết quả đã loại bỏ
    filtered_result = []

    # Duyệt qua từng phần tử trong danh sách result_convert
    for item in result_convert:
        # Kiểm tra nếu độ dài của transcription lớn hơn 1
        if len(item['transcription']) > 2:
            # Nếu đúng, thêm phần tử này vào danh sách kết quả đã lọc
            filtered_result.append(item)

    # Trả về danh sách kết quả đã lọc
    return filtered_result

def convert_result(input_data):
    output_data = []
    for item in input_data:
        transcription = item[1][0]
        points = [[int(coord) for coord in point] for point in item[0]]
        output_data.append({"transcription": transcription, "points": points})
    return output_data

class LabelProcessor:
    def __init__(self, image_path, labels_json):
        self.image_path = image_path
        self.labels_json = labels_json

    def calculate_average_y(self, points):
        total_y = sum(point[1] for point in points)
        return total_y / len(points)

    def find_centroid(self):
        image = cv2.imread(self.image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
        return (centroid_x, centroid_y)

    def separate_labels(self, centroid_y):
        above_centroid = []
        below_centroid = []
        for label in self.labels_json:
            average_y = self.calculate_average_y(label["points"])
            if average_y < centroid_y:
                above_centroid.append(label)
            else:
                below_centroid.append(label)
        return above_centroid, below_centroid

    def sort_labels(self, labels):
        return sorted(labels, key=lambda x: x['points'][0][0], reverse=True)

    def save_labels_to_file(self, labels, file_name):
        with open(file_name, "w", encoding="utf-8") as file:
            for label in labels:
                file.write(f"Transcription: {label['transcription']}, Points: {label['points']}\n")

    def merge_label_files(self, above_file, below_file, merged_file):
        with open(above_file, "r", encoding="utf-8") as above_f, \
                open(below_file, "r", encoding="utf-8") as below_f, \
                open(merged_file, "w", encoding="utf-8") as merged_f:

            above_content = above_f.readlines()
            below_content = below_f.readlines()
            print("en(above_content) - len(below_content)", len(above_content),len(below_content) )
            if abs(len(above_content) - len(below_content)) > 3:
                return 1;
            else:
                merged_content = []
                if above_content == 0:
                        merged_content = below_content
                if below_content == 0:
                        merged_content = above_content
                max_lines = max(len(above_content), len(below_content))

                for i in range(max_lines):
                    if i < len(above_content):
                        merged_content.append(above_content[i])
                    if i < len(below_content):
                        merged_content.append(below_content[i])

                for line in merged_content:
                    merged_f.write(line + "\n")
                return 0;


def exif_transpose(img):
    """
    If an image has an Exif Orientation tag, transpose the image
    accordingly.

    Note: Very recent versions of Pillow have an internal version
    of this function. So this is only needed if Pillow isn't at the
    latest version.

    :param image: The image to transpose.
    :return: An image.
    """
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img

def image_to_numpy(
    file,
    mode='RGB'
):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array

    Defaults to returning the image data as a 3-channel array of 8-bit data. That is
    controlled by the mode parameter.

    Supported modes:
        1 (1-bit pixels, black and white, stored with one pixel per byte)
        L (8-bit pixels, black and white)
        RGB (3x8-bit pixels, true color)
        RGBA (4x8-bit pixels, true color with transparency mask)
        CMYK (4x8-bit pixels, color separation)
        YCbCr (3x8-bit pixels, color video format)
        I (32-bit signed integer pixels)
        F (32-bit floating point pixels)

    :param file: image file name or file object to load
    :param mode: format to convert the image to - 'RGB' (8-bit RGB, 3 channels), 'L' (black and white)
    :return: image contents as numpy array
    """

    img = file
    if hasattr(PIL.ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        img = PIL.ImageOps.exif_transpose(img)
    else:
        # Otherwise, do the exif transpose ourselves
        img = exif_transpose(img)

    img = img.convert(mode)

    return np.array(img)

def rgba2rgb(
    pil_img: Image,
)-> Image:
    """"""
    row, col = pil_img.size
    mode = pil_img.mode
    
    print(f'row:{row}\ncol: {col}\nmode: {mode}')

    np_img = np.asarray(
        a = pil_img,
        dtype= np.uint8
    )

    if mode == "RGB": # RGB
        return pil_img
    elif mode == "RGBA": # RGBA
        rgb = np.zeros(
            shape= (row, col, 3),
            dtype= np.float32
        )

        image, a = np_img[...,:3], np_img[...,3:]/255.0

        # Convert
        rgb = image * a
        res = np.asarray(image, np.uint8)

        return Image.fromarray(res)
    else:
        raise Exception("Error image format")





app = FastAPI()

class ImageRequest(BaseModel):
    base64Data: str

class ImageResponse(BaseModel):
    imageBase64: str
    ocrResult: list

ocr = PaddleOCR(
    lang='ch',
    rec=True,
    det_model_dir = './models/det',
    rec_model_dir='./models/rec',
    rec_char_dict_path='./models/rec/vocab.txt',
    vis_font_path='./static/NomNaTong-Regular.ttf',
    drop_score = 0.3
)

@app.post("/nom-ocr")
def ocr_documment(image_request: ImageRequest)->ImageResponse:
    s_time = time.time()
    # Decode base64 image data
    image_data = base64.b64decode(image_request.base64Data)



    # Open the image using PIL
    image = Image.open(io.BytesIO(image_data))


    # # endregion Save image
    try:
        image_array = image_to_numpy(image)
        fn = f'{str(uuid.uuid4())}.png'
        path = os.path.join(
                FILES_DIRECTORY,
                fn
            )
        img = Image.fromarray(image_array)
        img.save(path)
        # image_array =  np.asarray(image)


        #ocr hinh dua vao
        result = ocr.ocr(path)
        print("result88888888888888: ", result[0])
        result_text = []



#################################################
        #convert sang định dạng mới để sắp xếp các khối
        result_convert = convert_result(result[0])
        print ("result_convert: ",result_convert)

        result_convert_filtered = remove_single_character_transcriptions(result_convert)
        print("result_convert_filtered : ", result_convert_filtered)
        result_convert = result_convert_filtered


        label_processor = LabelProcessor(path, result_convert)

        # tinh trong tam
        centroid = label_processor.find_centroid()

        above_centroid_labels, below_centroid_labels = label_processor.separate_labels(centroid[1])

        # Sắp xếp các nhãn theo tọa độ x của điểm đầu tiên trong points theo chiều giảm dần
        above_centroid_labels = label_processor.sort_labels(above_centroid_labels)
        below_centroid_labels = label_processor.sort_labels(below_centroid_labels)

        # Lưu các nhãn vào các file txt tương ứng
        label_processor.save_labels_to_file(above_centroid_labels,
                                            "./result_txt/above_centroid_labels.txt")
        label_processor.save_labels_to_file(below_centroid_labels,
                                            "./result_txt/below_centroid_labels.txt")

        flag = label_processor.merge_label_files(
            "./result_txt/above_centroid_labels.txt",
            "./result_txt/below_centroid_labels.txt",
            "./result_txt/merged_labels.txt")
        print("flag :", flag)
        with open('./result_txt/merged_labels.txt', 'r', encoding='utf-8') as f:
            content = f.readlines()

            # Lọc và lấy ra chỉ các dòng chứa "Transcription" và loại bỏ phần "Points"
        transcriptions = [line.split(":")[1].split(",")[0].strip() for line in content if
                              line.startswith("Transcription")]
        print("transcriptions :", transcriptions)

        transcription_data_dasapxep = []

            # Ghi các nội dung "Transcription" vào một tệp tin mới
        with open('./result_txt/transcriptions.txt', 'w',
                      encoding='utf-8') as f:
                for transcription in transcriptions:
                    print("transcription sap xep:", transcription)
                    transcription_data_dasapxep.append(transcription.strip())
                    f.write(transcription + '\n')
        print("transcription_data_dasapxep :", transcription_data_dasapxep)







##############################################################
        # sort by bbox
        result = order_boxes4nom(result[0])
        print(f'result after sorf by bbox: {result}')
        bboxes = []
        for bbox in result:
                bb = bbox[0]
                bounding_box_coords = [(int(x), int(y)) for x, y in bb]
                bboxes.append(bounding_box_coords)

                t = bbox[1][0]
                result_text.append(t)

        print(f'List of bboxes: {bboxes}')
        print(f'List of texts: {result_text}')
        for box in bboxes:
                bbox = np.array(box, dtype=np.int32)
                bbox = bbox.reshape((-1, 1, 2))
                cv2.polylines(
                    image_array,
                    [bbox],
                    isClosed=True,
                    color=(0, 0, 255),
                    thickness=2
                )

        # region Save base64 image
        pil_image = Image.fromarray(image_array)
        # If the image has only one channel (grayscale), you can convert it to RGB
        if pil_image.mode == 'L':
            pil_image = pil_image.convert('RGB')
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")

        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        if flag == 0 and len(result_convert_filtered) > 15:
            result_text = transcriptions
            print("*********Có sắp xếp thứ tự************")

        # endregion
    except Exception as e:
        traceback.print_exc()
        print("exception: ", str(e))
        result_text = ""
        image_base64 = image_data






    response = ImageResponse(
        imageBase64=image_base64,
        ocrResult=result_text
    )
    return response

app.mount("/", StaticFiles(directory="static", html=True), name="static")

# CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8808)
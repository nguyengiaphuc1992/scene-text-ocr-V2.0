# Nom text OCR
Detecting and recognize Sino-Nom texts in famous Vietnamese Nom literary works such as The Tale of Kieu, Luc Van Tien, Dai Viet Su Ky Toan Thu,...

![illustrates the results of recognition and detection of Tale of Kieu](img/demo.png)

**Table of content:**
1. [Prerequisites](#item-one)
2. [Install](#item-two)
3. [Architechture](#item-three)
4. [License](#item-four)

<!-- headings -->
<a id="item-one"></a>

## This source nom-text-ocr V2.0 (update 01/05/2024)
- Update detection model 
- Fix the error that results in the wrong sentence order in the following Tale of Kieu

## 1. Prerequisites
To run the code, you need to install the following libraries:
- fastapi
- uvicorn
- Pillow
- paddleocr==2.7.0.0

<a id="item-two"></a>

## 2. Install
```sh
source <path_to_venv>
pip install -r requirements.txt
python main.py
```
**Example**:
```sh
source D:/Master/OCR_Nom/deploy/azure/str_vietnam_temple/.venv/Scripts/activate
pip install -r requirements.txt
python main.py
```

<a id ="item-three"></a>

## 3. Architechture
- Detection: Pretrain [ch_PP-OCRv4-det](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar) with our dataset for library. 
- Recognition: Pretrain [ch_PP-OCRv4-rec](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar) with our data.

<a id="item-four"></a>

## 4. License
MIT License

Copyright (c) 2024 CLC Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
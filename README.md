# DeepRule
Compete code of DeepRule
## Getting Started
```
git clone https://github.com/22222bh/DeepRule.git
cd DeepRule
conda create  --name DeepRule --file DeepRule.txt python=3.7

# torch is not installed. Please install torch with your own version.
# Tested version: Torch 1.10.0 with cuda 11.3
conda activate DeepRule
```

Download pretrained ChartOCR model from https://drive.google.com/file/d/1qtCLlzKm8mx7kQOV1criUbqcGnNh58Rr/view
and unzip it. Unzipped file should be named as **data**

Install pytesseract and tesseract
```
pip install pytesseract
sudo apt-get install tesseract-ocr-kor
```

Test
```
python3 test_final.py
```


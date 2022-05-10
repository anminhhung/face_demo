Web Face tracking with SORT

# Setup
pip3 install requirements.txt

## prprocess face 
```
python .\src\align_dataset_mtcnn.py .\dataset\raw\ .\dataset\processed\ --image_size 160 --margin 32 
```

## train model
```
python .\src\classifier.py TRAIN .\dataset\processed\ .\models\20180402-114759.pb .\models\facemodel.pkl --batch_size 32 
```
# Run app
python3 app.py

# check web
0.0.0.0:5100
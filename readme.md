# Requirements
python >= 3.8
other packages are in the requirements.txt

# Run
## 1. run data_anly.py
It will get the raw data from /data_folder and generate txt form train test validate dataset. You can change the settings about how to change the cleaning settings, the file is detailed referenced.

## 2. using "python augment.py --xxxxx"
Using this script will generate EDA data in the /cleaned_data folder

## run model.py
Run the model.py script, the detailed modelling process will be printed in the console, the model will be saved at /model_save and the pics will be saved at /aug_pics
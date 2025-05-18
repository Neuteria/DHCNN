# Dormitory Houseplan Convolutional Neural Network (DHCNN)

this is an application can generate floorplan with furniture base on CNN

(Caution: parameters have not been adjust carefully, outcomes may be bad)

## Config Settings

data for room's size, furniture details, train epoches and learning rate were stored in config.json

change config as you want to apply this application in different generate mission

## Create Datasets

you can run layout_editor.py to edit and save your layout design as datasets

or use the datasets given in folder "user_layouts" instead

`python layout_editor.py`

press R to switch furniture's direction during editing

## Generate Floorplan

you can run DHmain.py to train and generate layout samples

samples will be saved in folder "generated_layouts"

`python DHmain.py`

## View Floorplan

you can run layout_viewer.py to view JSON layouts generated or created

`python layout_viewer.py`
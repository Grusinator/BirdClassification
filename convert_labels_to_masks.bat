
set train_name=dag2_better_all

mkdir training_data/image_annotations_png/%train_name%

python xmllabel2img/xmllabel2img.py ^
--labelpath training_data/image_annotations_xml/%train_name%/annotations/ ^
--imagepath training_data/image_annotations_xml/%train_name%/images/ ^
--outputpath training_data/image_annotations_png/%train_name%

pause
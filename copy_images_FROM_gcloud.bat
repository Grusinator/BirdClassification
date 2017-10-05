

gcloud compute scp ^
	ubuntu1604-4cpu-1gpu:/home/grusinator/workspace/BirdClassification/evaluation/input/* ^
	H:\Projects\LiDAR_Bird_detection_2015\image_segmentation_development\BirdClassification\evaluation\input\ ^
	--recurse




gcloud compute ssh ubuntu1604-4cpu-1gpu --command ^
    "cd /home/grusinator/workspace/BirdClassification/ ^
    && python3 predict.py --input_path ./evaluation/input/ --output_path ./evaluation/output/"

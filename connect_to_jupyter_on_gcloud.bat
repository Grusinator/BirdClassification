

REM local_user@local_host$ ssh -N -f -L localhost:8888:localhost:8889 remote_user@remote_host


REM On gcloud run!
REM remote_user@remote_host$ ipython notebook --no-browser --port=8889


gcloud compute ssh ^
	--ssh-flag="-N -L localhost:8888:localhost:8889" ^
	--project "bird-classification" ^
	--zone "europe-west1-d" "ubuntu1604-4cpu-1gpu"
	
pause


REM local_user@local_host$ ssh -N -f -L localhost:8888:localhost:8889 remote_user@remote_host


REM On gcloud run!
REM remote_user@remote_host$ ipython notebook --no-browser --port=8889


gcloud compute ssh root@ubuntu1604-4cpu-1gpu --command ^
    "jupyter notebook --no-browser --port=8889 --allow-root"



REM gcloud compute ssh ^
REM	--ssh-flag="-N -L localhost:8888:localhost:8889" ^
REM	--project "bird-classification" ^
REM	--zone "europe-west1-d" ^
	REM --command ^
    REM "jupyter notebook --no-browser --port=8889 --allow-root " ^
REM	"ubuntu1604-4cpu-1gpu"
	
pause
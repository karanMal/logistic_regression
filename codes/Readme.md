
How to run Codes.



The data_prep.py is for preparing the data from DBPEDIA.
IN this repository the data is being prepared and uploaded so no need of running this script.

For running local mode : python local_train.py

For Distributed(Change ip for ps and worker nodes in codes)

1.  Bulk Synchronous Parallel mode

on ps node : python bsp.py --job_name="ps" --task_index=0 

on worker0 node: python bsp.py --job_name="worker" --task_index=0

on worker 1 node: python bsp.py --job_name="worker" --task_index=1

2.  Asynchronous Parallel mode

on ps node : python async.py --job_name="ps" --task_index=0

on worker0 node: python async.py --job_name="worker" --task_index=0

on worker 1 node: python async.py --job_name="worker" --task_index=1

3. Stale Synchronus Parallel mode(change stale value as required)

on ps node : python ssp.py --stale=32 --job_name="ps" --task_index=0

on worker0 node: python ssp.py --stale=32 --job_name="worker" --task_index=0

on worker 1 node: python ssp.py --stale=32 --job_name="worker" --task_index=1

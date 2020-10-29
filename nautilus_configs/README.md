## Nautilus cluster usage

Please read the instructions carefully as given in the cse291j-fa20-nautilus repository.
The templates for jobs is given in jobs.
The templates for pods is given in pods.

Some main commands include
- To create pod/job
```bash
kubectl create -f <NAME>.yaml
```
- To check pod/job status
```bash
kubectl get pods # jobs if job
```
- To check pod detailed status
```bash
kubectl describe pods # jobs if job
```
- To check particular pod status
```bash
kubectl get pods ${POD_NAME} # jobs if job
```
- To check particular pod's/jobs detailed status
```bash
kubectl describe pods ${POD_NAME} # jobs if job
```
- To delete a pod/job
```bash
kubectl delete pod ${POD_NAME} # jobs if job
```
- Portforwarding from 8888 port in pod to 8888 remote
```bash
kubectl port-forward ${POD_NAME} 8888:8888
```
- Download from the pod
```bash
kubectl cp ${POD_NAME}:${REMOTE_FILE} ${LOCAL_FILE}
```
- Upload to the pod
```bash
kubectl cp ${LOCAL_FILE} ${POD_NAME}:${REMOTE_FILE}
```

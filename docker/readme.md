# How to build an image for this project
This docker image contains a bit more that is needed for the model to run but it will be handy for all your pytorch/lightning projects.  
Before building an image you might want to customize it:  
 - change locale of operating system (it is set to Russian)
 - setup token to access jupyter lab in line 30 of Dockerfile `ENV JUPYTER_TOKEN="<your token here>"`
 - in case you ever need a username and password for the image it is `testuser\testuser`
 - install on your client PC a ubuntu font - jupyter lab is set to use it or change font in `plugin.jupyterlab-settings`

```sh
docker build . -t sasrec_torch
```
This will make an image `sasrec_torch`  
Run image with command 
```
docker run -it --name torch --gpus all --rm --privileged -p 8888:8888 -p 6006:6006 -v <host path>:/home/testuser/shared_folder sasrec_torch jupyter lab
```

---
toc: true
layout: post
description: DevOps note series
categories: [note, docker, devops]
title: "Docker Manual"
comments: true
---

## List of common Docker commands

Note: `CONTAINER_ID` can be `CONTAINER_NAME` in all the commands here.

```zsh
docker run <IMAGE_NAME>:<TAG>  # run in foreground
docker run -d <IMAGE_NAME>:<TAG>  # run in background
docker run -it <IMAGE_NAME>:<TAG>  # run in interactive mode

docker attach <CONTAINER_ID>  # attach a container in the background

docker ps  # show running containers
docker ps -a  # show all containers

docker stop <CONTAINER_NAME>  # can also use first characters of id
docker rm <CONTAINER_NAME>

docker images  # show images
docker rmi <IMAGE_NAME>  # remove image

docker pull <IMAGE_NAME>  # pull and not run an image

# run a command inside a container
docker exec <CONTAINER_ID> <some bash command>
```

Docker doesn't run OS, it runs processes. Once the process, e.g. a web server crashes, the container exits.

## `docker run` options

### Port mapping

```zsh
$ docker run -p <CONTAINER_INTERNAL_PORT>:<EXTERNAL_PORT> <IMAGE_NAME>:<TAG>
```

The host machine Docker is running on can use `EXTERNAL_PORT` to access the application with this command. The application runs on `CONTAINER_INTERNAL_PORT` inside the container, and it maps to the `EXTERNAL_PORT` of the host machine.

<img src="{{ site.baseurl }}/images/misc/docker-port-mapping.png" alt="port mapping" align="middle"/>

### Check container internal IP

```zsh
$ docker inspect <CONTAINER_ID>
```

and find `Networks: bridge: IPAddress`.

Now we can use this internal IP to access the application in the browser at `<INTERNAL_IP>:<INTERNAL_PORT>`.

Usually we use `localhost` on the host machine instead of the internal IP. One way of doing this on Mac or Windows is to use port forwarding for the VM Docker is running on. Check [here](https://www.jhipster.tech/tips/020_tip_using_docker_containers_as_localhost_on_mac_and_windows.html).

Another way is to have your application server run `-b 0.0.0.0`, and use `-p` to map the ports.

### Persist data and configuration

If we want to persist data and configuration, we need to map a volume from the container to the host machine. For example, if we want to run Jenkins with some plugins installed, we need to persist the state of Jenkins where they are installed. Use `docker run -v` to map a volume.

```zsh
$ docker run -p 8080:8080 -v /root/my-jenkins-data:/var/jenkins_home -u root jenkins
```

Here `/root/my-jenkins-data` is a custom volume I set, `/var/jenkins_home` is the default place Jenkins has its data in the container.

### Check OS version

```zsh
docker run ubuntu:17.10 cat /etc/*release*
```

## Docker images

### Dockerfile

For example, if we want to containerize a Flask app, we write a Dockerfile

<img src="{{ site.baseurl }}/images/misc/dockerfile-build.png" alt="dockerfile" align="middle"/>

A Dockerfile has lines of `DOCKER_INSTRUCTION command_to_run`.

```
FROM Ubuntu

RUN apt-get update
RUN apt-get install python

RUN pip install flask
RUN pip install flask-mysql

# This line copies the source code from the current directory on
# the host machine to the container's `/opt/source-code`
COPY . /opt/source-code

ENTRYPOINT FLASK_APP=/opt/source-code/app.py flask run
```

<img src="{{ site.baseurl }}/images/misc/docker-build.png" alt="docker build" align="middle"/>

We can push the image to Docker Hub by `docker push <image-name>`.

To build:

```zsh
$ docker build . -t <SPECIFY_IMAGE_NAME>
```

For pushing, do `docker login` and

```zsh
$ docker build . -t <username>/<image-name>
$ docker push <username>/<image-name>
```

### Environment variables

```zsh
$ docker run -e MYVAR1 --env MYVAR2=foo --env-file ./env.list

$ cat env.list
# This is a comment
VAR1=value1
VAR2=value2
USER
```

We can inspect a container by `docker inspect <container>` and check `Config:Env:` for existing environment variables.

### COMMAND vs. ENTRYPOINT

```
CMD command param1
# also
CMD ["command", "param1"]
```

Say we want to do

```zsh
$ docker run ubuntu sleep 5
```

which starts ubuntu and sleeps for 5 seconds. We can use `CMD` or `ENTRYPOINT` in the Dockerfile. Say we have image `ubuntu-sleeper` with Dockerfile:

```
FROM ubuntu
CMD sleep 5
```

The next time we do `docker run ubuntu-sleeper` it will starts ubuntu and sleeps for 5 seconds.

If we want to pass in the parameter, we use `ENTRYPOINT` instead of `CMD`. How `ENTRYPOINT` works is that the following commands will be appended to it. In contrast, `CMD` gets overridden by the command in `docker run` if there's any.

```
FROM ubuntu
ENTRYPOINT ["sleep"]
```

Now we can run it as `docker run ubuntu-sleeper 5`. If we don't supply the param here, it will give and error. To have a default value for `sleep`, we write

```
FROM ubuntu
ENTRYPOINT ["sleep"]
CMD ["5"]
```

In this case, `docker run ubuntu-sleeper 10` will override `5`. And without any parameter, it defaults to `5`.

## Docker Compose with YAML

If we need to run multiple services (mutiple `docker run` commands at the same time and `--link` them together), we can use `docker-compose`.

Using Compose is basically a three-step process:

1. Define your app’s environment with a `Dockerfile` so it can be reproduced anywhere.
2. Define the services that make up your app in `docker-compose.yml` so they can be run together in an isolated environment.
3. Run `docker-compose up` and Compose starts and runs your entire app.

A `docker-compose.yml` file that looks something like:

```
services:
    web:
        image: "<username>/simple-web-app"
    database:
        image: "mongodb"
    messaging:
        image: "redis:alpine"
    orchestration:
        image: "ansible"
```

Consider this app that has

- `vote-app`: Python web app for users to cast vote in browser
- `redis`: in-memory database using Redis for storing the votes from `vote-app`
- `worker`: .NET worker that puts the data in Redis into a Postgres db
- `db`: a Postgres db
- `result-app`: a Node.js app that shows the votes in browser

<img src="{{ site.baseurl }}/images/misc/sample-app.png" alt="sample app" align="middle"/>

In the old times, people use `docker run --link` commands to let `vote-app` and `result-app` to know where to find host names `redis` and `db` in their source code. Now we have `docker compose`.

<img src="{{ site.baseurl }}/images/misc/docker-compose.png" alt="docker compose" align="middle"/>

The above is for docker compose version 1. Then we have version 2 and 3. To specify version, you must add

```
version: 3
```

in `docker-compose.yml`. Since version 2, `services` is added, and `link` is deprecated, we should use `depends_on` instead. Version 3 supports Docker swarm which will be discussed later.

<img src="{{ site.baseurl }}/images/misc/docker-compose-ver.png" alt="docker compose versions" align="middle"/>

For a more advanced setting, we can separate the front end and back end networks for this voting application system. We add `Networks` section in the file and specify which network a service belongs to under each service.

<img src="{{ site.baseurl }}/images/misc/docker-networks.png" alt="docker networks" align="middle"/>

([YAML course](https://kodekloud.com/p/json-path-quiz))

## Docker registry


## Docker engine, storage, and networking


## Docker on Mac and Windows


## Container orchestration - Docker Swarm and Kubernetes


## Reference

- [Udemy course](https://www.udemy.com/course/learn-docker/)


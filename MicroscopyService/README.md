# Microscopy Service

## Introduction
This project houses the backend service for microscopy service. This provides an endpoint which exposes the Mitotic Model in AI Models project through a webapi. This is utilized by the frontend to get predictions on live microscopy based images.

## Build Instructions
1) Place pip file produced by building project in AI Models in root folder of  the project
2) docker build -t microscopy_service:latest [if you want to use docker compose please match the name of the service in docker_compose.yml file to the name you are using to build image]
The command would change to docker build -t service_name:latest

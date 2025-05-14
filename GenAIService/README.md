# GenAI Service

## Introduction
This project houses the generative ai based chat service which utilizes gemini as backend and also uses AIModels project to provide a streaming based chat service. This chat service is only maintained in sessions i.e if you reload the chat session would disappear and only one chat session is maintained in one browser connection i.e if you commence a new chat from frontend than the old chat disappears.
The backend folder contains the backend service and frontend folder contains the frontend service written in streamlit for easier testing.

## Build Instructions
### backend
1) Place pip file produced by building project in AI Models in root folder of  the project
2) docker build -t backend_service:latest

### fronted
1) docker build -t frontend_service:latest

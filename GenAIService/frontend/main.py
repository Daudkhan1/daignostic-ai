import json
import base64
from io import BytesIO

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8001"  # Update with your FastAPI backend URL

st.title("Chat Application")
st.session_state["user_id"] = "123"
st.session_state["user_input"] = ""

if "history" not in st.session_state:
    st.session_state["history"] = requests.get(
        f"{API_URL}/history", params={"user_id": st.session_state["user_id"]}
    ).json()

if st.button("Clear Chat"):
    requests.get(
        f"{API_URL}/clear", params={"user_id": st.session_state["user_id"]}
    ).json()

    st.rerun()


def display_images(encoded_images_list):
    for idx, image in enumerate(encoded_images_list):
        image_bytes = base64.b64decode(image)
        st.image(BytesIO(image_bytes), caption=str(idx + 1) + " Received Image")


# Display all chat history here
for item in st.session_state["history"]:
    with st.chat_message(item["role"]):
        if item["image"]:
            display_images(item["image"])

        st.markdown(item["message"])


# ----------------- Display chat box ----------------------
if prompt := st.chat_input(
    "Ask your question ...",
    file_type=["jpg", "jpeg", "png", ".tiff", ".tif"],
    accept_file=True,
):
    message = prompt["text"]
    files = prompt["files"]

    if not message:
        st.text("Must have question associated with image")

    # Encode image to str using b64
    encoded_images = []
    if files:
        file_bytes = files[0].read()
        encoded_images = [base64.b64encode(file_bytes).decode("utf-8")]

    with st.chat_message("Human"):
        if encoded_images:
            display_images(encoded_images)
        st.markdown(message)

    # Send response to API
    response = requests.post(
        f"{API_URL}/chat",
        json={
            "user_id": st.session_state["user_id"],
            "message": message,
            "image": encoded_images,
        },
        headers={"Content-Type": "application/json"},
        stream=True,
    )

    # render respone
    ai_generated_image = []
    assistant_message = st.chat_message("AI").empty()
    full_message = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if data["type"] == "text":
                full_message += data["content"]
                assistant_message.markdown(full_message)
            elif data["type"] == "image":
                display_images([data["content"]])
                ai_generated_image.append(data["content"])

    # ---------- Save in history -----------
    st.session_state["history"].append(
        {
            "role": "Human",
            "message": message,
            "image": encoded_images,
        }
    )

    st.session_state["history"].append(
        {"role": "AI", "message": full_message, "image": ai_generated_image}
    )

    st.rerun()

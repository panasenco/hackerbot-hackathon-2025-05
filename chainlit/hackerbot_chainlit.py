import os

import chainlit as cl
import cv2
from google import genai
from google.genai import types

gclient = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

VIDEO_CAPTURE_ID = 2

@cl.on_message
async def chat(message: cl.Message):
    cap = cv2.VideoCapture(VIDEO_CAPTURE_ID)
    cap.set(3, 640)
    cap.set(4, 480)
    _, frame = cap.read()
    cap.release()
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()
    image = cl.Image(content=image_bytes, name="Camera Image", display="inline")
    # Attach the image to the message
    await cl.Message(
        content="Current image from the on-arm camera",
        elements=[image],
    ).send()
    response = gclient.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
        message.content,
        ]
    )
    await cl.Message(content=response.text).send()
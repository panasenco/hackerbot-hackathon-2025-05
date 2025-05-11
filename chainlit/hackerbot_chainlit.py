import json
import os

import chainlit as cl
import cv2
from google import genai
from google.genai import types

gclient = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

VIDEO_CAPTURE_ID = 2
ANNOTATION_COLOR = (0,255,0)
ANNOTATION_THICKNESS = 2

async def send_image(name, frame):
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()
    image = cl.Image(content=image_bytes, name=name, display="inline")
    # Attach the image to the message
    await cl.Message(
        content=name,
        elements=[image],
    ).send()
    return image_bytes

async def prompt_next_action():
    await cl.Message(content="What would you like to do next?", actions=[
        cl.Action(name="action_chat", payload={}, label="Chat about the arm view"),
        cl.Action(name="action_locate", payload={}, label="Locate an object within the arm view"),
    ]).send()

@cl.action_callback("action_chat")
async def on_action_chat(_):
    cl.user_session.set("mode", "chat")
    await cl.Message(content="Enter your query:").send()

@cl.action_callback("action_locate")
async def on_action_locate(_):
    cl.user_session.set("mode", "locate")
    await cl.Message(content="Enter the object to locate:").send()

@cl.on_chat_start
async def start_chat():
    await prompt_next_action()

@cl.on_message
async def chat(message: cl.Message):
    cap = cv2.VideoCapture(VIDEO_CAPTURE_ID)
    cap.set(3, 640)
    cap.set(4, 480)
    _, frame = cap.read()
    cap.release()
    image_bytes = await send_image("Camera image", frame)
    mode = cl.user_session.get("mode") or "chat"
    if mode == "chat":
        prompt = message.content
    elif mode == "locate":
        prompt = f"""
Return bounding box around the following object if it exists as a 4-element array:
[xmin, ymin, xmax, ymax]
If you can't find the object, output:
null
NO OTHER TEXT. NO ``` WRAPPERS.
OBJECT: {message.content}
        """
    else:
        raise RuntimeError(f"Invalid interaction mode : '{mode}'")
    response = gclient.models.generate_content(
        model='gemini-2.5-flash-preview-04-17',
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
            prompt,
        ],
    )
    await cl.Message(content=response.text).send()
    if mode == "locate":
        print(f"{response.text=}")
        coords = json.loads(response.text)
        if coords is not None:
            annotated_frame = cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), ANNOTATION_COLOR, ANNOTATION_THICKNESS)
            await send_image("Annotated image", annotated_frame)
    await prompt_next_action()
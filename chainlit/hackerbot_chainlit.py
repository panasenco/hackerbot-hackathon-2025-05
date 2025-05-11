import json
from math import sqrt
import os

import chainlit as cl
import cv2
from google import genai
from google.genai import types
from hackerbot import Hackerbot

gclient = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

VIDEO_CAPTURE_ID = 2
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
CAPTURE_FOV_DEGREES = 120
DEGREES_X_MULTIPLIER = CAPTURE_WIDTH / sqrt(CAPTURE_WIDTH**2 + CAPTURE_HEIGHT**2) * CAPTURE_FOV_DEGREES
DEGREES_Y_MULTIPLIER = CAPTURE_HEIGHT / sqrt(CAPTURE_WIDTH**2 + CAPTURE_HEIGHT**2) * CAPTURE_FOV_DEGREES
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
    mode = cl.user_session.get("mode")
    await cl.Message(content=f"Set the interaction mode. The current mode is: {mode}", actions=[
        cl.Action(name="action_chat", payload={}, label="Chat about the arm view"),
        cl.Action(name="action_locate", payload={}, label="Locate an object within the arm view"),
        cl.Action(name="action_center", payload={}, label="Center the arm on an object"),
    ]).send()

async def move_hackerbot_arm(joints, speed):
    bot = Hackerbot()
    print(f"Moving the arm to {joints=}")
    bot.arm.move_joints(joints[0], joints[1], joints[2], joints[3], joints[4], joints[5], speed)
    cl.user_session.set("hackerbot_joints", joints)
    bot.base.destroy(auto_dock=False)

@cl.action_callback("action_chat")
async def on_action_chat(_):
    cl.user_session.set("mode", "chat")
    await cl.Message(content="Enter your query:").send()

@cl.action_callback("action_locate")
async def on_action_locate(_):
    cl.user_session.set("mode", "locate")
    await cl.Message(content="Enter the object to locate:").send()

@cl.action_callback("action_center")
async def on_action_center(_):
    cl.user_session.set("mode", "center")
    await cl.Message(content="Enter the object to center on:").send()

@cl.on_chat_start
async def start_chat():
    print("Resetting arm joints")
    await move_hackerbot_arm([0, 0, 0, 0, 0, 0], 0)
    print("Starting chat")
    cl.user_session.set("mode", "chat")
    await prompt_next_action()

@cl.on_message
async def chat(message: cl.Message):
    cap = cv2.VideoCapture(VIDEO_CAPTURE_ID)
    cap.set(3, CAPTURE_WIDTH)
    cap.set(4, CAPTURE_HEIGHT)
    _, frame = cap.read()
    cap.release()
    image_bytes = await send_image("Camera image", frame)
    mode = cl.user_session.get("mode")
    if mode == "chat":
        prompt = message.content
    elif mode in ["locate", "center"]:
        prompt = f"""
Locate the following object within the image: "{message.content}"
Return a 4-element NORMALIZED bounding box around the object:
[x_min, y_min, x_max, y_max]
Where x_min, y_min, x_max, and y_max are all between 0 and 1 from the top left corner of the image.
If you can't find the object, output null.
Feel free to use scratch space to think about the answer.
When you're ready to output the answer, prefix it with the string "BOUNDING BOX: ".
Please only output a valid 4-element array or `null`, no other text after "BOUNDING BOX: ".
        """
    else:
        raise RuntimeError(f"Invalid interaction mode : '{mode}'")
    response = gclient.models.generate_content(
        model='gemini-2.5-pro-preview-05-06',
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
            prompt,
        ],
    )
    await cl.Message(content=response.text).send()
    if mode in ["locate", "center"]:
        bounding_box_raw = response.text.split("BOUNDING BOX: ")[1]
        coords = json.loads(bounding_box_raw)
        if coords is not None:
            start_point = (int(coords[0]*CAPTURE_WIDTH), int(coords[1]*CAPTURE_HEIGHT))
            end_point = (int(coords[2]*CAPTURE_WIDTH), int(coords[3]*CAPTURE_HEIGHT))
            print(f"{start_point=}, {end_point=}")
            annotated_frame = cv2.rectangle(frame, start_point, end_point, ANNOTATION_COLOR, ANNOTATION_THICKNESS)
            await send_image(f"Annotated image ({start_point=}, {end_point=})", annotated_frame)
        if mode == "center":
            # Get the normalized position from center
            # x is measured from the left so left of center will be negative and right of center will be positive
            xc = ((coords[0] - 0.5) + (coords[2] - 0.5)) / 2
            # y is measured from the top so above the center will be negative and below will be positive
            yc = ((coords[1] - 0.5) + (coords[3] - 0.5)) / 2
            joints = cl.user_session.get("hackerbot_joints")
            # pan (joint 5) positive = left, negative = right
            pan = -xc * DEGREES_X_MULTIPLIER
            joints[4] += pan
            # tilt (joint 4) positive = up, negative = down
            tilt = -yc * DEGREES_Y_MULTIPLIER
            joints[3] += tilt
            print(f"{xc=}, {yc=}, {pan=}, {tilt=}, {joints=}")
            await cl.Message(content=f"Panning {pan} degrees to the left and tilting {tilt} degrees up").send()
            await move_hackerbot_arm(joints, 0)
    await prompt_next_action()
import base64

import cv2
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("hackerbot")

# Initialize the USB Camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)

@mcp.tool()
async def get_camera_image():
    """Gets the image from the Hackerbot on-arm camera
    """
    ret, frame = cap.read()
    jpg_img = cv2.imencode('.jpg', frame)
    b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')
    return {
      "type": "image",
      "data": b64_string,
      "mimeType": "image/jpeg",
    }

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
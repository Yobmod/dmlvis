import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image
from PIL import ImageTk
import cv2

from typing import Optional

panelA: Optional[Label] = None
panelB: Optional[Label] = None


def select_image() -> None:
    # grab a reference to the image panels
    global panelA, panelB

    # open a file chooser dialog and allow the user to select an input image
    path = filedialog.askopenfilename()

    # ensure a file path was selected
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)

        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # convert the images to PIL format...
        PILimage = Image.fromarray(image)
        PILedged = Image.fromarray(edged)

        # ...and then to ImageTk format
        image_tk = ImageTk.PhotoImage(PILimage)
        edged_tk = ImageTk.PhotoImage(PILedged)

        # if the panels are None, initialize them
        if not panelA:
            # the first panel will store our original image
            panelA = Label(image=image_tk)
        else:
            panelA.configure(image=image_tk)
        panelA.image = image_tk
        panelA.pack(side="left", padx=10, pady=10)

        if not panelB:
            panelB = Label(image=edged_tk)
        else:
            panelB.configure(image=edged_tk)
        panelB.image = edged_tk
        panelB.pack(side="right", padx=10, pady=10)


def process_image() -> None:
    ...


def save_image() -> None:
    ...


# initialize the window toolkit along with the two image panels
root = tk.Tk()


# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
"""create window with 9 blank panels (load with no image or empty image?)
e.g 
then select image loads image to first, process button loads next 2
keep reference to origin, edged, threshed, lightened, processed, plot in class?
"""
panel_orig = Label(image=blank_img)
panel_edged = Label(image=blank_img)

load_btn = tk.Button(root, text="Select an image", command=select_image)
load_btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

process_btn = tk.Button(root, text="Select an image", command=process_image)
process_btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

save_btn = tk.Button(root, text="Select an image", command=save_image)
save_btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

# kick off the GUI
root.mainloop()

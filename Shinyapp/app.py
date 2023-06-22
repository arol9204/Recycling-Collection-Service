
import numpy as np
import skimage
from PIL import Image, ImageOps
from shiny import App, Inputs, Outputs, Session, render, ui
from shiny.types import FileInfo, ImgData, SilentException


# theme


app_ui = ui.page_navbar(
    # Available themes:
    #  cerulean, cosmo, cyborg, darkly, flatly, journal, litera, lumen, lux,
    #  materia, minty, morph, pulse, quartz, sandstone, simplex, sketchy, slate,
    #  solar, spacelab, superhero, united, vapor, yeti, zephyr

    

    # Land tab App information -----------------------------
    ui.nav("App Information", 
          "Here we are planning to put information regarding the application"),
    
    # Request submission tab -------------------------------
    ui.nav("Request Submission", 
                   ui.layout_sidebar(
                       ui.panel_sidebar(
                                        ui.input_file(
                                                        "file",
                                                        "Please upload your picture here",
                                                        button_label="Open camera",
                                                        # This tells it to accept still photos only (not videos).
                                                        accept="image/*",
                                                        # This tells it to use the phone's rear camera. Use "user" for the front camera.
                                                        capture="environment",
                                                        ),
                                        ui.output_image("image"),
                                       ),
                       ui.panel_main(),
                   ),
          ),

    # Map for exploring requests ---------------------------
    ui.nav("Map"),
    
    
    
    title="RecyclingMates",
)



def server(input: Inputs, output: Outputs, session: Session):
    @output
    @render.image
    async def image() -> ImgData:
        file_infos: list[FileInfo] = input.file()
        if not file_infos:
            raise SilentException()

        file_info = file_infos[0]
        img = Image.open(file_info["datapath"])

        # # Resize to 1000 pixels wide
        # basewidth = 1000
        # wpercent = basewidth / float(img.size[0])
        # hsize = int((float(img.size[1]) * float(wpercent)))
        # img = img.resize((basewidth, hsize), Image.ANTIALIAS)

        # # Convert to grayscale
        # img = ImageOps.grayscale(img)

        # # Rotate image based on EXIF tag
        # img = ImageOps.exif_transpose(img)

        # # Convert to numpy array for skimage processing
        # image_data = np.array(img)

        # # Apply thresholding
        # val = skimage.filters.threshold_otsu(image_data)
        # mask = image_data < val

        # Save for render.image
        skimage.io.imsave("small.png", skimage.util.img_as_ubyte(img))
        return {"src": "small.png", "width": "100%"}




app = App(app_ui, server)

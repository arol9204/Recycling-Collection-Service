
import numpy as np
import skimage
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo, ImgData, SilentException


# Request tab
import tensorflow as tf
from CNN import predictions, probability_chart




# Map tab
from shiny import *
from shinywidgets import output_widget, render_widget
import ipyleaflet as L

basemaps = {
  "Satellite": L.basemaps.Gaode.Satellite,
  "OpenStreetMap": L.basemaps.OpenStreetMap.Mapnik,
  "Stamen.Toner": L.basemaps.Stamen.Toner,
  "Stamen.Terrain": L.basemaps.Stamen.Terrain,
  "Stamen.Watercolor": L.basemaps.Stamen.Watercolor,
}


# from pathlib import Path
# df = pd.read_csv(Path(__file__).parent / "salmon.csv")





# theme
#import shinyswatch

app_ui = ui.page_navbar(
    # Available themes:
    #  cerulean, cosmo, cyborg, darkly, flatly*, journal, litera, lumen, lux,
    #  materia, minty, morph, pulse, quartz, sandstone, simplex, sketchy, slate*,
    #  solar, spacelab*, superhero, united, vapor, yeti, zephyr*
    #shinyswatch.theme.zephyr(),
    

    # Land tab App information -----------------------------
    ui.nav("App Information", 
          "Here we are planning to put information regarding the application"),
    
    # Request submission tab -------------------------------
    ui.nav("Request Submission", 
                   ui.layout_sidebar(
                       ui.panel_sidebar(
                                        ui.input_file(  "file",
                                                        "Please upload your picture here",
                                                        button_label="Open camera",
                                                        # This tells it to accept still photos only (not videos).
                                                        accept="image/*",
                                                        # This tells it to use the phone's rear camera. Use "user" for the front camera.
                                                        capture="environment",
                                                        ),
                                        ui.output_image("image"),
                                       ),
                       ui.panel_main(
                                        ui.output_plot("plot1", click=True, dblclick=True, hover=True, brush=True),
                                    ),
                   ),
          ),

    # Map for exploring requests ---------------------------
    ui.nav("Map",
             ui.input_select(
                            "basemap", "Choose a basemap",
                            choices=list(basemaps.keys())
                            ),
            output_widget("map")
          ),
    
    
    
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
        #img = Image.open(file_info["datapath"])
        img = tf.keras.preprocessing.image.load_img(file_info["datapath"]) # load image
        img = tf.keras.preprocessing.image.save_img('img.jpg', img)
        #return img.show()
        return img
    @render.plot()
    def plot1():
        img = output.image()
        p = predictions(img)
        fig = probability_chart(p)
        return fig 
    @render_widget
    def map():
        basemap = basemaps[input.basemap()]
        return L.Map(basemap=basemap, center=[42.297471, -83.008058], zoom=7)




app = App(app_ui, server)

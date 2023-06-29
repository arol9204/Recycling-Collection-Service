
import numpy as np
#import skimage
import matplotlib.pyplot as plt
#from PIL import Image, ImageOps
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo, ImgData, SilentException


# Request tab
import tensorflow as tf

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('model_2_94p.h5')

def predictions(image):
     img_array = tf.keras.preprocessing.image.img_to_array(image) # convert the image to an NumPy array
     img_array_resized = tf.keras.preprocessing.image.smart_resize(img_array, size = (128, 128)) / 255.0 # if necessary, resize the image to 128 by 128
     reshaped_image = np.expand_dims(img_array_resized, axis=0) # reshape the image from (height, width, 3) to (1, height, width, 3) as an input to the CNN model
     probabilities = new_model.predict(reshaped_image)
     probabilities = np.round(probabilities[0,:],4 ) * 100
     return probabilities



# Map tab
from shiny import *
from shinywidgets import output_widget, render_widget
import ipyleaflet as L

basemaps = {
  "OpenStreetMap": L.basemaps.OpenStreetMap.Mapnik,
  "Stamen.Toner": L.basemaps.Stamen.Toner,
  "Stamen.Terrain": L.basemaps.Stamen.Terrain,
  "Stamen.Watercolor": L.basemaps.Stamen.Watercolor,
  "Satellite": L.basemaps.Gaode.Satellite,
}



# theme
import shinyswatch

app_ui = ui.page_navbar(
    # Available themes:
    #  cerulean, cosmo, cyborg, darkly, flatly*, journal, litera, lumen, lux,
    #  materia, minty, morph, pulse, quartz, sandstone, simplex, sketchy, slate*,
    #  solar, spacelab*, superhero, united, vapor, yeti, zephyr*
    shinyswatch.theme.zephyr(),
    

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
        img: ImgData = {"src": str(file_info["datapath"]), "width": "300px"}
        return img
    

    # request tab plot
    @output
    @render.plot()
    def plot1():
        file_infos: list[FileInfo] = input.file()
        if not file_infos:
            raise SilentException()

        file_info = file_infos[0]
        img = tf.keras.preprocessing.image.load_img(file_info["datapath"])
        p = predictions(img)
        x = np.array([0, 1, 2, 3])  # X-axis values
        labels = ['Can', 'Cardboard', 'Glass bottle', 'Plastic Bottle']  # Replace with your labels
        fig = plt.bar(x, p)
        fig = plt.xticks(x, labels)  # Assigning labels to x-axis ticks
        fig = plt.xlabel('Type of Object')
        fig = plt.ylabel('Probability')
        fig = plt.title('CNN Model Predictions')
        for i, value in enumerate(p):
           fig = plt.text(i, value, str(value) + "%", ha='center', va='bottom')
        return fig

    # Map Tab
    @output 
    @render_widget
    def map():
        basemap = basemaps[input.basemap()]
        return L.Map(basemap=basemap, center=[42.297471, -83.008058], zoom=9)


app = App(app_ui, server)

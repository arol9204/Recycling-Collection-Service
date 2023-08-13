import numpy as np
import matplotlib.pyplot as plt
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo, ImgData, SilentException
#import requests
import exifread # to extract image metadata

# Request tab
import tensorflow as tf

# Loading the model trainied
new_model = tf.keras.models.load_model('model_2_94p.h5')

def predictions(image):
     img_array = tf.keras.preprocessing.image.img_to_array(image) # convert the image to an NumPy array
     img_array_resized = tf.keras.preprocessing.image.smart_resize(img_array, size = (128, 128)) / 255.0 # if necessary, resize the image to 128 by 128
     reshaped_image = np.expand_dims(img_array_resized, axis=0) # reshape the image from (height, width, 3) to (1, height, width, 3) as an input to the CNN model
     probabilities = new_model.predict(reshaped_image)
     probabilities = np.round(probabilities[0,:],4 ) * 100
     return probabilities

# Defining the function to extract the metadata of the image
def get_gps_info(image):
    with open(image, 'rb') as image_file:
        tags = exifread.process_file(image_file, details=False)

    latitude = None
    longitude = None
    timestamp = None

    if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
        lat_parts = tags['GPS GPSLatitude'].values
        lon_parts = tags['GPS GPSLongitude'].values

        lat_deg = lat_parts[0].num / lat_parts[0].den
        lat_min = lat_parts[1].num / lat_parts[1].den
        lat_sec = lat_parts[2].num / lat_parts[2].den

        lon_deg = lon_parts[0].num / lon_parts[0].den
        lon_min = lon_parts[1].num / lon_parts[1].den
        lon_sec = lon_parts[2].num / lon_parts[2].den

        latitude = lat_deg + (lat_min / 60.0) + (lat_sec / 3600.0)
        longitude = lon_deg + (lon_min / 60.0) + (lon_sec / 3600.0)
    
    if tags['GPS GPSLongitudeRef'].values == 'W':
            longitude *= -1  # Convert to negative longitude

    if 'EXIF DateTimeOriginal' in tags:
        timestamp = tags['EXIF DateTimeOriginal'].values

    return latitude, longitude, timestamp

# Map tab
from shiny import *
from shinywidgets import output_widget, render_widget
import ipyleaflet as L
from ipyleaflet import Icon


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
          "This research focuses on developing an image-based classification system for recycling objects, targeting four key items: cardboard, tin, glass, and plastic bottles. Leveraging Convolutional Neural Networks (CNNs), our aim is to enhance waste sorting accuracy and efficiency through automated object classification. Unlike conventional methods, our approach classifies one item at a time, ensuring greater precision while simplifying the process."),
    
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
                                        ui.output_text_verbatim("lat"),
                                        ui.output_text_verbatim("lon"),
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
                            output_widget("map"),
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
        fig = plt.title('CNN Model Predictions \n *The class with the highest probability is the object class identified by the model in the picture.')
        for i, value in enumerate(p):
           fig = plt.text(i, value, str(value) + "%", ha='center', va='bottom')
        return fig

    
    
    @output
    @render.text
    def lat():
        file_infos: list[FileInfo] = input.file()
        if not file_infos:
            raise SilentException()
        file_info = file_infos[0]["datapath"]
        latitude = get_gps_info(file_info)[0]
        return f"Latitude: {latitude}"
    
    @output
    @render.text
    def lon():
        file_infos: list[FileInfo] = input.file()
        if not file_infos:
            raise SilentException()
        file_info = file_infos[0]["datapath"]
        longitude = get_gps_info(file_info)[1]
        return f"Longitude: {longitude}"
    
    # Map Tab
    @output 
    @render_widget
    def map():
        basemap = basemaps[input.basemap()]
        m = L.Map(basemap=basemap, center=[42, -83], zoom=9)
        marker = L.Marker(location=[42.31253333333333, -83.04131944444444], draggable=True)
        icon = Icon(icon_url='https://leafletjs.com/examples/custom-icons/leaf-green.png', icon_size=[20, 50], icon_anchor=[22,94])
        beer_store_mark1 = L.Marker(location=[42.31263551985872, -83.03326561020128], icon=icon, rotation_angle=90, rotation_origin='22px 94px', draggable=False)
        beer_store_mark2 = L.Marker(location=[42.30366417918876, -83.05465990194318], icon=icon, rotation_angle=90, rotation_origin='22px 94px', draggable=False)
        m.add_layer(marker)
        m.add_layer(beer_store_mark1)
        m.add_layer(beer_store_mark2)
        return m

app = App(app_ui, server)

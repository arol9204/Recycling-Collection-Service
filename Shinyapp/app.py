import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo, ImgData, SilentException
#import requests
import exifread # to extract image metadata

from asyncio import sleep

import cv2

# importing the model from Robolow
from roboflow import Roboflow
rf = Roboflow(api_key="F7o8gC2NLhuzMSLzk98A")
project = rf.workspace().project("recycling-objects-4aqr3")
model = project.version(3).model


from geopy.geocoders import Nominatim
# Create a Nominatim geocoder
geolocator = Nominatim(user_agent="reverse_geocoding")


# postgreSQL connexion
import psycopg2



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
    else:
        latitude = None
        longitude = None
    
    if 'EXIF DateTimeOriginal' in tags:
        timestamp = tags['EXIF DateTimeOriginal'].values
    else:
        timestamp = None

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





###############################################
# ------------     UI     ------------------- #
###############################################

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
                                        ui.input_action_button("detect", "Detect!"),

                                        

                                       ),
                       ui.panel_main(
                                        ui.output_plot("image_p"),
                                        
                                        ui.row(
                                                ui.column(4, "# Cans "),
                                                ui.column(4, "# Glass bottle"),
                                                ui.column(4, "# Plastic bottle"),
                                              ),
                                        ui.row(
                                                ui.column(4, ui.output_text_verbatim("can", placeholder=True)),
                                                ui.column(4, ui.output_text_verbatim("glass_bottle", placeholder=True)),
                                                ui.column(4, ui.output_text_verbatim("plastic_bottle", placeholder=True)),
                                              ),

                                        ui.row(
                                                ui.column(6, "Latitude"),
                                                ui.column(6, "Longitude"),
                                              ),
                                        
                                        ui.row(
                                                ui.column(6, ui.output_text_verbatim("lat", placeholder=True)),
                                                ui.column(6, ui.output_text_verbatim("lon", placeholder=True)),
                                              )
                                    ),
                   ),
                   ui.input_action_button("submit", "Submit Request!"),
          ),

    # Map for exploring requests ---------------------------
    ui.nav("Map",
             
                            ui.input_select(
                                            "basemap", "Choose a basemap",
                                            choices=list(basemaps.keys())
                                            ),
                            output_widget("map"),
                            
                            ui.output_table("requests"),
          ),
    
    title="Recycling Service Request",
)





###############################################
# ------------     SERVER ------------------- #
###############################################

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
    
    
    @reactive.Calc
    def predictions():
        file_infos: list[FileInfo] = input.file()
        if not file_infos:
            raise SilentException()

        path = file_infos[0]['datapath']

        input.detect()
        with reactive.isolate():
            predictions = model.predict(path, confidence=40, overlap=30)
                 
            return predictions
        
    # Here we count all the recyling objects in the image
    @reactive.Calc
    def classes():

        input.detect()
        with reactive.isolate():
            json_predictions = predictions().json()
            l_classes = []
            for i in json_predictions['predictions']:
                l_classes.append(i['class'])

        
            return l_classes
    
    # Getting the image path
    @reactive.Calc
    def image_path():
        file_infos: list[FileInfo] = input.file()
        
        # If there is NOT GPS information it should raise an exception message
        if not file_infos:
            raise SilentException()
        file_info = file_infos[0]["datapath"]
        return file_info

    # Returning the image after drawing the detection boxes
    @reactive.Calc
    def image_anotations():

        image = mpimg.imread(image_path())
       
        # infer on a local image
        json_predictions = predictions().json()

        for bounding_box in json_predictions['predictions']:
            x0 = bounding_box['x'] - bounding_box['width'] / 2
            x1 = bounding_box['x'] + bounding_box['width'] / 2
            y0 = bounding_box['y'] - bounding_box['height'] / 2
            y1 = bounding_box['y'] + bounding_box['height'] / 2
    
            start_point = (int(x0), int(y0))
            end_point = (int(x1), int(y1))

            cv2.rectangle(image, start_point, end_point, color=(255,0,0), thickness=5)
    
            cv2.putText(
                        image,
                        bounding_box["class"],
                        (int(x0), int(y0) - 10),
                        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 1.5,
                        color = (255, 0, 0),
                        thickness=3
                        )
        
        return image

    # Here we show the image with the annotations
    @output
    @render.plot()
    def image_p():
        input.detect()
        with reactive.isolate():
            return plt.imshow(image_anotations())

    
    @output
    @render.text()
    def can():
        input.detect()
        with reactive.isolate():
            return np.count_nonzero(np.array(classes()) == 'can')
    
    @output
    @render.text()
    def glass_bottle():
        input.detect()
        with reactive.isolate():
            return np.count_nonzero(np.array(classes()) == 'glass bottle')
    
    @output
    @render.text()
    def plastic_bottle():
        input.detect()
        with reactive.isolate():
            return np.count_nonzero(np.array(classes()) == 'plastic bottle')
    

    @output
    @render.text
    def lat():
        input.detect()
        with reactive.isolate():
            latitude = get_gps_info(image_path())[0]
        
            return latitude
    
    @output
    @render.text
    def lon():
        input.detect()
        with reactive.isolate():
            longitude = get_gps_info(image_path())[1]
            return longitude
    
    # Data Frame requests ---------------------------------------------------------
    df=reactive.Value(pd.DataFrame({'request_ID': [],
                                    'date': [],
                                   'cans': [],
                                   'glass_bottles': [],
                                   'plastic_bottles': [],
                                   'latitude': [],
                                   'longitude': [],
                                   'city': [],
                                   'province': [],
                                   'country': [],
                                   'path': []
                                   }))
    # # Selecting all the information from the request table
    # cursor.execute("SELECT * FROM requests")
    # record = cursor.fetchall()
    # df.set(pd.DataFrame(record, columns=["request_id", "n_cans", "n_glassbottles", "n_plasticbottles", "latitude", "longitude", "city", "province", "country", "image_path", "date_image"]))


    @reactive.Effect
    @reactive.event(input.submit)
    def add_value_to_list():
        

        if get_gps_info(image_path())[0] != None and get_gps_info(image_path())[1] != None:
            # Perform reverse geocoding to get the name of city, province and country of the lat and long point
            location = geolocator.reverse(f"{get_gps_info(image_path())[0]}, {get_gps_info(image_path())[1]}", exactly_one=True)
            
            # Extract city, province, and country
            if location:
                address = location.raw['address']
                city = address.get('city', '')
                province = address.get('state', '')
                country = address.get('country', '')
            else:
                city = province = country = 'Unknown'
            
           
            # PostgreSQL connexion
            connection = psycopg2.connect(user="postgres",
                              password="sqladmin",
                              host="127.0.0.1",
                              port="5432",
                              database="RECYCLING_DB")

            # This is neccesary to perform operations inside the database
            cursor = connection.cursor()
            # Selecting all the information from the request table
            cursor.execute("SELECT * FROM requests")
            record = cursor.fetchall()
            df.set(pd.DataFrame(record))
            request_id = df().shape[0] + 1


            # 2. Inserting values by predefining the values in a variable
            # We used a parameterized query to use Python variables as parameter values at execution time. Using a parameterized query, we can pass python variables as a query parameter using placeholders (%s).
            insert_query = """ INSERT INTO requests (request_id, n_cans, n_glassbottles, n_plasticbottles, latitude, longitude, city, province, country, image_path, date_image) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) """
            record_to_insert = (request_id, 
                                np.count_nonzero(np.array(classes()) == 'can'),
                                np.count_nonzero(np.array(classes()) == 'glass bottle'), 
                                np.count_nonzero(np.array(classes()) == 'plastic bottle'),
                                get_gps_info(image_path())[0],
                                get_gps_info(image_path())[1],
                                city,
                                province,
                                country,
                                image_path(),
                                get_gps_info(image_path())[2]
                                )
            cursor.execute(insert_query, record_to_insert)
            connection.commit()

            # Selecting all the information from the request table
            cursor.execute("SELECT * FROM requests")
            record = cursor.fetchall()
            df.set(pd.DataFrame(record, columns=["request_id", "n_cans", "n_glassbottles", "n_plasticbottles", "latitude", "longitude", "city", "province", "country", "image_path", "date_image"]))



            #print(record_to_insert) # here we can check if we are getting the information
            #print("1 Record inserted succesfully")
            cursor.close()
            connection.close()

            #df.set(pd.DataFrame(new_data))
        
        else:
            ui.notification_show("Sorry but as the image does not contain the localization information it can not be submitted as a request for collection")
            #await sleep(1)
            ui.notification_show("Warning message", type="warning")
        
    # Map Tab ---------------------------------------------------------------------
    @output 
    @render_widget
    def map():
        coin_icon = Icon(icon_url = 'https://raw.githubusercontent.com/arol9204/arol9204/main/images/1.cent.png', icon_size=[15, 15])
        leaf_icon = Icon(icon_url='https://leafletjs.com/examples/custom-icons/leaf-green.png', icon_size=[38, 95], icon_anchor=[22,94])
        # Now that we have the image information we can plot the map with the markers
        basemap = basemaps[input.basemap()]
        
        # If there are lat and long info in the last request open the map centered at the last request
        if get_gps_info(image_path())[0] != None and get_gps_info(image_path())[1] != None:
            m = L.Map(basemap=basemap, center=[get_gps_info(image_path())[0], get_gps_info(image_path())[1]], zoom=15)
            # Here is the lat and lon of the image that will be upload
            marker = L.Marker(location=[get_gps_info(image_path())[0], get_gps_info(image_path())[1]], draggable=True)
            # Adding the last request    
            m.add_layer(marker)
        # If there is not information center the map at the ip addres of the device (it needs more thinking)
        else:
            m = L.Map(basemap=basemap)

        # Beer stores markers
        beer_store_mark1 = L.Marker(location=[42.31263551985872, -83.03326561020128], icon=leaf_icon, draggable=False)
        beer_store_mark2 = L.Marker(location=[42.30366417918876, -83.05465990194318], icon=leaf_icon, draggable=False)
        m.add_layer(beer_store_mark1)
        m.add_layer(beer_store_mark2)

        # Create circle markers from the DataFrame
        for _, row in df().iterrows():
            request_marker = L.Marker(location=(row['latitude'], row['longitude']), icon=coin_icon, draggable=False)
            m.add_layer(request_marker)
        return m

  
    @output
    @render.table
    def requests():
        #df = df.append(new_request(), ignore_index=True)
        return df()

app = App(app_ui, server)


#print("PostgreSQL connection is closed")


# When the image do not have lat and long it throw and error (DONE)
    # But now the map needs to be reactive, so if there is not lat nor long in the last image, it should still show the previous submissions
# When an object wasn't detected it is necesary to pop up a message saying that it happened and allow the user to select if it was an error or not, also i case that it was true that there is not a recycling object do not sumbit a request
# Make the map reactive, update it only if a new request with lat and long is submitted

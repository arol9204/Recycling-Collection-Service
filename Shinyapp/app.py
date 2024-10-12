import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo, ImgData, SilentException

# Map tab
from shiny import *
from shinywidgets import output_widget, render_widget, render_pydeck
import ipyleaflet as L
from ipyleaflet import Icon, MarkerCluster


import faicons
from faicons import icon_svg

#from shinywidgets import value_box

#import requests
import exifread # to extract image metadata

from PIL import Image

from asyncio import sleep

import cv2


# ----------------------------------#
# Loading Computer Vision Models ---# 
# ----------------------------------#


# # Loading custom YOLOv11 model -------------------------
# from ultralytics import YOLO
# model_path = 'D:/3. Projects/RecylingNet/Models/YOLO Models/YOLOv11s/200 epochs/best.pt'
# model = YOLO(model_path)



# Load your custom YOLOv5 model ---------------------------------------
# Importing the trained YOLOv5 model in local folder
import torch
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model_path = 'C:/Users/alero/Documents/GitHub/Recycling-Collection-Service/Shinyapp/Local/models/YOLOv5/best.pt'  # Update this with the path to your trained YOLOv5 model
yolov5_dir = 'C:/Users/alero/Documents/GitHub/Recycling-Collection-Service/Shinyapp/yolov5'  # Update this with the path to your cloned YOLOv5 directory
model = torch.hub.load(yolov5_dir, 'custom', path=model_path, source='local', force_reload=True)




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

basemaps = {
    "WorldImagery": L.basemaps.Esri.WorldImagery,
    "Mapnik": L.basemaps.OpenStreetMap.Mapnik,
    "Positron": L.basemaps.CartoDB.Positron,
    "DarkMatter": L.basemaps.CartoDB.DarkMatter,
    "NatGeoWorldMap": L.basemaps.Esri.NatGeoWorldMap,
    "France": L.basemaps.OpenStreetMap.France,
    "DE": L.basemaps.OpenStreetMap.DE,
}

choices = {"WI": "WorldImagery",
           "MN": "Mapnik", 
           "PO": "Positron", 
           "NGM": "NatGeoWorldMap", 
           "HM": "Heatmap"}


# theme
import shinyswatch

from pathlib import Path


###############################################
# ------------     UI     ------------------- #
###############################################

app_ui = ui.page_navbar(
    # Available themes:
    #  cerulean, cosmo, cyborg, darkly, flatly*, journal, litera, lumen, lux,
    #  materia, minty, morph, pulse, quartz, sandstone, simplex, sketchy, slate*,
    #  solar, spacelab*, superhero, united, vapor, yeti, zephyr*
    #shinyswatch.theme.zephyr(),
    
    # Land tab App information -----------------------------
    ui.nav_panel("App Information",
                ui.page_fluid(
        ui.h1("Recycling Collection Service", class_="text-center", style="font-weight: bold; color: #2E8B57;"),
        ui.output_image("UI_image1"),
        ui.markdown(
            """
            Welcome to the Recycling Collection Service app! Our goal is to promote responsible recycling by using technology to make the recycling process easier and more efficient. This app leverages the power of computer vision and location-based data to help users identify recyclable items and contribute to the community's sustainability efforts.
            """,
            
        ),
        ui.h2("What is the App's Purpose?", style="color: #4CAF50; margin-top: 30px;"),
        ui.markdown(
            """
            The Recycling Collection Service app aims to assist you in identifying recyclable objects such as cans, glass bottles, and plastic bottles. By using advanced image-based detection models, the app helps sort these items efficiently, improving waste management and enhancing recycling efforts. The goal is to help create a cleaner and more sustainable environment.
            """,
            
        ),
        ui.h2("Roadmap to Using the App", style="color: #4CAF50; margin-top: 30px;"),
        ui.markdown(
            """
            Follow these simple steps to use the Recycling Collection Service app effectively:
            """,
            
        ),
        ui.tags.ol(
            ui.tags.li(
                ui.tags.div(
                    ui.output_image("UI_image2"),
                    style="text-align: center;"
                
                ),
                ui.tags.b("Upload Your Picture"),
                ": Start by clicking on the \"Request Submission\" tab and uploading an image of an item you would like to recycle. You can capture the image using your phone's camera or select one from your gallery.",
                
            ),
            ui.tags.li(
                ui.tags.div(
                    ui.output_image("UI_image3"),
                    style="text-align: center;"
                ),
                ui.tags.b("Detect the Item"),
                ": After uploading the picture, click the \"Detect!\" button to let the app identify the item in the image. The app utilizes a custom YOLO model to detect cans, glass bottles, and plastic bottles, and it will show you the detection results with bounding boxes.",
                
            ),
            ui.tags.li(
                ui.tags.div(
                    ui.output_image("UI_image4"),
                    style="text-align: center;"
                ),
                ui.tags.b("Submit a Request"),
                ": If you wish to add your detected items to the community map, click the \"Submit Request!\" button. This will add your contribution to the recycling request map, which helps the community understand areas with high recycling activity.",
                
            ),
            ui.tags.li(
                ui.tags.div(
                    ui.output_image("UI_image5"),
                    style="text-align: center;"
                ),
                ui.tags.b("View Map and Dashboard"),
                ": Once your request is submitted, explore the \"Map\" tab to see all community contributions, and visit the \"Requests Dashboard\" to get insights into recycling activities.",
                
            )
        ),
        ui.h2("Features of the App", style="color: #4CAF50; margin-top: 30px;"),
        ui.tags.ul(
            ui.tags.li(
                ui.tags.b("Real-time Detection"),
                ": The app uses cutting-edge object detection technology to identify recyclable items accurately.",
                
            ),
            ui.tags.li(
                ui.tags.b("Interactive Map"),
                ": On the \"Map\" tab, you can explore all the requests submitted by users in your area. The map displays the locations of recyclable items detected, allowing the community to visualize local recycling efforts.",
                
            ),
            ui.tags.li(
                ui.tags.b("Dashboard Insights"),
                ": The \"Requests Dashboard\" provides valuable insights into the total recycling requests, the number of different items collected, and visualizations of recycling activities over time.",
                
            ),
        ),
        ui.h2("Get Involved!", style="color: #4CAF50; margin-top: 30px;"),
        ui.markdown(
            """
            Your participation matters! Each submission helps improve local waste management and promotes responsible recycling. You can track your contributions, explore recycling hotspots, and get a clearer picture of the collective impact our community is making.

            Check out the [GitHub Repository](https://github.com/arol9204/Recycling-Collection-Service) for more information and source code. Let's make recycling smarter and our community greener, together!
            """,
            
        ),
        ui.h2("Questions or Feedback?", style="color: #4CAF50; margin-top: 30px;"),
        ui.markdown(
            """
            If you have any questions or feedback, feel free to reach out! Together, we can make a significant impact on reducing waste and promoting sustainability.
            """,
            
        )
                ),


    ),

    # Request submission tab -------------------------------
    ui.nav_panel("Request Submission", 
           ui.page_fluid(
               ui.card(
                   ui.layout_sidebar(
                                     ui.sidebar(
                                        
                                        ui.input_file(  "file", 
                                                                "Upload your picture here",
                                                                button_label="Open camera",
                                                                # This tells it to accept still photos only (not videos).
                                                                accept="image/*",
                                                                # This tells it to use the phone's rear camera. Use "user" for the front camera.
                                                                capture="environment",
                                                    ),
                                        ui.card(ui.output_image("image", width='100%', height='50%',),),
                                        
                                        ui.layout_column_wrap(
                                                ui.input_action_button("detect", "Detect!"),
                                                ui.input_action_button("submit", "Submit Request!"),
                                            ),
                                            open='open',
                                     ),
                       #ui.panel_main(
                          #{"style": "background-color: rgba(0, 255, 128, 0.1)"},
                                    ui.card("Model detections",
                                            ui.card(
                                                    ui.output_plot("image_p"), width='100%', height='100%', fill=True
                                                    ),
                                            ui.layout_column_wrap(
                                                         ui.value_box(
                                                                        "Cans",
                                                                        ui.output_text("can"),
                                                                        theme="bg-gradient-red-yellow",
                                                                        showcase=icon_svg("jar"), # trash-can-arrow-up, trash, dumpster
                                                                        height="150px",
                                                                    ),
                                                         ui.value_box(
                                                                        "Glass Bottles",
                                                                        ui.output_text("glass_bottle"),
                                                                        theme="bg-gradient-yellow-blue",
                                                                        showcase=icon_svg("wine-bottle"),
                                                                        height="150px",
                                                                    ),
                                                         ui.value_box(
                                                                        "Plastic Bottles",
                                                                        ui.output_text("plastic_bottle"),
                                                                        theme="bg-gradient-cyan-blue",
                                                                        showcase=icon_svg("bottle-water"),
                                                                        height="150px",
                                                                    ),
                                                         ),
                                        
                                        ui.row(
                                            
                                                ui.column(3, "Latitude"),
                                                ui.column(3, ui.output_text_verbatim("lat", placeholder=True)),
                                                ui.column(3, "Longitude"),
                                                ui.column(3, ui.output_text_verbatim("lon", placeholder=True)),
                                              ),
                                        
                                        
                                    #),
                                    ),
                                    {"style": "background-color: rgba(0, 255, 128, 0.1)"},
                   ),
                ),
             ),
          ),
    
    

    # Map for exploring requests ---------------------------
    ui.nav_panel("Map",
                ui.page_fluid(
                    ui.row(
                            ui.input_select(
                                            "basemap", "Choose a basemap",
                                            choices=list(basemaps.keys())
                                            ),
                            #ui.input_radio_buttons("type_map", "Type of Map", choices),
                            output_widget("map", height="700px"),
                    ),
                    ui.row(
                            ui.output_table("requests", placeholder=True),
                    ),
                ),
          ),


    # Dashboard -------------------------------------------------

    ui.nav_panel("Requests Dashboard",
           {"style": "background-color: rgba(0, 255, 128, 0.1)"},

           ui.page_fluid(
               ui.layout_sidebar(
                   ui.sidebar(
                                ui.input_date_range("daterange", "Date Range:", start="2023-01-01"), 
                                ui.input_checkbox_group(  
                                                            "checkbox_group",  
                                                            "Select Recycling Objects:",  
                                                            {  
                                                                "c": "Cans",  
                                                                "gb": "Glass Bottles",  
                                                                "pb": "Plastic Bottles",  
                                                            },
                                                            selected=['c', 'gb', 'pb'],
                                                        ),
                                ui.input_radio_buttons(  
                                                        "radio_fig1",  
                                                        "Fig 1 Plot Type:",  
                                                        {"1": "Bar plot", "2": "Line plot"},  
                                                    ),
                                ui.input_radio_buttons(  
                                                        "radio_fig2",  
                                                        "Fig 2 Map Type",  
                                                        {"1": "Heat Map", "2": "High-scale spatial Map"},  
                                                    ),
                                #ui.input_action_button("action_button", "Apply"),  
                            ),
                            ui.card(
                                    ui.value_box(
                                                    "Total Requests",
                                                    ui.output_text("total_requests"),
                                                    theme="bg-gradient-teal-green",  #gradient-cyan-teal
                                                    showcase=icon_svg("recycle"), # trash-can-arrow-up, trash, dumpster
                                                    height="100px",
                                                    ),

                                    ui.layout_column_wrap(
                                                            ui.value_box(
                                                                        "Total Cans",
                                                                        ui.output_text("total_cans"),
                                                                        theme="bg-gradient-red-yellow",
                                                                        showcase=icon_svg("jar"), # trash-can-arrow-up, trash, dumpster
                                                                        height="100px",
                                                                    ),
                                                            ui.value_box(
                                                                        "Total Glass Bottles",
                                                                        ui.output_text("total_glassbottles"),
                                                                        theme="bg-gradient-yellow-blue",
                                                                        showcase=icon_svg("wine-bottle"),
                                                                        height="100px",
                                                                    ),
                                                            ui.value_box(
                                                                        "Totla Plastic Bottles",
                                                                        ui.output_text("total_plasticbottles"),
                                                                        theme="bg-gradient-cyan-blue",
                                                                        showcase=icon_svg("bottle-water"),
                                                                        height="100px",
                                                                    ),
                                                            ),
                                ),
                            ui.card(
                                ui.layout_column_wrap(  "Fig 1: Recycling Requests by Date",
                                                        "Fig 2: Density Map",
                                                     ),
                                ui.layout_column_wrap(
                                                        output_widget("requests_by_date"),
                                                        output_widget("chart2"),
                                                     )
                            ),
               )
           
           ),
    ),
    
    title="Recycling Service Request",
)


###############################################
# ------------     SERVER ------------------- #
###############################################
def server(input: Inputs, output: Outputs, session: Session):

    # Landing page images
    @render.image  
    def UI_image1():
        img = {"src": 'C:/Users/alero/Documents/GitHub/Recycling-Collection-Service/Assets/AI generated images/DALL.png', "height":'400px', "width":'100%'}  
        return img 
    
    @render.image  
    def UI_image2():
        img = {"src": 'C:/Users/alero/Documents/GitHub/Recycling-Collection-Service/Assets/AI generated images/DALL·E step2.webp',"height":'200px', "width":'25%'}  
        return img 

    @render.image  
    def UI_image3():
            img = {"src": 'C:/Users/alero/Documents/GitHub/Recycling-Collection-Service/Assets/AI generated images/DALL·E step3.webp',"height":'200px', "width":'25%'}  
            return img 
    
    @render.image  
    def UI_image4():
        img = {"src": 'C:/Users/alero/Documents/GitHub/Recycling-Collection-Service/Assets/AI generated images/DALL·E step4.webp',"height":'200px', "width":'25%'}  
        return img 

    @render.image  
    def UI_image5():
        img = {"src": 'C:/Users/alero/Documents/GitHub/Recycling-Collection-Service/Assets/AI generated images/DALL·E step4.webp',"height":'200px', "width":'25%'}  
        return img 















    # Ploting the uploaded image
    @render.image
    async def image() -> ImgData:
        file_infos: list[FileInfo] = input.file()
        if not file_infos:
            raise SilentException()

        file_info = file_infos[0]
        img: ImgData = {"src": str(file_info["datapath"]), "width": "100%", "height": '400px'}
        return img
    
    # Getting the image path
    @reactive.Calc
    def image_path():
        file_infos: list[FileInfo] = input.file()
        # If there is NOT GPS information it should raise an exception message
        if not file_infos:
            raise SilentException()
        file_info = file_infos[0]["datapath"]
        return file_info
        
    # Getting the prediction from the object detection model
    @reactive.Calc
    def predictions():
        file_infos: list[FileInfo] = input.file()
        if not file_infos:
            raise SilentException()
        path = file_infos[0]['datapath']
        input.detect()
        with reactive.isolate():
            # Using my YOLOv5 model ----------
            predictions = model(path)
            print(predictions)
            return predictions
        
    # Here we put into a list all the recyling object classes in the image
    @reactive.Calc
    def classes():
        input.detect()
        with reactive.isolate():
            # With my YOLOv5 model ----------
            results = predictions().pandas().xyxy[0]
            l_classes = results['name'].tolist()
            print(results)
            return l_classes


    # Drawing the detection boxes in the image uploaded
    @reactive.Calc
    def image_anotations():
        
        # Load the image
        image = mpimg.imread(image_path())

        # Rotate the image 90 degrees clockwise
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # Using YOLOv5 output results ---------------
        # Iterate over the results DataFrame
        results_df = predictions().pandas().xyxy[0]
        for _, pred in results_df.iterrows():
            x0, y0, x1, y1 = int(pred['xmin']), int(pred['ymin']), int(pred['xmax']), int(pred['ymax'])
            label = pred['name']
            confidence = pred['confidence']

            # Set box color based on the class label
            if label == 'can':
                box_color = (255, 0, 0)
            elif label == 'glass bottle':
                box_color = (255, 255, 0)
            elif label == 'plastic bottle':
                box_color = (0, 255, 255)
            else:
                box_color = (255, 255, 255)  # default to white if no class match

            # Draw the bounding box
            cv2.rectangle(image, (x0, y0), (x1, y1), color=box_color, thickness=5)

            # Create the label with confidence score
            label_with_confidence = f"{label}: {confidence:.0%}"

            # Put the class label on the bounding box
            cv2.putText(
                image,
                label_with_confidence,
                (x0, y0 - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=5,
                color=box_color,
                thickness=10
            )
        # Convert image back to RGB for displaying with matplotlib
        #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        return image

    # Plotting the image with the annotations
    @render.plot
    #@render.image
    def image_p():
        input.detect()
        with reactive.isolate():

            # Using my trained YOLOv5 model -------------------
            # Annotate the image
            annotated_image = image_anotations()

            # Plot the annotated image using matplotlib
            #fig, ax = plt.subplots(figsize=(10, 10), dpi=100)  # Specify DPI
            fig, ax = plt.subplots()  # Specify DPI
            plt.imshow(annotated_image)
            plt.axis('off')  # Hide axis
            #plt.gcf()
            #plt.gcf()
            return fig

    # Getting the number of cans detected
    @render.text()
    def can():
        input.detect()
        with reactive.isolate():
            return np.count_nonzero(np.array(classes()) == 'can') ### when there is not detection it through this warning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
    
    # Getting the number of glass bottles detected
    @render.text()
    def glass_bottle():
        input.detect()
        with reactive.isolate():
            return np.count_nonzero(np.array(classes()) == 'glass bottle') ### when there is not detection it through this warning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
    
    # Getting the number of plastic bottles detected
    @render.text()
    def plastic_bottle():
        input.detect()
        with reactive.isolate():
            return np.count_nonzero(np.array(classes()) == 'plastic bottle') ### when there is not detection it through this warning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
    
    # Getting the latitude of place where the image was taken
    @render.text
    def lat():
        input.detect()
        with reactive.isolate():
            latitude = get_gps_info(image_path())[0]
            return latitude
    
    # Getting the longitude of place where the image was taken
    @render.text
    def lon():
        input.detect()
        with reactive.isolate():
            longitude = get_gps_info(image_path())[1]
            return longitude
    
    # Data Frame requests ---------------------------------------------------------

    # Creating a reactive data frame that will be used to show the table "requests" where we are saving all the information pulled from the image
    df = reactive.Value(pd.DataFrame({'request_ID': [],
                                   'cans': [],
                                   'glass_bottles': [],
                                   'plastic_bottles': [],
                                   'latitude': [],
                                   'longitude': [],
                                   
                                   'city': [],
                                   'province': [],
                                   'country': [],
                                   'path': [],
                                   'date_image': [],
                                   'postcode': [],
                                   }))
    
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
    df.set(pd.DataFrame(record, columns=["request_id", "n_cans", "n_glassbottles", "n_plasticbottles", "latitude", "longitude", "city", "province", "country", "image_path", "date_image", "postcode"]))
    cursor.close()
    connection.close()

    # Here we are inserting all the information from the image into the PostgreSQL table Request (localization, date, predictions), using the button submit to insert the new information detected in the image
    @reactive.effect
    @reactive.event(input.submit)
    def add_value_to_list():
        if get_gps_info(image_path())[0] != None and get_gps_info(image_path())[1] != None and len(classes()) > 0:
            # Perform reverse geocoding to get the name of city, province and country of the lat and long point
            location = geolocator.reverse(f"{get_gps_info(image_path())[0]}, {get_gps_info(image_path())[1]}", exactly_one=True)
            
            # Extract city, province, and country
            if location:
                address = location.raw['address']
                postcode = address.get('postcode', '')
                city = address.get('city', '')
                province = address.get('state', '')
                country = address.get('country', '')
            else:
                postcode = city = province = country = 'Unknown'
            
           
            # PostgreSQL connexion
            connection = psycopg2.connect(user="postgres",
                              password="sqladmin",
                              host="127.0.0.1",
                              port="5432",
                              database="RECYCLING_DB")

            # This is neccesary to perform operations inside the database
            cursor = connection.cursor()
            
            
            # # Selecting all the information from the request table
            # cursor.execute("SELECT * FROM requests")
            # record = cursor.fetchall()
            # df.set(pd.DataFrame(record, columns=["request_id", "n_cans", "n_glassbottles", "n_plasticbottles", "latitude", "longitude", "city", "province", "country", "image_path", "date_image", "postcode"]))
            request_id = df().shape[0] + 1


            # 2. Inserting values by predefining the values in a variable
            # We used a parameterized query to use Python variables as parameter values at execution time. Using a parameterized query, we can pass python variables as a query parameter using placeholders (%s).
            insert_query = """ INSERT INTO requests (request_id, n_cans, n_glassbottles, n_plasticbottles, latitude, longitude, city, province, country, image_path, date_image, postcode) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) """
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
                                get_gps_info(image_path())[2],
                                postcode,
                                )
            cursor.execute(insert_query, record_to_insert)
            connection.commit()

            # Selecting all the information from the request table
            cursor.execute("SELECT * FROM requests")
            record = cursor.fetchall()
            df.set(pd.DataFrame(record, columns=["request_id", "n_cans", "n_glassbottles", "n_plasticbottles", "latitude", "longitude", "city", "province", "country", "image_path", "date_image", "postcode"]))

            #print(record_to_insert) # here we can check if we are getting the information
            #print("1 Record inserted succesfully")
            cursor.close()
            connection.close()
        
        # Through an error in case that there is not information about the localization of the image (no lat, no long)    
        elif get_gps_info(image_path())[0] == None and get_gps_info(image_path())[1] == None:
            ui.notification_show("Sorry but as the image does not contain the localization information it can not be submitted as a request for collection", type='message')
            #await sleep(1)
            ui.notification_show("Warning message", type="warning")
        elif len(classes()) == 0:
            ui.notification_show("Sorry but looks like the model did not detected any of the target classes (cans, glass bottle, plastic bottle)")
            #await sleep(1)
            ui.notification_show("Warning message", type="warning")
        
    # Map Tab ---------------------------------------------------------------------

    # Defining the different type of icons
    coin_icon = Icon(icon_url = 'https://raw.githubusercontent.com/arol9204/arol9204/main/images/1.cent.png', icon_size=[15, 15])
    leaf_icon = Icon(icon_url='https://leafletjs.com/examples/custom-icons/leaf-green.png', icon_size=[38, 95], icon_anchor=[22,94])
    can_icon = Icon(icon_url='https://raw.githubusercontent.com/arol9204/arol9204/main/images/4.1can.png', icon_size=[20, 20])
    glass_bottle_icon = Icon(icon_url='https://raw.githubusercontent.com/arol9204/arol9204/main/images/4.2red-wine.png', icon_size=[20, 20])
    plastic_bottle_icon = Icon(icon_url='https://raw.githubusercontent.com/arol9204/arol9204/main/images/4.3water-bottle.png', icon_size=[25, 25])
    canglassplastic_mix_icon = Icon(icon_url='https://raw.githubusercontent.com/arol9204/arol9204/main/images/4.4canglassplastic_mix.png', icon_size=[20, 20])

    @reactive.Calc
    def mapping():
        basemap = basemaps[input.basemap()]
        map = L.Map(basemap = basemap, center=(0, 0), zoom = 2, scroll_wheel_zoom=True)
        map.layout.height = "700px"
        # Beer stores markers
        beer_store_mark1 = L.Marker(location=[42.31263551985872, -83.03326561020128], icon=leaf_icon, draggable=False)
        beer_store_mark2 = L.Marker(location=[42.30366417918876, -83.05465990194318], icon=leaf_icon, draggable=False)
        map.add_layer(beer_store_mark1)
        map.add_layer(beer_store_mark2)

        markers = []
        for _, row in df().iterrows():

            # Getting the latitude and longitude from the image
            if row['n_cans'] > 0 and row['n_glassbottles'] == 0 and row['n_plasticbottles'] == 0:
                request_marker = L.Marker(location=(row['latitude'], row['longitude']), icon=can_icon, draggable=False)
            elif row['n_cans'] == 0 and row['n_glassbottles'] > 0 and row['n_plasticbottles'] == 0:
                request_marker = L.Marker(location=(row['latitude'], row['longitude']), icon=glass_bottle_icon, draggable=False)
            elif row['n_cans'] == 0 and row['n_glassbottles'] == 0 and row['n_plasticbottles'] > 0:
                request_marker = L.Marker(location=(row['latitude'], row['longitude']), icon=plastic_bottle_icon, draggable=False)
            else:
                request_marker = L.Marker(location=(row['latitude'], row['longitude']), icon=canglassplastic_mix_icon, draggable=False)

            # Applying cluster
            markers.append(request_marker)
            marker_cluster = MarkerCluster(
                                markers = markers
                            )
        map.add_layer(marker_cluster)

            # # Without cluster
            # map.add_layer(request_marker)

        @reactive.effect()
        @reactive.event(input.submit)
        def _():
            if get_gps_info(image_path())[0] != None and get_gps_info(image_path())[1] != None and len(classes()) > 0:
                map.center = [get_gps_info(image_path())[0], get_gps_info(image_path())[1]]
                map.zoom = 15
                # Here is the lat and lon of the image that will be upload
                marker = L.Marker(location=[get_gps_info(image_path())[0], get_gps_info(image_path())[1]], draggable=True)
                # Adding the last request    
                map.add_layer(marker)

        return map

    @output 
    @render_widget
    def map():
        return mapping()

    # # Here we are showing the database in the UI app
    # @output
    # @render.table
    # def requests():
    #    #df = df.append(new_request(), ignore_index=True)
    #    return df()
    
    # Dashboard tab -----------------------------------------------------------------------------------

    @reactive.Calc
    def dashboard_df():
        df()

        # Convert the 'date_image' column to a datetime object
        df()['date_image'] = pd.to_datetime(df()['date_image'], format='%Y:%m:%d %H:%M:%S')

        df_filtered = df()

        # Saving the data range values 
        start_date = pd.Timestamp(input.daterange()[0])
        end_date = pd.Timestamp(input.daterange()[1])

        # Saving in a list the check box value
        box_values = list(input.checkbox_group())

        # Initialize a mask for the selection
        mask = pd.Series(False, index=df_filtered.index)

        # Apply filters for the selected objects
        if "c" in box_values:
            mask = mask | (df_filtered['n_cans'] > 0)
        if "gb" in box_values:
            mask = mask | (df_filtered['n_glassbottles'] > 0)
        if "pb" in box_values:
            mask = mask | (df_filtered['n_plasticbottles'] > 0)

        # Filter the DataFrame using the combined mask
        df_filtered = df_filtered[mask]

        # Filtering the dataframe by the date range selected in the filters area
        df_filtered = df_filtered[(df()['date_image'] >= start_date) & (df_filtered['date_image'] <= end_date)]

        # Extract the date component
        df_filtered['date_image'] = df_filtered['date_image'].dt.strftime('%Y-%m-%d')
        return df_filtered
    
    @reactive.Calc
    @render_widget
    def requests_by_date():
        # Group by date and count the number of requests for each date
        df_grouped = dashboard_df().groupby('date_image').size().reset_index(name='num_requests')

        if input.radio_fig1() == '1':
            # Create a barplot  figure
            fig = px.bar(df_grouped, x='date_image', y='num_requests',
                        labels={'date_image': 'Date', 'num_requests': 'Number of Requests'},
                        text_auto=True,) #title='Recycling Requests by Date',
        else:
            # Create a lineplot figure
            fig = px.line(df_grouped, x='date_image', y='num_requests', text='num_requests',
                        labels={'date_image': 'Date', 'num_requests': 'Number of Requests'},
                        )
            # Customize the layout (optional)
            fig.update_traces(textposition="top center")
        
        # Customize the layout (optional)
        fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Number of Requests',
                xaxis_tickangle=-45,  # Rotate x-axis labels for better readability
                margin=dict(l=20, r=20, t=40, b=10),
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor="rgba(0, 0, 0, 0.1)", # "rgba(0, 255, 128, 0.1)",
            )
        # Customize aspect
        fig.update_traces(marker_color='rgb(158,225,205)', marker_line_color='rgb(8,107,48)', # marker_color='rgb(158,225,205)', marker_line_color='rgb(8,107,48)',
                        marker_line_width=1.5, opacity=0.6)
        fig.layout.height = 400
        
        return fig
    
    # # Fig 2 -------------
    #@reactive.calc
    #@render_widget
    #@output(id="chart2")
    @reactive.calc
    @render_pydeck
    def chart2():
        if input.radio_fig2() == '1':
            # Heatmap Chart --------------------------------------------------------------------------
            # "open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain", "stamen-toner" or "stamen-watercolor"
            fig = px.density_mapbox(                                                  # z = n_cans, n_glassbottles, n_plasticbottles
                                    dashboard_df(), lat='latitude', lon='longitude', z='request_id', radius=5, 
                                    center=dict(lat=42.3126, lon=-83.0332), zoom=10, title="",
                                    mapbox_style="carto-darkmatter"
                                    )
            fig.update(layout_coloraxis_showscale=False)
            fig.update_layout(
                            margin=dict(l=20, r=20, t=20, b=20),
                            paper_bgcolor="rgba(0, 0, 0, 0.1)", # "rgba(0, 255, 128, 0.1)",
                            )
        else:
            # High-scale Map ----------------------------------------------------------------------------        
            # Get the filtered data from your DataFrame
            data = dashboard_df()[['longitude', 'latitude']]

            #print(type(data))
            # Define the HeatmapLayer
            layer = pdk.Layer(
                'HexagonLayer', # "HeatmapLayer",# 'ColumnLayer', # "HexagonLayer",
                data=data,
                get_position=['longitude', 'latitude'],
                auto_highlight=True,
                radius=100, # size for grouping elements
                elevation_scale=10,
                pickable=True,
                elevation_range=[0, 100],
                get_fill_color=[10, 50, 75, 100],
                extruded=True,
                coverage=1,
            )
            tooltip = {
                #"html": "<b>{mrt_distance}</b> meters away from an MRT station, costs <b>{price_per_unit_area}</b> NTD/sqm",
                "style": {"background": "grey", "color": "white", "font-family": '"Helvetica Neue", Arial', "z-index": "10000"},
            }

            # Set the initial view of the map
            # UK Center longitude=-1.415, latitude=52.2323,
            # Windsor Center = latitude=42.3126, longitude=-83.0332,

            view_state = pdk.ViewState(
                longitude=-83.0332, 
                latitude=42.3126,
                zoom=10,
                min_zoom=5,
                max_zoom=15,
                pitch=40.5,
                bearing=-5.36,
            )

            fig = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip,)
            #fig.to_html('hexagon-example.html')

        return fig



    # KPI Values ---------------
    @reactive.Calc
    def counting_cans():
        total = dashboard_df()['n_cans'].sum()
        return total
    
    @reactive.Calc
    def counting_glassbottles():
        total = dashboard_df()['n_glassbottles'].sum()
        return total
    
    @reactive.Calc
    def counting_plasticbottles():
        total = dashboard_df()['n_plasticbottles'].sum()
        return total
    
    @reactive.Calc
    def counting_requests():
        total = dashboard_df().shape[0]
        return total
    

    
    @render.text()
    def total_requests():
        return counting_requests()
    
    
    @render.text()
    def total_cans():
        return counting_cans()
    
    @render.text()
    def total_glassbottles():
        return counting_glassbottles()

    @render.text()
    def total_plasticbottles():
        return counting_plasticbottles()
    



app = App(app_ui, server)



#print("PostgreSQL connection is closed")


# When the image do not have lat and long it throw and error (DONE)
    # But now the map needs to be reactive, so if there is not lat nor long in the last image, it should still show the previous submissions
# When an object wasn't detected it is necesary to pop up a message saying that it happened and allow the user to select if it was an error or not, also i case that it was true that there is not a recycling object do not sumbit a request
# Make the map reactive, update it only if a new request with lat and long is submitted

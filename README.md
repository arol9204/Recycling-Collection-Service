# Recycling Collection Service
Capstone Project for Data Analytics for Business at St. Clair College

One of the main challenges that we face when we arrive in Canada as international students is finding a job, recycling is one of the ways that we can earn some extra money without uploading hundreds of  resumes or hacking a personal interview, and the best part is we are doing great work for our new city and our planet. 

Currently, the city of Windsor's Open Data Catalogue records nearly 5,000 service requests for uncollected recycling in 2022. Assuming an average of [5 glass bottles (10¢), and 15 cans (TBS/LCBO) (10¢)](https://www.thebeerstore.ca/about-us/environmental-leadership/bag-it-back-odrp/), a basic profit of $10,000 can be generated.

However, monetary gain is not the sole focus. The [Government of Canada](https://www.canada.ca/en/environment-climate-change/services/managing-reducing-waste/reduce-plastic-waste.html) emphasizes the importance of recycling: 

"Canadians throw away over 3 million tonnes of plastic waste every year. Only 9% is recycled while the rest ends up in our landfills"

![image](https://github.com/arol9204/Capstone/blob/bc1b9d2b3a971ca94f2298a3ca5419ad949225f5/Assets/Plastic_waste.png)

This project aims to enhance the circular life of recycled materials in Windsor by engaging St. Clair students in the process. 

A key project outcome entails the development of a mobile application that showcases current recycling collection requests in the area. By leveraging machine learning algorithms, we can optimize collection routes to maximize the number of recyclable materials collected, resulting in greater financial returns for participants and considering factors such as waste weight and type (e.g., glass bottles, cans, etc.). 

In essence, this initiative operates similarly to business models such as Uber or Lyft, but instead of connecting riders with drivers, we connect recyclable object collectors with multiple recycling service requests.

App UI

- Request tab:
  In this tab, people can upload a picture taken with some of the recycling objects that the model was trained to detect (tin cans, glass bottles, plastic bottles). The interface will show then the number of elements per class together with the latitude and longitude where the image was taken.
<img width="1919" alt="1  Request tab" src="https://github.com/arol9204/Recicling-Collection-Service/assets/63767771/2787098e-9881-4472-addd-993c5cf9d04c">

- Map:
  This tab shows all the requests collected on a map. I have been collecting images in Windsor for around 10 days, and so far I have taken more than 100 pictures with one or more of the targeted objects.
<img width="1918" alt="2  Map tab" src="https://github.com/arol9204/Recicling-Collection-Service/assets/63767771/eebdbda8-eaf6-4d72-bd79-c5fc9dcd6a22">

- Dashboard:
  This tab shows a dashboard about the key metrics for this project. It has the total number of requests, the total cans, glass bottles, and plastic bottles detected among all the requests. In addition, there is a bar chart with the number of requests per day since I started collecting images. Lastly, there are three heatmaps that visualize the magnitude of the presence of each of the classes in Windsor areas.
<img width="1918" alt="3  Dashboard tab" src="https://github.com/arol9204/Recicling-Collection-Service/assets/63767771/c7bbc8fd-47a5-47aa-b78a-ab472da44d7c">


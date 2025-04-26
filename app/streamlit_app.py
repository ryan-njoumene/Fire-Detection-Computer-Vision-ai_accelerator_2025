"""Import Streamlit Library"""
import streamlit as st

# ---------------------------------
# ---------------------------------

# PACKAGES TO DOWNLOAD

# pip install streamlit
# confirm that streamlit package has been correctly installed
# by using command : streamlit hello

# ---------------------------------
# ---------------------------------

# USEFUL COMMANDS

# To run your app
# streamlit run your_script.py [-- script args]

# ---------------------------------
# ---------------------------------

import sys
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from io import StringIO
import torch

st.title("Fire Detection Model")

st.markdown(
    """
    **A Computer Vision AI using [YOLO](https://www.ultralytics.com/) (You Only Look Once) Architecture**  
    *Made during AI-Accelerator 2025, hosted by [AI Launch Lab](https://launchlab.ai/) and [Dawson AI](https://www.dawsoncollege.qc.ca/ai/)*\n
    
    ***by [Ryan Njoumene](https://www.linkedin.com/in/ryan-njoumene-506a55345/), [Meriam Laabou](https://www.linkedin.com/in/meriam-ld-23b6962a4), [Edwin Kintu](https://www.linkedin.com/in/edwin-kintu-323926301/)***  

    ![alt text](https://www.dawsoncollege.qc.ca/ai/wp-content/uploads/sites/180/Website-Banner_AI-Accelerator-Programs-Website-800x450.jpg)

    ## Problem Definition
    Every year, with the effect of the Climate Change that keep on ramping up,
    many casualities caused by Natural Disasters have been reported and they aren't there to stop.  
    They are only getting worse.

    ### Old Objective
    Originally, through this project and the training of our model, we had for goal to build a detection tool
    that will help First Respondent and Disaster Reliefs Organization to find, retrieve more effectively and with less delay 
    people in distress and in the worse case their corpses in hazarduous natural environment caused by Natural and Human made Disasters.
    Because in those situations, every second can be a life saved.

    ![alt text](https://cdn.redcross.ca/prodmedia/crc/azure/volunteer/220118vr-vr-portal-emergencydisasters-banner.jpg?ext=.jpg)

    ### Revised Objective (Scoped Down)
    After a more through consideration, we decided to scope down our project to something simpler and less computationally
    intensive task to respect our time constraint and the feasibility of the project while taking in account our novice experience
    in the field of Machine Learning (Supervised) and Computer Vision.\n
    
    As such, here is our new objective for this model:\n

    This current year, after having observed a resurgence of Wildfire brought to light be the mediatisation of the 2025 Los Angeles's Wildfire
    in the U.S.A, we where put face to face with today's reality. Wildfires aren't a thing that only happens in developing country or
    in far away regions. They knock at our doorsteps and threaten to burn our homes.

    ![alt text](https://www.8newsnow.com/wp-content/uploads/sites/59/2025/01/HRecovery-1920x1080-copy.jpg?w=900)

    Thus, we felt that building a AI tool specialized in detecting the occurence of a fire (including smoke) could shine in our current
    time by detecting any sign of fire and alerting the related Fire Departement Authorities. This kind of tool could allow them to use drones
    surveillance to cover large forests area in high risk of fire hazard, while reacting faster in the case of a detected fire and limit it's spreads,
    or in the worst scenario execute an evacuation order to the population.

    ## Scope and Constraints

    As you have seen earlier, we decided to scope down our project objective to only detecting Fire Disasters because we lacked the ressources
    required for such a large scale project. One that will require way more computing power by training on a larger and more diverse dataset
    to detect human trapped in a Natural Disaster Environment. Doing so, also accounted for our lack of expertise in the field of Machine
    Learning and Computer Vision.

    Thus, we decided to focus on simply detecting a specific disaster situation (fire) to allow a quick disaster response and maximise
    the amount of people saved.\
    As for the constraints, we are aiming to build a AI Model that can account for many climate variables (northern vs southern hemisphere,
    the season, time of day, etc. ) and environment type (forests, urbans) by training it on correspoding image


    ## Success Criteria

    One of the Main Criteria we are aiming for in the effectiveness of our model is a high enough precision rate about the confidence
    in which the AI detect can fire. To be more clear, we believe that anything less than 0.75 could potentially cause harm by giving false
    information to authorities and overloading them with false request.\n

    However, we believe that the recall (number of time the AI think it detected a sign of fire) should take priority, because the
    best way to stop a fire is to contain it when it is small and inconspicuous. Thus having a wider net, even if prone to more errors,
    should detects more potential fires. It is also good to remember that the feed image/video (taken by a drone camera) processed by
    the model can also be analysed by human to confirm its truthfulness before sending fire fighters.

    ## Data Sources

    ***For Fire Detection***  
    *[AIDER: Aerial Image Database for Emergency Response applications (GitHub)](https://github.com/ckyrkou/AIDER/tree/master?tab=readme-ov-file)*  
    [original article](https://zenodo.org/records/3888300#.XvCPQUUzaUk)

    ***For Humans Detection in Disaster Situation***  
    *[C2A Dataset: Human Detection in Disaster Scenarios](https://www.kaggle.com/datasets/rgbnihal/c2a-dataset/data)*
    
    """
)


# ---------------------------------
# ---------------------------------

st.markdown("## Demo")

source = "fire_image0111.jpg"
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if 'is_upload' not in st.session_state:
    st.session_state['is_upload'] = False

if uploaded_file:
    image = Image.open(uploaded_file)
    if st.session_state['is_upload'] == False:
        st.session_state['detection_result'] = image
        st.session_state['is_upload'] = True
else:
    image = Image.open(source)
    if st.session_state['is_upload'] == True:
        st.session_state['detection_result'] = image
        st.session_state['is_new_upload'] = False


if 'detection_result' not in st.session_state:
    st.session_state['detection_result'] = image


tab1, tab2 = st.tabs(["Detecting Fire", "Original"])
tab1.image(st.session_state['detection_result'], use_container_width=True)
tab2.image(image, use_container_width=True)


confidence_threshold = st.slider("Confidence Threshold",0.0, 1.0, 0.25, key="conf")  # 👈 this is a widget
st.write("conf = ", confidence_threshold)

# ---------------------------------
# ---------------------------------

visualize = None

if st.toggle("Activates visualization of model features during inference", True):
    visualize = True
else:
    visualize = False

# ---------------------------------
# ---------------------------------

augment = None
if st.toggle("Enables test-time augmentation (TTA)", True):
    augment = True
else:
    augment = False

# ---------------------------------
# ---------------------------------

agnostic_nms = None
if st.toggle("Enables class-agnostic Non-Maximum Suppression (NMS)", True):
    agnostic_nms = True
else:
    agnostic_nms = False

# --------------------------------- 
# ---------------------------------

verbose = None
if st.toggle("Controls whether to display detailed inference logs in the terminal", True):
    verbose = True
else:
    verbose = False


#  classes 

# ---------------------------------
# ---------------------------------
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
string_io = StringIO()
handler = logging.StreamHandler(string_io)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

@st.cache_resource
def  detectFire(source: str):
    from ultralytics import YOLO

    log.info("Logs: ")

    model = YOLO("best.pt")
    # Test the model
    results = model.predict(
                            source=source,
                            conf=confidence_threshold,
                            visualize=visualize,
                            augment=augment,
                            agnostic_nms=agnostic_nms,
                            verbose=verbose
                            )
    
    log.info(f"\n{results[0].path}\n{results[0].verbose()}\n{results[0].summary()}\n{results[0].speed} ")

    #return all predictions
    return results

# ---------------------------------
# ---------------------------------

def prediction():
    with st.container():


        prediction = detectFire(source)[0]
        st.session_state['detection_result'] = prediction.plot(pil=True, img=image, labels=True, boxes=True, conf=True)
        # prediction.save(filename="detection_result.jpg")
        

        # st.subheader("Function Output:")

# ---------------------------------
# ---------------------------------

st.button("predict!", on_click=prediction())

if visualize or verbose:
    st.code(f"""{string_io.getvalue()}""")


st.markdown(
    """
    ## Review Given by one of the Event Manager of the AI Accelerator Program
    >*A very real problem as we have seen! I really like that you provide insights about your data and provide a concise presentation of your results. 
    An interesting approach to the vision dataset. Adding future considerations goes well with the potential of the approach you provided. 
    Your demo very nicely provides a full overview of the problem. Well done! Please share a link if possible in the chat. 
    Great work overall! Thank you! --[Aditi Maheshwari](https://www.linkedin.com/in/aditi-maheshwari)*
    """
)
import db
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
from bson.objectid import ObjectId


# Constants
VERBOSE = True
class_names = ['Fire', 'Smoke']
custom_weights = """weights/yolov8x-frozen/best.pt"""
uploaded_imgs_folder_path = "data/images/uploaded"
plotted_imgs_folder_path = "data/images/plotted"


# Database connection
client = db.mongo_conn(VERBOSE)
bdd = client.incendies
collection_images = bdd.images

# upload image, testing all different formats
img_test_01_path = "data/images/test/test-fire-01.jpg"
img_test_02_path = "data/images/test/test-fire-02.jpg"
img_test_03_url = 'https://puroclean.ca/wp-content/uploads/2022/11/AdobeStock_20903474.jpeg'
img_test_04_list = [img_test_01_path, img_test_02_path]

# CV2
# def show_image(title, img):
#   cv2_img = cv2.imread(img)
#   cv2.imshow(title, cv2_img)
#   cv2.waitKey(0) # waits until a key is pressed
#   cv2.destroyAllWindows() # destroys the window showing image

# PIL
# img2 = Image.open(img_test_01)
# img2.show()


# inference on file path, list of path, url, ndarray, CV2, PIL
# @st.cache_data
def img_inference(_imgs_input):
  """Function that :
  - makes a detection on the image
  - save original image and plotted image in a folder
  - returns a list of dictionnaries. One dictionnary of each image."""
  # Detection / Inference
  results = model(_imgs_input) # list of Results Object
  # Building a list of dictionnaries for the Results
  results_list_of_dict = []
  # List of plotted images
  imgs_plotted = []

  for result in results:
    # Each result is a dictionnary that is saved as document in MongoDB
    result_dict = {}
    # Original image
    img_uploaded = result.orig_img # ndarray
    # Plotted image
    img_plotted = result.plot() # ndarray
    imgs_plotted.append(img_plotted)
    # image name with the current date-time
    timestr = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')
    img_name = f"{timestr}.png"
    # Building the dictionnary
    # result_dict['filename'] = # file name of the uploaded file
    result_dict['original_name'] = img_name # named with the current date-time.
    result_dict['original_path'] = f"{uploaded_imgs_folder_path}/{img_name}"
    result_dict['original_width'] = result.orig_shape[1]
    result_dict['original_height'] = result.orig_shape[0]
    result_dict['plotted_name'] = img_name
    result_dict['plotted_path'] = f"{plotted_imgs_folder_path}/{img_name}"
    result_dict['detection_time'] = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')
    result_dict['speed'] = result.speed # dict
    # Class Names
    names_dict = result.names
    keys_values = names_dict.items()
    result_dict['names'] = {str(key): value for key, value in keys_values}
    # BOXES
    # Convert torch tensors to ndarrays
    result_cpu = result.cpu()
    result_ndarray = result_cpu.numpy()
    result_dict['boxes'] = {}
    for idx, box in enumerate(result_ndarray.boxes.data):
    # for box in result_ndarray.boxes.data:
      box_dict = {}
      box_dict["x1"] = int(box[0])
      box_dict["y1"] = int(box[1])
      box_dict["x2"] = int(box[2])
      box_dict["y2"] = int(box[3])
      box_dict["conf"] = round(box[4].item(), 4)
      box_dict["classname"] = int(box[5])
      result_dict['boxes'][f"{idx}"] = box_dict

    # Append the dictionnary to a list of results
    results_list_of_dict.append(result_dict)

    # Saving images
    # Original image
    cv2.imwrite(result_dict['original_path'], img_uploaded)
    # Plotted image
    cv2.imwrite(result_dict['plotted_path'], img_plotted)
    if VERBOSE:
      print(f"Image {result_dict['original_name']} saved at : {result_dict['original_path']}")
      print(f"Image {result_dict['plotted_name']} saved at : {result_dict['plotted_path']}")

  # return a list of dictionnaries
  return imgs_plotted, results_list_of_dict


# Count
# count_images_documents = collection_images.count_documents({})
# print("Images documents count : ", count_images_documents)

# save images in DB
# imgs_inserted = collection_images.insert_many(imgs_results)
# imgs_inserted_inserted_ids = imgs_inserted.inserted_ids



# Streamlit
# Working with file uploads in Streamlit :
# https://blog.jcharistech.com/2020/11/08/working-with-file-uploads-in-streamlit-python/

# IDEA
# Past detection
# Caroussel of plotted images
# or GRID

st.set_page_config(
  page_title="Fire detection 🧯🔥",
  page_icon="🧯🔥",
  layout='wide',
  menu_items={
    'Get Help': 'https://www.extremelycoolapp.com/help',
    'Report a bug': "https://www.extremelycoolapp.com/bug",
    'About': "# This is a header. This is an *extremely* cool app!"
  }
)

# Loading the model (cached)
@st.cache_resource
def load_model(custom_weights=custom_weights):
  model = YOLO(custom_weights)
  return model
model = load_model()

# About caching in Streamlit:
# https://docs.streamlit.io/library/advanced-features/caching
@st.cache_data 
def load_image(image_file):
	img = Image.open(image_file) # PIL Object
	return img 

def main():
  # Page Title
  st.title("Fire detection 🧯🔥")
  # Side menu
  menu = ["Image","Video","Webcam"]
  choice = st.sidebar.selectbox("Menu",menu)

  if choice == "Image":
    st.subheader("Image")
    image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
    if image_file is not None:
      # prediction / inference + saving images files
      with st.spinner('Wait for it...'):
        img = load_image(image_file)
        imgs_plotted, results_list_of_dict = img_inference(img)
      st.success('Done!')

      # Detection results
      st.header("Detection results")

      # Results paragraph
      # st.write("X fires and Y smoke in this image.")

      # Columns for detection
      col1, col2 = st.columns(2)

      # To See Details
      # if VERBOSE:
      #   st.write(type(image_file))
      #   st.write(dir(image_file))
      #   file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
      #   st.write(file_details)

      # Column 1
      # Show original image
      col1.image(img, width=420)

      # Column 2
      # Show plotted image
      col2.image(imgs_plotted[0], width=420, channels='BGR')

      # Display results as JSON Object
      if VERBOSE:
        st.markdown('### Document saved in MongoDB "Images" Collection:')
        st.json(results_list_of_dict[0])

      # Save results to DB
      db.save_document(collection_images, results_list_of_dict[0])

    st.divider()


    # Show past detections
    st.header("Past detections")
    # Get past detections from DB, sortted by = "detection_time" DESC
    imgs_documents = db.get_all_documents(collection_images).sort("detection_time", -1)
    # Display images and informations
    for img_document in imgs_documents:
      # st.json(img_document)

      # How to check if key exists in a python dictionary?
      # https://flexiple.com/python/check-if-key-exists-in-dictionary-python/#section1
      # if img_document.get('plotted_path') is not None:
      if 'plotted_path' in img_document :
        # Display image
        st.image(img_document['plotted_path'], width=520)

        # Display filename
        # st.write(f"Original filename: xxxxxxx.jpg")

        # Display detection date/time 

        # The strptime() method creates a datetime object from the given string.
        # https://www.programiz.com/python-programming/datetime/strptime

        # What do T and Z mean in timestamp format ?
        # The T doesn't really stand for anything. It is just the separator that the ISO 8601 combined date-time format requires. You can read it as an abbreviation for Time. 

        date_obj = datetime.strptime(img_document["detection_time"], "%Y-%m-%dT%H:%M:%S.%f")
        st.write(f"""Detection time: {date_obj.strftime("%Y-%m-%d at %H:%M:%S")}""")


  elif choice == "Video":
    st.subheader("Video")
    video_file = st.file_uploader("Upload Video",type=['.mpeg', '.mp4', '.avi'])
    if st.button("Process"):
      if video_file is not None:
        file_details = {"Filename":video_file.name,"FileType":video_file.type,"FileSize":video_file.size}
        st.write(file_details)

        # df = pd.read_csv(video_file)
        # st.dataframe(df)

  elif choice == "Webcam":
    st.write('Webcam')

  else:
    st.subheader("About")
    st.info("Built with Streamlit")
    st.info("Instagram: @davidscanu_")
    st.text("David Scanu")

if __name__ == '__main__':
	main()

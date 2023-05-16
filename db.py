from pymongo import MongoClient
from pprint import pprint
from bson.objectid import ObjectId

# Connection to MongoDB
# @st.cache_resource
def mongo_conn(VERBOSE):
  try:
    client = MongoClient("mongodb://root:example@localhost:27017/")
    if VERBOSE:
      print("Connection Successful")
    # return client.grid_file
    return client
  except Exception as e:
    print("error", e)

# get by "_id" : {"_id": ObjectId(str(_id))}
def get_document(collection, filters={}):
    document = collection.find_one(filters)
    return document

# get all images
def get_all_documents(collection, filters={}):
    documents = collection.find(filters) # returns a Cursor instance, use for loop
    return documents

# save in db.collection
def save_document(collection, data, VERBOSE=True):
    inserted_id = collection.insert_one(data).inserted_id # returns "_id" object.
    if VERBOSE: 
       pprint(data)
       print(f"Document saved in DB.")
    return inserted_id

def delete_all_documents(collection, VERBOSE=True):
  x = collection.delete_many({})
  print(x.deleted_count, " documents deleted.")
   
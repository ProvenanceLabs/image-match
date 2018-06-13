import urllib.request
import os
from elasticsearch import Elasticsearch
import time
from numpy import random

from image_match.elasticsearch_driver import SignatureES as SignatureES_fields
from image_match.elasticsearchflat_driver import SignatureES as SignatureES_flat
from image_match.elasticsearchflatint_driver import SignatureES as SignatureES_flatint

# To run this test, have an elasticsearch on ports 9200 and 9300
# docker run -d -p 9200:9200 -p 9300:9300 elasticsearch:5.5.2

# Params
delete_indices = True
populate_indices = True
max_msm = 6
range_msm = range(1, max_msm + 1)

test_img_url1 = 'https://camo.githubusercontent.com/810bdde0a88bc3f8ce70c5d85d8537c37f707abe/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f7468756d622f652f65632f4d6f6e615f4c6973612c5f62795f4c656f6e6172646f5f64615f56696e63692c5f66726f6d5f4332524d465f7265746f75636865642e6a70672f36383770782d4d6f6e615f4c6973612c5f62795f4c656f6e6172646f5f64615f56696e63692c5f66726f6d5f4332524d465f7265746f75636865642e6a7067'
test_img_url2 = 'https://camo.githubusercontent.com/826e23bc3eca041110a5af467671b012606aa406/68747470733a2f2f63322e737461746963666c69636b722e636f6d2f382f373135382f363831343434343939315f303864383264653537655f7a2e6a7067'
urllib.request.urlretrieve(test_img_url1, 'test1.jpg')
urllib.request.urlretrieve(test_img_url2, 'test2.jpg')

# ES for fields
INDEX_NAME_FIELDS = 'test_environment_fields'
DOC_TYPE_FIELDS = 'image'
MAPPINGS_FIELDS = {
  "mappings": {
    DOC_TYPE_FIELDS: {
      "dynamic": True,
      "properties": {
        "metadata": {
            "type": "nested",
            "dynamic": True,
            "properties": {
                "tenant_id": { "type": "keyword" },
                "project_id": { "type": "keyword" }
            }
        }
      }
    }
  }
}

# ES for flat
INDEX_NAME_FLAT = 'test_environment_flat'
DOC_TYPE_FLAT = 'image_flat'
MAPPINGS_FLAT = {
  "mappings": {
    DOC_TYPE_FLAT: {
      "dynamic": True,
      "properties": {
        "metadata": {
            "type": "nested",
            "dynamic": True,
            "properties": {
                "tenant_id": { "type": "keyword" },
                "project_id": { "type": "keyword" }
            }
        }
      }
    }
  }
}

# ES for flatint
INDEX_NAME_FLATINT = 'test_environment_flatint'
DOC_TYPE_FLATINT = 'image_flatint'
MAPPINGS_FLATINT = {
  "mappings": {
    DOC_TYPE_FLATINT: {
      "dynamic": True,
      "properties": {
        "metadata": {
            "type": "nested",
            "dynamic": True,
            "properties": {
                "tenant_id": { "type": "keyword" },
                "project_id": { "type": "keyword" }
            }
        },
        "simple_words": {
          "type": "long",
          "doc_values": False,
          "store": False
        }
      }
    }
  }
}

es = Elasticsearch()

if delete_indices:
    print("Delete indices")
    es.indices.delete(INDEX_NAME_FIELDS)
    es.indices.delete(INDEX_NAME_FLAT)
    es.indices.delete(INDEX_NAME_FLATINT)

    print("Create indices")
    es.indices.create(index=INDEX_NAME_FIELDS, body=MAPPINGS_FIELDS)
    es.indices.create(index=INDEX_NAME_FLAT, body=MAPPINGS_FLAT)
    es.indices.create(index=INDEX_NAME_FLATINT, body=MAPPINGS_FLATINT)

# Define three ses
print("Created index {} for fields documents".format(INDEX_NAME_FIELDS))
print("Created index {} for flat documents".format(INDEX_NAME_FLAT))
print("Created index {} for flatint documents".format(INDEX_NAME_FLATINT))
ses_fields = SignatureES_fields(es=es, index=INDEX_NAME_FIELDS, doc_type=DOC_TYPE_FIELDS)
ses_flat = SignatureES_flat(es=es, index=INDEX_NAME_FLAT, doc_type=DOC_TYPE_FLAT)
ses_flatint = SignatureES_flatint(es=es, index=INDEX_NAME_FLATINT, doc_type=DOC_TYPE_FLATINT)

# Download dataset
print("Download dataset if does not exist")
dataset_url = "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"
dir_path = os.path.dirname(os.path.realpath(__file__))
local_file = os.path.join(dir_path, "101_ObjectCategories.tar.gz")
local_directory = os.path.join(dir_path, "101_ObjectCategories")
if not os.path.exists("101_ObjectCategories"):
    cmd = "wget {} -O {}".format(dataset_url, local_file)
    print(cmd)
    os.system(cmd)

    cmd = "tar xzf {}".format(local_file)
    print(cmd)
    os.system(cmd)

# Populate the three  indexes with images
print("Ingest documents")
all_files = []
total_time_ingest_fields = 0
total_time_ingest_flat = 0
total_time_ingest_flatint = 0
for root, dirs, files in os.walk(local_directory):
    for file in files:
        full_path = os.path.join(root, file)
        all_files.append(full_path)

        if len(all_files) % 1000 == 0:
            print("{} documents ingested (in each index)".format(len(all_files)))

        if populate_indices:
            t_fields = time.time()
            ses_fields.add_image(full_path)
            total_time_ingest_fields += (time.time() - t_fields)

            t_flat = time.time()
            ses_flat.add_image(full_path)
            total_time_ingest_flat += (time.time() - t_flat)

            t_flatint = time.time()
            ses_flatint.add_image(full_path)
            total_time_ingest_flatint += (time.time() - t_flatint)

print("{} to ingest fields documents".format(total_time_ingest_fields))
print("{} to ingest flats documents".format(total_time_ingest_flat))
print("{} to ingest flatint documents".format(total_time_ingest_flatint))

# Pick 500 random files and request both indexes
total_time_search_fields = 0
total_time_search_flat = 0
total_time_search_flatint = 0
num_random = 500
total_res_flat = 0  # Total results cumulated between flat and fields
total_res_flatint = 0  # Total results cumulated between flatint and fields
in_common_flat = 0  # Number of results returned from both indices
in_common_flatint = 0  # Number of results returned from both indices

random_images = random.choice(all_files, num_random).tolist()

for msm in range_msm:
    ses_flat.minimum_should_match = msm
    ses_flatint.minimum_should_match = msm

    found_flat = [0, 0, 0]  # (less_results, equal_results, more_results)
    found_flatint = [0, 0, 0]  # (less_results, equal_results, more_results)
    same_first_flat = 0  # Number of time the first result is the same
    same_first_flatint = 0  # Number of time the first result is the same

    for image in random_images:
        t_search_fields = time.time()
        res_fields = ses_fields.search_image(image)
        total_time_search_fields += (time.time() - t_search_fields)

        t_search_flat = time.time()
        res_flat = ses_flat.search_image(image)
        total_time_search_flat += (time.time() - t_search_flat)

        t_search_flatint = time.time()
        res_flatint = ses_flatint.search_image(image)
        total_time_search_flatint += (time.time() - t_search_flatint)

        # stats for flat
        if len(res_fields) == len(res_flat):
            found_flat[1] += 1
        elif len(res_fields) > len(res_flat):
            found_flat[2] += 1
        else:
            found_flat[0] += 1

        total_res_flat += len(res_fields) + len(res_flat)
        in_common_flat += len(
            list(
                set([r["path"] for r in res_fields]).intersection(
                    [r["path"] for r in res_flat])
            )
        )

        if len(res_fields) > 0 and len(res_flat) > 0:
            if res_fields[0]["path"] == res_flat[0]["path"]:
                same_first_flat += 1

        # stats for flatint
        if len(res_fields) == len(res_flatint):
            found_flatint[1] += 1
        elif len(res_fields) > len(res_flatint):
            found_flatint[2] += 1
        else:
            found_flatint[0] += 1

        total_res_flatint += len(res_fields) + len(res_flatint)
        in_common_flatint += len(
            list(
                set([r["path"] for r in res_fields]).intersection(
                    [r["path"] for r in res_flatint])
            )
        )

        if len(res_fields) > 0 and len(res_flatint) > 0:
            if res_fields[0]["path"] == res_flatint[0]["path"]:
                same_first_flatint += 1

    print("")
    print("minimum_should_match = {}".format(msm))

    print("--flat--")
    print("{} less results in flat, {} same num results, {} more results in flat"
          .format(found_flat[0], found_flat[1], found_flat[2]))
    print("{} common results out of {} total results. {}% match"
          .format(in_common_flat, total_res_flat, in_common_flat * 100 / total_res_flat))
    print("{} same first results (out of {})".format(same_first_flat, num_random))

    print("--flatint--")
    print("{} less results in flatint, {} same num results, {} more results in flatint"
          .format(found_flatint[0], found_flatint[1], found_flatint[2]))
    print("{} common results out of {} total results. {}% match"
          .format(in_common_flatint, total_res_flatint, in_common_flatint * 100 / total_res_flatint))
    print("{} same first results (out of {})".format(same_first_flatint, num_random))

print("")
print("{} searches total".format(num_random * max_msm))
print("{} to search fields documents".format(total_time_search_fields))
print("{} to search flat documents".format(total_time_search_flat))
print("{} to search flatint documents".format(total_time_search_flatint))

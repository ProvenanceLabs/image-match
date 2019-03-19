import os
from elasticsearch import Elasticsearch
import time
from numpy import random
import numpy as np
from PIL import ImageFilter, Image
import matplotlib.pyplot as plt

from image_match.elasticsearch_driver \
    import SignatureES as SignatureES_fields
from image_match.elasticsearchflat_driver \
    import SignatureES as SignatureES_flat
from image_match.elasticsearchflatint_driver \
    import SignatureES as SignatureES_flatint

# To run this test, have an elasticsearch on ports 9200 and 9300
# docker run -d -p 9200:9200 -p 9300:9300 elasticsearch:5.5.2

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--delete-indices", default=False,
                    help="Will delete existing ES indices (test_environment_"
                         "fields, test_environment_int and test_environment_"
                         "flatint")
parser.add_argument("--populate-indices", default=False,
                    help="Ingest into indices all the images from dataset")
parser.add_argument("--max-msm", default=6,
                    help="Until which minimum should match (msm) value to run "
                         "this benchmark")
parser.add_argument("--num-random", default=500,
                    help="Total number of images to search for a given msm. "
                         "Total num searches = num_random * len(range_msm)")
args = parser.parse_args()


# Params
delete_indices = args.delete_indices
populate_indices = args.populate_indices
max_msm = args.max_msm
num_random = args.num_random
range_msm = range(1, max_msm + 1)


def noise_generator(noise_type, image):
    """
    Found on https://stackoverflow.com/questions/22937589/
    how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    Generate noise to a given Image based on required noise type

    Input parameters:
        image: ndarray (input image data. It will be converted to float)
        noise_type: string
            'gauss'        Gaussian-distribution based noise
            's&p'          Salt and Pepper noise, 0 or 1
    """
    if noise_type == "gauss":
        row, col, ch = image.shape
        mean = 0.5
        var = 0.01
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy.astype('uint8')
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.01
        out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, idx - 1, int(num_salt))
                  for idx in image.shape]
        out[coords] = 255
        # Generate Pepper '0' noise
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, idx - 1, int(num_pepper))
                  for idx in image.shape]
        out[coords] = 0
        return out
    else:
        return image


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
                "tenant_id": {"type": "keyword"},
                "project_id": {"type": "keyword"}
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
                "tenant_id": {"type": "keyword"},
                "project_id": {"type": "keyword"}
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
                "tenant_id": {"type": "keyword"},
                "project_id": {"type": "keyword"}
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

# The relatively small size of returned document (100, which is default)
# can lead to hard to inconsistent results (typically
# documents only found in flat but not in fields) because the correct document
# might not be in the top 100 for a fields search, but in the top 100 for a
# flat or flatint search.
ses_fields = SignatureES_fields(es=es, index=INDEX_NAME_FIELDS,
                                doc_type=DOC_TYPE_FIELDS, size=100)
ses_flat = SignatureES_flat(es=es, index=INDEX_NAME_FLAT,
                            doc_type=DOC_TYPE_FLAT, size=100)
ses_flatint = SignatureES_flatint(es=es, index=INDEX_NAME_FLATINT,
                                  doc_type=DOC_TYPE_FLATINT, size=100)

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
            print("{} documents ingested (in each index)"
                  .format(len(all_files)))

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

random_images = random.choice(all_files, num_random).tolist()

# Store all stats per msm {"1": {"same_first_flat": 489,
# "not_same_first_flat": [0, 0, 0], "same_first_flatint": 3,
# "not_same_first_flatint": [0, 0, 0]}}
stats_msm = {}

for msm in range_msm:
    ses_flat.minimum_should_match = msm
    ses_flatint.minimum_should_match = msm

    same_first_flat = 0  # Number of time the first result is the same
    not_same_first_flat = [0, 0, 0]  # both not found, found in fields, in flat
    same_first_flatint = 0  # Number of time the first result is the same
    not_same_first_flatint = [0, 0, 0]  # idem

    for image_path in random_images:
        original_image = Image.open(image_path)
        altered_path = "altered.jpg"
        # altered_image = original_image.filter(ImageFilter.BLUR)
        img_array_with_noise = noise_generator("s&p", np.array(original_image))
        altered_image = Image.fromarray(img_array_with_noise)
        altered_image.save(altered_path)
        image_path_to_search = altered_path

        t_search_fields = time.time()
        res_fields = ses_fields.search_image(image_path_to_search)
        total_time_search_fields += (time.time() - t_search_fields)

        t_search_flat = time.time()
        res_flat = ses_flat.search_image(image_path_to_search)
        total_time_search_flat += (time.time() - t_search_flat)

        t_search_flatint = time.time()
        res_flatint = ses_flatint.search_image(image_path_to_search)
        total_time_search_flatint += (time.time() - t_search_flatint)

        # FLAT analysis
        # Precision of first result
        same_first_flat_bool = False
        if len(res_fields) > 0 and len(res_flat) > 0:
            if res_fields[0]["path"] == res_flat[0]["path"]:
                same_first_flat_bool = True
        elif len(res_fields) == 0 and len(res_flat) == 0:
            same_first_flat_bool = True  # both fields and flat didn't find

        # When the first result is not the same, find out more details
        if same_first_flat_bool:
            same_first_flat += 1
        else:
            pathes_fields = [res["path"] for res in res_fields] + [""]
            pathes_flat = [res["path"] for res in res_flat] + [""]
            if image_path not in pathes_fields and image_path not in pathes_flat:
                not_same_first_flat[0] += 1
            elif image_path not in pathes_fields and pathes_flat[0] == image_path:
                not_same_first_flat[2] += 1
            elif image_path not in pathes_flat and pathes_fields[0] == image_path:
                not_same_first_flat[1] += 1

        # FLATINT analysis
        # Precision of first result
        same_first_flatint_bool = False
        if len(res_fields) > 0 and len(res_flatint) > 0:
            if res_fields[0]["path"] == res_flatint[0]["path"]:
                same_first_flatint_bool = True
        elif len(res_fields) == 0 and len(res_flatint) == 0:
            same_first_flatint_bool = True  # both fields and flatint didn't find

        # When the first result is not the same, find out more details
        if same_first_flatint_bool:
            same_first_flatint += 1
        else:
            pathes_fields = [res["path"] for res in res_fields] + [""]
            pathes_flatint = [res["path"] for res in res_flatint] + [""]
            if image_path not in pathes_fields and image_path not in pathes_flatint:
                not_same_first_flatint[0] += 1
            elif image_path not in pathes_fields and pathes_flatint[0] == image_path:
                not_same_first_flatint[2] += 1
            elif image_path not in pathes_flatint and pathes_fields[0] == image_path:
                not_same_first_flatint[1] += 1

        # Delete blurred image
        os.remove(altered_path)

    stats_msm[str(msm)] = {
        "same_first_flat": same_first_flat,
        "not_same_first_flat": not_same_first_flat,
        "same_first_flatint": same_first_flatint,
        "not_same_first_flatint": not_same_first_flatint
    }

    print("")
    print("minimum_should_match = {}".format(msm))

    print("--flat--")
    print("{} same first results (out of {})".format(same_first_flat, num_random))
    print("When not same first results ({} cases)".format(sum(not_same_first_flat)))
    print(". {} both wrong".format(not_same_first_flat[0]))
    print(". {} found in fields but not in flat".format(not_same_first_flat[1]))
    print(". {} found in flat but not in fields".format(not_same_first_flat[2]))

    print("--flatint--")
    print("{} same first results (out of {})".format(same_first_flatint, num_random))
    print("When not same first results ({} cases)".format(sum(not_same_first_flatint)))
    print(". {} both wrong".format(not_same_first_flatint[0]))
    print(". {} found in fields but not in flatint".format(not_same_first_flatint[1]))
    print(". {} found in flatint but not in fields".format(not_same_first_flatint[2]))


print(stats_msm)

print("")
num_searches = num_random * max_msm
print("{} searches total".format(num_searches))
print("{} to search fields documents".format(total_time_search_fields))
print("{} to search flat documents".format(total_time_search_flat))
print("{} to search flatint documents".format(total_time_search_flatint))

# -----------------------------------------------------------------------------
#
# Draw plots in png files for further analysis
#
# -----------------------------------------------------------------------------
# Typical stats_msm:
#   {'1': {'same_first_flat': 499, 'not_same_first_flat': [0, 1, 0],
#          'same_first_flatint': 499, 'not_same_first_flatint': [0, 0, 0]},
#    '2': {'same_first_flat': 497, 'not_same_first_flat': [2, 1, 0],
#          'same_first_flatint': 494, 'not_same_first_flatint': [4, 1, 0]},
#    '3': {'same_first_flat': 496, 'not_same_first_flat': [1, 2, 0],
#          'same_first_flatint': 495, 'not_same_first_flatint': [1, 2, 0]},
#    '4': {'same_first_flat': 494, 'not_same_first_flat': [2, 2, 0],
#          'same_first_flatint': 495, 'not_same_first_flatint': [2, 2, 0]},
#    '5': {'same_first_flat': 488, 'not_same_first_flat': [6, 4, 0],
#          'same_first_flatint': 489, 'not_same_first_flatint': [6, 4, 0]},
#    '6': {'same_first_flat': 490, 'not_same_first_flat': [4, 5, 0],
#          'same_first_flatint': 489, 'not_same_first_flatint': [4, 5, 0]}}

# Generate stat plot for ingestion
names = ["fields", "flat_txt", "flat_int"]
values_ingest = [total_time_ingest_fields, total_time_ingest_flat, total_time_ingest_flatint]
colors = ["red", "green", "blue"]
plt.xlabel("Ingestion Time (ms)")
plt.title("Average Ingestion Time ({} documents)".format(len(all_files)))
values_ingest_mean = [v / float(len(all_files)) for v in values_ingest]
plt.barh(names, values_ingest_mean, color=colors)
for i, v in enumerate(values_ingest_mean):
    plt.text(v - .01, i + .25, "{:.4f} ms".format(v), color='black')
plt_file_name = 'plot_time_ingestion.png'
plt.savefig(plt_file_name)
plt.clf()
print("Save plot {}".format(plt_file_name))

# Size on disk
size_fields = es.indices.stats(INDEX_NAME_FIELDS)["indices"][INDEX_NAME_FIELDS]["total"]["store"]["size_in_bytes"]
size_flat = es.indices.stats(INDEX_NAME_FLAT)["indices"][INDEX_NAME_FLAT]["total"]["store"]["size_in_bytes"]
size_flatint = es.indices.stats(INDEX_NAME_FLATINT)["indices"][INDEX_NAME_FLATINT]["total"]["store"]["size_in_bytes"]
sizes = [size_fields, size_flat, size_flatint]
plt.xlabel("Size on disk (MB)")
plt.title("Size of ES index on disk")
values = [s/1024/1024 for s in sizes]
plt.barh(names, values, color=colors)
for i, v in enumerate(values):
    plt.text(v - .01, i + .25, "{:.2f} MB".format(v), color='black')
plt_file_name = "plot_disk_usage.png"
plt.savefig(plt_file_name)
plt.clf()
print("Save plot {}".format(plt_file_name))

# Search Average Time
values_search = [total_time_search_fields, total_time_search_flat, total_time_search_flatint]
plt.xlabel("Search Time (ms)")
plt.title("Average Search Time ({} searches)".format(num_searches))
values_search_mean = [v / float(len(all_files)) for v in values_search]
plt.barh(names, values_search_mean, color=colors)
for i, v in enumerate(values_search_mean):
    plt.text(v - .01, i + .25, "{:.4f} ms".format(v), color='black')
plt_file_name = "plot_search_time.png"
plt.savefig(plt_file_name)
plt.clf()
print("Save plot {}".format(plt_file_name))

# Quantitative results


def draw_qualitative_plot(suffix):
    """
    Draw a plot in a file from the stats_msm.
    :param suffix: "flat" or "flatint"
    :return: None
    """
    names_msm = list(map(str, list(range_msm)))
    same_first = [stats_msm[str(i_msm)]["same_first_" + suffix] for i_msm in list(range_msm)]
    both_not_found = [stats_msm[str(i_msm)]["not_same_first_" + suffix][0] for i_msm in list(range_msm)]
    found_in_fields = [stats_msm[str(i_msm)]["not_same_first_" + suffix][1] for i_msm in list(range_msm)]
    found_in_flat = [stats_msm[str(i_msm)]["not_same_first_" + suffix][2] for i_msm in list(range_msm)]
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    bar1 = ax2.bar(names_msm, both_not_found)
    bar2 = ax2.bar(names_msm, found_in_fields, bottom=both_not_found)
    bar3 = ax2.bar(names_msm, found_in_flat, bottom=found_in_fields)
    maxval = max([sum([stats_msm[r]["not_same_first_" + suffix][idx_res] for r in names_msm])
                  for idx_res in range(0, 3)])
    ax2.set_ylim(0, int(maxval + maxval*.75))  # Expand y limit of max histogram to have some space

    plot1 = ax1.plot(names_msm, same_first, "r-o")
    minval = min([stats_msm[s]["same_first_" + suffix] for s in names_msm])
    ax1.set_ylim(int(500 - ((500 - minval) * 2.75)), 500)

    plt.xlabel("Minimum Should Match")
    plt.ylabel("Hits")
    plt.legend((plot1[0], bar1[0], bar2[0], bar3[0]), ('Same first result', 'Not found in both',
                                                       'Found only in fields', 'Found only in ' + suffix))
    plt_file_name = 'plot_qualitative_{}.png'.format(suffix)
    plt.savefig(plt_file_name)
    plt.clf()
    print("Save plot {}".format(plt_file_name))


draw_qualitative_plot("flat")
draw_qualitative_plot("flatint")

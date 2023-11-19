import os
import json
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np



app = Flask(__name__,static_folder='static', static_url_path='/static')

# Load images from JSON file
with open('Database.json', 'r') as file:
    images_data = json.load(file)
    images = images_data.get('images', [])
    
#Function to calculate histogram with quatified images    
def calculate_histogram(image, k):
    # Quantize the image using k-means clustering
    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    quantized_image = centers[labels.flatten()]
    quantized_image = quantized_image.reshape(image.shape)

    # Calculate histogram on the quantized image
    hist_channels = [cv2.calcHist([quantized_image[..., i]], [0], None, [64], [0, 256]) for i in range(3)]

    combined_histogram = np.concatenate(hist_channels, axis=0)

    return combined_histogram


#calculate texture descriptor 

def calculate_texture_descriptor(image):

  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  fourier_transform = np.fft.fft2(gray_image)

  amplitude_spectrum = np.abs(fourier_transform)

  blocks = np.array_split(amplitude_spectrum[:gray_image.shape[0]//2, :], 6, axis=1)

  subblocks = [np.array_split(block, 3, axis=1) for block in blocks]

  flattened_subblocks = [subblock for block in subblocks for subblock in block]

  log_energy_results = []
  for block in flattened_subblocks:
      energy = np.mean(np.square(np.log(1 + block)))
      log_energy_results.append(energy)

  return log_energy_results


#Distance between histograms
def euclidean_distance(hist1, hist2):

    return np.linalg.norm(hist1 - hist2)

#Distance between texture descriptors
def manhattan_distance(list1, list2):
    """
    Calcule la distance de Manhattan entre deux listes.
    """
    if len(list1) != len(list2):
        raise ValueError("Les listes doivent avoir la même longueur.")
    return sum(abs(a - b) for a, b in zip(list1, list2))
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    

@app.route('/')
def index():
    return render_template('index.html')




@app.route('/show_images', methods=['POST'])
def show_images():
    rows = int(request.form['rows'])
    columns = int(request.form['columns'])

    # Slice the images based on the selected rows and columns
    sliced_images = images[:rows * columns]

    return render_template('show_images.html', rows=rows, columns=columns, images=sliced_images)





@app.route('/show_top_similar_images', methods=['GET'])
def show_top_similar_images():
    
     # Retrieve the selected image name from the URL parameter
    image_name = request.args.get('image_name', '')
    
    # Find the corresponding image path
    selected_image_path = next((img['path'] for img in images if img['name'] == image_name), None)

    # Load the selected image
    selected_image = cv2.imread(os.path.join('static', selected_image_path))

    # Quantize the selected image
    k_means_clusters = 8  # You can adjust this value
    

    # Calculate histogram and texture descriptor for the quantized selected image
    selected_histogram = calculate_histogram(selected_image , k_means_clusters)
    selected_texture_descriptor = calculate_texture_descriptor(selected_image )

    # Calculate distances for each image in the database
    distances_combined = []
    for image_data in images:
        # Load the database image
        db_image = cv2.imread(os.path.join('static', image_data['path']))



        # Calculate histogram and texture descriptor for the quantized database image
        db_histogram = calculate_histogram( db_image, k_means_clusters)
        db_texture_descriptor = calculate_texture_descriptor(db_image)

        # Calculate distances
        distance_histogram = euclidean_distance(selected_histogram, db_histogram)
        distance_texture = manhattan_distance(selected_texture_descriptor, db_texture_descriptor)

        # Combine distances using a weighted average
        weight_histogram = 0.7  # Adjust weights as needed
        weight_texture = 0.3
        combined_distance = (weight_histogram * distance_histogram) + (weight_texture * distance_texture)

        distances_combined.append({'image': image_data, 'distance': combined_distance})

    # Sort distances by increasing order
    distances_combined.sort(key=lambda x: x['distance'])

    # Display the top 3 images based on combined distances
    top_images = distances_combined[:3]

    return render_template('show_top_similar_images.html', selected_image=selected_image_path, top_images=top_images)
    
    

if __name__ == '__main__':
    app.run(debug=True)


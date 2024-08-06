
import logging
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import hashlib
import random
import os
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import numpy as np
import tensorflow as tf

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to log messages with a consistent format
def log(message):
    logging.debug(message)

# Verify TensorFlow GPU setup
log("Checking TensorFlow GPU availability...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        log(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
    except RuntimeError as e:
        logging.error(e)
else:
    logging.warning("No GPU available, using CPU")

# Load the VGG16 model once outside the loop to avoid retracing warnings
log("Loading VGG16 model...")
model = VGG16(weights='imagenet', include_top=False)
log("VGG16 model loaded.")

def extract_features_vgg16(img_path):
    log(f"Extracting features from image: {img_path}")
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    log(f"Features extracted for image: {img_path}")
    return features.flatten()

def apply_random_transformations(img):
    log("Applying random transformations...")
    width, height = img.size
    
    # Modify random pixels
    for _ in range(5):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        r, g, b = img.getpixel((x, y))
        img.putpixel((x, y), ((r + random.randint(-10, 10)) % 256, (g + random.randint(-10, 10)) % 256, (b + random.randint(-10, 10)) % 256))
    
    # Add random noise
    noise_level = random.randint(5, 20)
    for _ in range(noise_level):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        r, g, b = img.getpixel((x, y))
        img.putpixel((x, y), ((r + random.randint(-5, 5)) % 256, (g + random.randint(-5, 5)) % 256, (b + random.randint(-5, 5)) % 256))
    
    # Apply random blur
    if random.choice([True, False]):
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
    
    # Apply random rotation
    if random.choice([True, False]):
        img = img.rotate(random.uniform(-5, 5))
    
    # Apply random flip
    if random.choice([True, False]):
        img = ImageOps.mirror(img)
    
    # Apply random color enhancement
    if random.choice([True, False]):
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    log("Random transformations applied.")
    return img

def modify_image(image_path, output_path):
    log(f"Modifying image: {image_path}")
    # Open the image
    img = Image.open(image_path).convert('RGB')
    
    # Apply random transformations
    img = apply_random_transformations(img)
    
    # Extract features using VGG16 and modify them
    features = extract_features_vgg16(image_path)
    features += np.random.normal(0, 0.01, features.shape)  # Perturb features slightly
    
    # Save with different compression
    img.save(output_path, quality=random.randint(85, 95))
    log(f"Image saved to: {output_path}")

def calculate_md5(image_path):
    log(f"Calculating MD5 for image: {image_path}")
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
        md5_hash = hashlib.md5(image_data).hexdigest()
    log(f"MD5 for {image_path}: {md5_hash}")
    return md5_hash

def process_images(input_directory, output_directory, iterations=1):
    log(f"Processing images in directory: {input_directory}")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        log(f"Created output directory: {output_directory}")
    
    image_count = 1
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_image = os.path.join(input_directory, filename)
            for i in range(iterations):
                output_path = os.path.join(output_directory, f"image{image_count}modified{i+1}.jpg")
                modify_image(input_image, output_path)
                original_md5 = calculate_md5(input_image)
                modified_md5 = calculate_md5(output_path)
                log(f"Processed {filename} (Iteration {i+1}):")
                log(f"Original MD5: {original_md5}")
                log(f"Modified MD5: {modified_md5}")
                log("----")
            image_count += 1

# Example usage
input_directory = 'ModelContent'
output_directory = 'ModelContentModified'
iterations = 5
process_images(input_directory, output_directory, iterations)

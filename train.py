import jittor as jt
from PIL import Image
import jclip as clip
import os
import argparse
from tqdm import tqdm
import random
import numpy as np

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='Dataset/')
parser.add_argument('--model_path', type=str, default='model/ViT-B-32.pkl')
parser.add_argument('--split', type=str, default='A')
args = parser.parse_args()

model, preprocess = clip.load("model/ViT-B-32.pkl")
classes = open('train_classes.txt').read().splitlines()

# Remove prefixes and format class names
new_classes = []
for c in classes:
    c = c.split(';')[0]
    if c.startswith('Animal'):
        c = c[7:]
    if c.startswith('Thu-dog'):
        c = c[8:]
    if c.startswith('Caltech-101'):
        c = c[12:]
    if c.startswith('Food-101'):
        c = c[9:]
    c = 'a photo of ' + c
    new_classes.append(c)

# Convert class descriptions to text features
class_descriptions = clip.tokenize(new_classes)
with jt.no_grad():
    text_features = model.encode_text(class_descriptions)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# Training data loading
train_labels = open(os.path.join(args.dataset_path, 'train.txt')).read().splitlines()
train_imgs = [l.split(' ')[0] for l in train_labels]
train_labels = [int(l.split(' ')[1]) for l in train_labels]

# Randomly select 4 images per class based on train_labels
from collections import defaultdict

cnt = defaultdict(list)
for i in range(len(train_imgs)):
    label = train_labels[i]
    cnt[label].append(train_imgs[i])

new_train_imgs = []
new_train_labels = []
for label, imgs in cnt.items():
    selected_imgs = random.sample(imgs, min(4, len(imgs)))  # Randomly select up to 4 images
    new_train_imgs.extend(selected_imgs)
    new_train_labels.extend([label] * len(selected_imgs))

# Define augmentation functions
def random_rotation(image):
    return image.rotate(random.uniform(-30, 30))

def add_noise(image, stddev=10):
    # Check if image is grayscale or RGB
    if image.mode == 'L':
        noise = np.random.normal(0, stddev, (image.size[1], image.size[0])).astype(np.uint8)
    else:
        noise = np.random.normal(0, stddev, (image.size[1], image.size[0], 3)).astype(np.uint8)
    
    noisy_image = Image.fromarray(np.clip(np.array(image) + noise, 0, 255).astype(np.uint8))
    return noisy_image

# Extend the training data with augmented images
augmented_train_imgs = []
augmented_train_labels = []
for img_path, label in zip(new_train_imgs, new_train_labels):
    img_path_full = os.path.join(args.dataset_path, img_path)
    image = Image.open(img_path_full)
    
    # Append original image
    augmented_train_imgs.append(image)
    augmented_train_labels.append(label)
    
    # Append rotated image
    rotated_img = random_rotation(image)
    augmented_train_imgs.append(rotated_img)
    augmented_train_labels.append(label)
    
    # Append noisy image
    noisy_img = add_noise(image, stddev=10)
    augmented_train_imgs.append(noisy_img)
    augmented_train_labels.append(label)
    
    # Append rotated and noisy image
    rotated_noisy_img = add_noise(random_rotation(image), stddev=10)
    augmented_train_imgs.append(rotated_noisy_img)
    augmented_train_labels.append(label)

# Convert labels to tensor
augmented_train_labels = [jt.int64([label]) for label in augmented_train_labels]

# Fine-tuning network
model.train()
optimizer = jt.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)

# Training loop
num_epochs = 20
batch_size = 16
save_path = 'saved_models'
os.makedirs(save_path, exist_ok=True)

for epoch in range(num_epochs):
    # Shuffle data at the start of each epoch
    combined = list(zip(augmented_train_imgs, augmented_train_labels))
    random.shuffle(combined)
    augmented_train_imgs[:], augmented_train_labels[:] = zip(*combined)

    total_loss = 0
    with jt.enable_grad():
        for i in tqdm(range(0, len(augmented_train_imgs), batch_size)):
            img_batch = augmented_train_imgs[i:i+batch_size]
            label_batch = augmented_train_labels[i:i+batch_size]

            images = []
            labels = []
            for img, label in zip(img_batch, label_batch):
                image = preprocess(img).unsqueeze(0)
                images.append(image)
                labels.append(label)
            
            images = jt.concat(images, dim=0)  # Concatenate images into a single batch tensor
            labels = jt.array([label.numpy()[0] for label in labels])  # Convert to tensor
            
            # Encode the images
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute logits for all classes
            logits_per_image, _ = model(images, class_descriptions)

            # Compute probabilities
            probs = logits_per_image.softmax(dim=-1)
            
            # Convert labels to one-hot encoding
            targets = jt.zeros((len(labels), logits_per_image.shape[1]))
            for idx, label in enumerate(labels):
                targets[idx, label] = 1.0
            
            # Compute loss
            loss = jt.nn.cross_entropy_loss(probs, labels)

            optimizer.backward(loss)  # Compute gradients
            optimizer.step()          # Update parameters
            total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(augmented_train_imgs)}")
    
    # Save model parameters
    model_path = os.path.join(save_path, f"test_data_strength16_10_epoch_{epoch+1}.pkl")
    jt.save(model.state_dict(), model_path)
    print(f"Model parameters saved at {model_path}")

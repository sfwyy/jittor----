import jittor as jt
from PIL import Image
import jclip as clip
import os
import argparse
from tqdm import tqdm
import random


jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='B')
args = parser.parse_args()

model, preprocess = clip.load("saved_models/best.pkl")
classes = open('test_classes.txt', encoding='utf-8').read().splitlines()

test_split = 'TestSet' + args.split
test_imgs_dir = 'Dataset/' + test_split
test_imgs = os.listdir(test_imgs_dir)

save_file = open(f'result.txt', 'w')

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



preds = []
with jt.no_grad():
    for img in tqdm(test_imgs):
        img_path = os.path.join(test_imgs_dir, img)
        image = Image.open(img_path)
        image = preprocess(image).unsqueeze(0)

        image_features = model.encode_image(image)
        #image_features += adapter(image_features) # 加上adapter
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 *
                    image_features @ text_features.transpose(0, 1)).softmax(
                        dim=-1)

        # top5 predictions
        _, top_labels = text_probs[0].topk(5)
        preds.append(top_labels)
        # save top5 predictions to file
        save_file.write(img + ' ' +
                        ' '.join([str(p.item()) for p in top_labels]) + '\n')
        save_file.flush()  # Ensure data is written to disk          
            
print(f"results saved at result.txt")


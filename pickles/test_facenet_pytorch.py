from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import imutils.paths as paths
import numpy as np
import cv2
import pickle
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(device=device)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# dataset = "./FaceNet/dataset/Test/Dima/"
# dataset = "./test7/"
dataset = "dataset/"
pickle_file = "facenet_pytorch.pickle"

imagepaths = list(paths.list_images(dataset))
knownEncodings = []
knownNames = []
for (i, imagePath) in enumerate(imagepaths):
    print("[INFO] processing image {} {}/{}".format(imagePath, i + 1, len(imagepaths)))
    name = imagePath.split(os.path.sep)[-2]
    img = Image.open(imagePath)

    # Detect faces
    boxes, _ = mtcnn.detect(img)

    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img)

    # Calculate embedding (unsqueeze to add batch dimension)
    encoding = resnet(img_cropped.unsqueeze(0)).detach().numpy()[0]

    knownEncodings.append(encoding)  # векторизация
    knownNames.append(name)
    data = {"encodings": knownEncodings, "names": knownNames}
    output = open(pickle_file, "wb")
    pickle.dump(data, output)
    output.close()

test_encodings = pickle_file
print('Reading DATASET', test_encodings)
test_face_data = pickle.loads(open(test_encodings, "rb").read())
count_data = test_face_data['names']
print(f'TEST DATASET include {len(count_data)} items.')
test_data = np.array(test_face_data['encodings'])
print(test_data.shape)
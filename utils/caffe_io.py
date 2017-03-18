import numpy as np
import cv2
import h5py
import os
import pdb

# This should be written as a IO module
# Load image, preprocess image
# Load video
# Dump to HDF5 for caffe training

def load_image(img_name):
  # BGR order, 0-255
  return cv2.imread(img_name)

def load_video_from_list(vid_name, frame_list):
    assert os.path.isfile(vid_name)
    cap = cv2.VideoCapture(vid_name)
    frames = []
    if not cap.isOpened():
        print 'Cannot open: ', vid_name
        return frames
    ind = 0
    find = 0
    while cap.grab() and find < len(frame_list):
        if ind == frame_list[find][0]:
            (flag, frame) = cap.retrieve()
            if flag:
                frames.append(frame)
            find += 1
        ind += 1
    return frames

def load_video(vid_name, sample_per_sec = 1):
  cap = cv2.VideoCapture(vid_name)
  frames = []
  if not cap.isOpened():
    print 'Cannot open', vid_name
    return frames
  # get FPS
  fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
  sample_step = np.ceil(float(fps) / sample_per_sec)
  ind = 0
  while cap.grab():
    ind += 1
    if ind % sample_step == 1:
      (flag, frame) = cap.retrieve()
      if flag:
        frames.append(frame)
  cap.release()
  return frames

def save_matrix(mat, output_path):
  batch = h5py.File(output_path, 'w')
  batch['data'] = np.array(mat)
  batch.close()

def save_as_hdf5(data, output_path):
  batch = h5py.File(output_path, 'w')
  for blob in data.keys():
    batch[blob] = np.array(data[blob])
  batch.close()

# Default parameters are for VGG net
# Input: Height x Width x Channel
# Output: #Sample x Channel x Height x Width
def transform_image(img, over_sample = False, mean_pix = [103.939, 116.779, 123.68], image_dim = 256, crop_dim = 224):
  # convert to BGR
  if len(img.shape) < 3 or img.shape[2] == 1:
    img = cv2.cvtColor(img, cv2.cv.CV_GRAY2BGR)
  # resize image, the shorter side is set to image_dim
  if img.shape[0] < img.shape[1]:
    # Note: OpenCV uses width first...
    dsize = (int(np.floor(float(image_dim)*img.shape[1]/img.shape[0])), image_dim)
  else:
    dsize = (image_dim, int(np.floor(float(image_dim)*img.shape[0]/img.shape[1])))
  img = cv2.resize(img, dsize, interpolation=cv2.INTER_CUBIC)

  # convert to float32
  img = img.astype(np.float32, copy=False)

  if over_sample:
    imgs = np.zeros((10, crop_dim, crop_dim, 3), dtype=np.float32)
  else:
    imgs = np.zeros((1, crop_dim, crop_dim, 3), dtype=np.float32)

  # crop
  indices_y = [0, img.shape[0]-crop_dim]
  indices_x = [0, img.shape[1]-crop_dim]
  center_y = int(np.floor(indices_y[1]/2))
  center_x = int(np.floor(indices_x[1]/2))

  imgs[0] = img[center_y:center_y+crop_dim, center_x:center_x+crop_dim, :]
  if over_sample:
    curr = 1
    for i in indices_y:
      for j in indices_x:
        imgs[curr] = img[i:i+crop_dim, j:j+crop_dim, :]
        imgs[curr+5] = imgs[curr, :, ::-1, :]
        curr += 1
    imgs[5] = imgs[0, :, ::-1, :]

  # subtract mean
  for c in range(3):
    imgs[:, :, :, c] = imgs[:, :, :, c] - mean_pix[c]
  # reorder axis
  return np.rollaxis(imgs, 3, 1)

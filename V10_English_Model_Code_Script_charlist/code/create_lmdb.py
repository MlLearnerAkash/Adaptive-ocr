# import os
# import lmdb
# import cv2
# import numpy as np
# import pandas as pd

# class LMDBCreator:
#     def __init__(self, image_dir, label_txt_file, df, output_path, check_valid=True):
#         """
#         Args:
#             image_dir (str): Directory containing image files.
#             label_txt_file (str): Text file with labels. Each line should contain the image filename and its label separated by whitespace.
#             df (pd.DataFrame): DataFrame with columns ['image_path', 'predicted_label', 'confidence'].
#             output_path (str): Target directory where the LMDB database will be created.
#             check_valid (bool): Whether to check if an image is valid.
#         """
#         self.image_dir = image_dir
#         self.label_txt_file = label_txt_file
#         self.df = df
#         self.output_path = output_path
#         self.check_valid = check_valid

#     def check_image_is_valid(self, imageBin):
#         """Check if an image binary is valid by attempting to decode it."""
#         if imageBin is None:
#             print("Empty image binary")
#             return False
#         imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
#         if len(imageBuf) > 0:
#             img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
#             if img is None:
#                 return False
#             imgH, imgW = img.shape[0], img.shape[1]
#             if imgH * imgW == 0:
#                 return False
#         else:
#             return False
#         return True

#     def write_cache(self, env, cache):
#         """Write a dictionary of key-value pairs into the LMDB environment."""
#         with env.begin(write=True) as txn:
#             for k, v in cache.items():
#                 # Store bytes directly; otherwise convert to string bytes.
#                 if isinstance(v, bytes):
#                     txn.put(str(k).encode(), v)
#                 else:
#                     txn.put(str(k).encode(), str(v).encode())

#     def create_dataset(self):
#         """
#         Combines samples from the label text file and the DataFrame and creates an LMDB database.
#         The text file should have lines formatted as:
#              <image_filename> <label>
#         The DataFrame should have columns:
#              ['image_path', 'predicted_label', 'confidence']
#         Returns:
#             int: The total number of samples successfully stored in LMDB.
#         """
#         # Read and parse the label text file.
#         samples = []
#         if os.path.exists(self.label_txt_file):
#             with open(self.label_txt_file, 'r') as f:
#                 for line in f:
#                     # Expect each line to have at least two parts: filename and label.
#                     parts = line.strip().split(maxsplit=1)
#                     if len(parts) < 2:
#                         continue
#                     filename, label = parts
#                     full_path = os.path.join(self.image_dir, filename)
#                     samples.append((full_path, label))
#         else:
#             print("Label text file not found:", self.label_txt_file)

#         # Process the DataFrame entries.
#         if self.df is not None and not self.df.empty:
#             for idx, row in self.df.iterrows():
#                 image_path_df = row['image_path']
#                 # If not an absolute path, join with image_dir.
#                 if not os.path.isabs(image_path_df):
#                     full_path = os.path.join(self.image_dir, image_path_df)
#                 else:
#                     full_path = image_path_df
#                 label = str(row['predicted_label']).strip()
#                 samples.append((full_path, label))
        
#         total_samples = len(samples)
#         print("Total combined samples: %d" % total_samples)

#         # Open LMDB environment (using a large map_size to ensure plenty of space)
#         env = lmdb.open(self.output_path, map_size=1099511627776)
#         cache = {}
#         cnt = 1

#         # Process each sample.
#         for img_path, label in samples:
#             if not os.path.exists(img_path):
#                 print("File does not exist:", img_path)
#                 continue
#             with open(img_path, 'rb') as f:
#                 imageBin = f.read()
#             if self.check_valid and not self.check_image_is_valid(imageBin):
#                 print("Invalid image:", img_path)
#                 continue

#             image_key = 'image-%09d' % cnt
#             label_key = 'label-%09d' % cnt
#             cache[image_key] = imageBin
#             cache[label_key] = label

#             # Flush cache to LMDB every 50000 samples.
#             if cnt % 50000 == 0:
#                 self.write_cache(env, cache)
#                 cache = {}
#                 print('Written %d / %d samples' % (cnt, total_samples))
#             cnt += 1

#         nSamples = cnt - 1
#         cache['num-samples'] = str(nSamples)
#         self.write_cache(env, cache)
#         print("Created LMDB dataset with %d samples" % nSamples)
#         return nSamples


import os
import lmdb
import cv2
import numpy as np
import pandas as pd

class LMDBCreator:
    def __init__(self, image_txt_file, label_txt_file, df, output_path, image_dir='', check_valid=True):
        """
        Args:
            image_txt_file (str): Text file with full image paths (one per line).
            label_txt_file (str): Text file with labels (one per line) corresponding to the images.
            df (pd.DataFrame): DataFrame with columns ['image_path', 'predicted_label', 'confidence'].
            output_path (str): Target directory where the LMDB database will be created.
            image_dir (str): Optional. Directory to prepend if image paths are relative.
            check_valid (bool): Whether to check if an image is valid.
        """
        self.image_txt_file = image_txt_file
        self.label_txt_file = label_txt_file
        self.df = df
        self.output_path = output_path
        self.image_dir = image_dir
        self.check_valid = check_valid

    def check_image_is_valid(self, imageBin):
        """Check if an image binary is valid by attempting to decode it."""
        if imageBin is None:
            print("Empty image binary")
            return False
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        if len(imageBuf) > 0:
            img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False
            imgH, imgW = img.shape[0], img.shape[1]
            if imgH * imgW == 0:
                return False
        else:
            return False
        return True

    def write_cache(self, env, cache):
        """Write a dictionary of key-value pairs into the LMDB environment."""
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                # Store bytes directly; otherwise convert to string bytes.
                if isinstance(v, bytes):
                    txn.put(str(k).encode(), v)
                else:
                    txn.put(str(k).encode(), str(v).encode())

    def create_dataset(self):
        """
        Combines samples from the image and label text files (linewise) and the DataFrame,
        and creates an LMDB database.
        
        Returns:
            int: The total number of samples successfully stored in LMDB.
        """
        samples = []
        
        # Process the text file samples.
        if os.path.exists(self.image_txt_file) and os.path.exists(self.label_txt_file):
            with open(self.image_txt_file, 'r') as f_img, open(self.label_txt_file, 'r') as f_lbl:
                image_paths = [line.strip() for line in f_img if line.strip() != '']
                labels = [line.strip() for line in f_lbl if line.strip() != '']
            
            if len(image_paths) != len(labels):
                print("Warning: Number of image paths and labels do not match. Using minimum count.")
            n = min(len(image_paths), len(labels))
            for i in range(n):
                img_path = image_paths[i]
                # If image_dir is provided and the path is not absolute, join them.
                if self.image_dir and not os.path.isabs(img_path):
                    full_path = os.path.join(self.image_dir, img_path)
                else:
                    full_path = img_path
                samples.append((full_path, labels[i]))
        else:
            print("One or both text files not found:", self.image_txt_file, self.label_txt_file)
        txt_samples = len(samples)
        print(">>>>>>>>>>>>samples from the txt file is:::::", txt_samples)

        # Process the DataFrame entries.
        if self.df is not None and not self.df.empty:
            for idx, row in self.df.iterrows():
                image_path_df = row['image_path']
                # If not absolute and image_dir is provided, join them.
                if self.image_dir and not os.path.isabs(image_path_df):
                    full_path = os.path.join(self.image_dir, image_path_df)
                else:
                    full_path = image_path_df
                label = str(row['predicted_label']).strip()
                samples.append((full_path, label))
        
        total_samples = len(samples)
        print(">>>>>>>>>>>samples added from dataframe is>>>>", {total_samples-txt_samples})
        print("Total combined samples: %d" % total_samples)

        # Open LMDB environment (using a large map_size to ensure plenty of space)
        env = lmdb.open(self.output_path, map_size=1099511627776)
        cache = {}
        cnt = 1

        # Process each sample.
        for img_path, label in samples:
            if not os.path.exists(img_path):
                print("File does not exist:", img_path)
                continue
            with open(img_path, 'rb') as f:
                imageBin = f.read()
            if self.check_valid and not self.check_image_is_valid(imageBin):
                print("Invalid image:", img_path)
                continue

            image_key = 'image-%09d' % cnt
            label_key = 'label-%09d' % cnt
            cache[image_key] = imageBin
            cache[label_key] = label

            # Flush cache to LMDB every 50000 samples.
            if cnt % 50000 == 0:
                self.write_cache(env, cache)
                cache = {}
                print('Written %d / %d samples' % (cnt, total_samples))
            cnt += 1

        nSamples = cnt - 1
        cache['num-samples'] = str(nSamples)
        self.write_cache(env, cache)
        print("Created LMDB dataset with %d samples" % nSamples)
        return nSamples

# imports
import os
import pandas as pd
import numpy as np
import pydicom
from skimage.measure import label, marching_cubes, mesh_surface_area, regionprops
from skimage.feature import graycomatrix, graycoprops
from scipy.spatial.distance import  pdist
from scipy.ndimage import zoom

from utils import helpers as hlp


def get_image_filepath(df: pd.DataFrame, patient_id: str, image_type: str) -> str:
        """
        Get the filepath to the image(s)
        """

        def access_dataframe_value(column: str) -> str:

            value = (
                df
                .loc[(df['Subject ID']==patient_id)&(df['Modality']==image_type),
                     column]
                .values[0]
            )
            return value

        base_dir = '/user/home/ms13525/scratch/mshds-ml-data-2025'
        dataset = access_dataframe_value('Dataset')
        filepath = access_dataframe_value('File Location')

        return os.path.join(base_dir, dataset, filepath)


def load_dicom(filepath: str) -> np.ndarray:

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")

    try:
        dicom_file = pydicom.dcmread(filepath)
        pixel_array = dicom_file.pixel_array
        # normalise the image
        image = (pixel_array / np.max(pixel_array) * 255).astype(np.uint8)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        raise(e)
    
    return image


def load_image_list(folderpath: str) -> list:
    """
    Load the images from the given filepath
    """
    
    image_filepaths = sorted(os.listdir(folderpath))
    images = []
    for file in image_filepaths:
        filepath = os.path.join(folderpath, file)
        image = load_dicom(filepath)
        images.append(image)
        del image
    
    return images


def load_images(patient_id: str, seg_filepath: str, ct_filepath: str) -> dict:

    # load the images
    image_type_dict = {
        'seg': load_image_list(seg_filepath),
        'ct': load_image_list(ct_filepath)
    }

    return image_type_dict


def create_3d_image(slices: list) -> np.ndarray:
    
    return np.stack(slices, axis=0)


def segmentation_features(tumour_array: np.ndarray, voxel_size: list) -> dict:

    if np.sum(tumour_array) > 0:
        _, n_tumours = label(tumour_array, return_num=True)
        verts,faces,_,_ = marching_cubes(tumour_array,0.5,spacing=voxel_size)
        area = mesh_surface_area(verts,faces)
        volume = np.sum(tumour_array > 0.1)*voxel_size[0]*voxel_size[1]*voxel_size[2]
        radius = (3.0/(4.0*np.pi)*volume)**(1.0/3)

        # break vertices into chunks to avoid memory issues
        chunk_size = 1000
        max_distance = 0
        for i in range(0, len(verts), chunk_size):
            chunk = verts[i:(i + chunk_size)]
            distance = pdist(chunk)
            max_distance = max(max_distance, np.amax(distance))


        features = {"n_tumours": n_tumours,
                    "maximum_diameter": max_distance,
                    "surface_area": area,
                    "surface_to_volume_ratio" : area/volume,
                    "volume": volume,
                    "radius": radius
                    }


        return features
    else:
        return 0



def shape_morphological_features(tumour_array: np.ndarray) -> dict:

    # label the connected regions in the tumor array
    labeled_tumour = label(tumour_array)
    properties = regionprops(labeled_tumour)

    # calculate surface area
    verts, faces, _, _ = marching_cubes(tumour_array, 0.5)
    surface_area = mesh_surface_area(verts, faces)

    # calculate max diameter
    chunk_size = 1000
    max_distance = 0
    for i in range(0, len(verts), chunk_size):
        chunk = verts[i:(i + chunk_size)]
        distance = pdist(chunk)
        max_distance = max(max_distance, np.amax(distance))
     
    features = {
        "n_tumours": len(properties),
        "maximum_diameter": max_distance,
        "surface_area": surface_area,
        "surface_to_volume_ratio": 0,
        "volume": 0,
        "radius": 0,
        "sphericity": 0,
        "elongation": [],
        "compactness": 0
    }
    
    for prop in properties:
        # calculate volume
        volume = prop.area

        # calculate elongation
        try:
            min_axis_length = min(prop.major_axis_length, prop.minor_axis_length)
            max_axis_length = max(prop.major_axis_length, prop.minor_axis_length)
            features["elongation"].append(max_axis_length / min_axis_length)
        except:
            # no elongation
            features["elongation"].append(1)

        # calculate radius
        radius = (3 * volume / (4 * np.pi))**(1/3)

        # aggregate features
        features["volume"] += volume
        features["radius"] += radius

    # calculate surface to volume ratio
    features["surface_to_volume_ratio"] = features["surface_area"] / features["volume"]
    # calculate sphericity
    features["sphericity"] = (np.pi**(1/3) * (6 * features["volume"])**(2/3)) / surface_area
    # calculate compactness
    features["compactness"] = (surface_area**3)/(features["volume"]**2)

    # average elongation over all regions
    features["elongation"] = np.mean(features["elongation"])

    return features



def glcm_features_3d(image: np.ndarray,
                     mask: np.ndarray,
                     distances=[1],
                     angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]) -> dict:
    
    # apply the mask to the image
    masked_image = image * (mask > 0)

    # normalize the image to have integer values
    normalized_image = (
        ((masked_image - masked_image.min()) / (masked_image.max() - masked_image.min()) * 255)
        .astype(np.uint8))

    # initialize GLCM features
    features = {
        "contrast": 0,
        "correlation": 0,
        "dissimilarity": 0,
        "homogeneity": 0,
        "energy": 0
    }

    # iterate through each slice of the 3D image
    for i in range(normalized_image.shape[0]):
        slice_image = normalized_image[i, :, :]
        slice_mask = mask[i, :, :]

        # skip slices without any mask
        if np.sum(slice_mask) == 0:
            continue

        # compute GLCM for the slice
        glcm = graycomatrix(slice_image, distances=distances, angles=angles,
                            levels=256, symmetric=True, normed=True)

        # accumulate GLCM features
        for feature in features.keys():
            features[feature] += np.mean(graycoprops(glcm, feature))

    # average the features over all slices
    num_slices = np.sum([np.sum(mask[i, :, :]) > 0 for i in range(mask.shape[0])])
    if num_slices > 0:
        for feature in features.keys():
            features[feature] /= num_slices

    return features
    

def intensity_features(img: np.ndarray, mask: np.ndarray) -> dict:

    tumour_pixels = img[mask > 0]
    features = {
        "mean_intensity": np.mean(tumour_pixels),
        "std_intensity": np.std(tumour_pixels),
        "min_intensity": np.min(tumour_pixels),
        "max_intensity": np.max(tumour_pixels),
        "median_intensity": np.median(tumour_pixels)
    }
    return features


def extract_tumour_properties(img: np.ndarray, mask: np.ndarray) -> list:
    """
    Extract tumour properties using the
    grayscale and segmented images
    """

    # extract features from segmented image
    img_lesion = mask > 0
    #segmented_features = segmentation_features(img_lesion, [1, 1, 1])
    morph_features = shape_morphological_features(mask)

    # re-scale the grayscale image to be same size as the mask
    # and halve the dimensions to decrease processing time
    img = zoom(img, (0.5*mask.shape[0]/img.shape[0], 0.5, 0.5), order=1)
    mask = zoom(mask, (0.5, 0.5, 0.5), order=1)

    # extract features from the grayscale image
    glcm_features = glcm_features_3d(img.copy(), mask.copy())

    # extract features from both the grayscale and the segmented images
    intensity_feats = intensity_features(img.copy(), mask.copy())

    features = (#list(segmented_features.values())
                list(morph_features.values())
                + list(intensity_feats.values())
                + list(glcm_features.values()))

    return features



# imports
import os
import pandas as pd
import numpy as np
from skimage.measure import label, marching_cubes, mesh_surface_area
from scipy.spatial.distance import  pdist
from scipy.ndimage import zoom

from code.utils import helpers as hlp


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

        dataset = access_dataframe_value('Dataset')
        filepath = access_dataframe_value('File Location')

        return os.path.join(self.input_path, dataset, filepath)


def load_image_list(folderpath: str) -> list:
    """
    Load the images from the given filepath
    """
    
    image_filepaths = sorted(os.listdir(folderpath))
    images = []
    for file in image_filepaths:
        filepath = os.path.join(folderpath, file)
        image = hlp.load_dicom(filepath, self.LOGGER)
        images.append(image)
        del image
    
    return images


def load_images(patient_id: str) -> dict:

    metadata = hlp.get_metadata()

    seg_filepath = get_image_filepath(metadata, patient_id, 'SEG')
    ct_filepath = get_image_filepath(metadata, patient_id, 'CT')

        
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

    
def gray_level_cooccurrence_features(img: np.ndarray, mask: np.ndarray) -> dict:

    bins = np.arange(np.amin(img),np.amax(img),step=25)

    bin_img = np.digitize(img,bins)

    glcm = hlp.calculate_glcm2(bin_img, mask, bins.size)

    glcm = glcm/np.sum(glcm,axis=(0,1))

    ix = np.array(np.arange(1,bins.size+1)[:,np.newaxis,np.newaxis],dtype=np.float64)
    iy = np.array(np.arange(1, bins.size + 1)[np.newaxis,:, np.newaxis],dtype=np.float64)


    px = np.sum(glcm,axis=0)
    px = px[np.newaxis,...]
    py = np.sum(glcm,axis=1)
    py = py[:,np.newaxis,:]

    ux = np.mean(glcm,axis=0)
    ux = ux[np.newaxis,...]
    uy = np.mean(glcm,axis=1)
    uy = uy[:,np.newaxis,:]

    sigma_x = np.std(glcm,axis=0)
    sigma_x = sigma_x[np.newaxis,...]
    sigma_y = np.std(glcm,axis=1)
    sigma_y = sigma_y[:,np.newaxis,:]

    pxylog = np.log2(px * py+1e-7)
    pxylog[(px * py) == 0] = 0

    pxyp = np.zeros((2*bins.size,glcm.shape[2]))
    ki = ix+iy
    ki = ki[:,:,0]

    for angle in range(glcm.shape[2]):
        for k in range(1,2*bins.size):
            glcm_view =glcm[...,angle]
            pxyp[k,angle] = np.sum(glcm_view[ki==(k+1)])

    pxyplog = np.log2(pxyp+1e-7)
    pxyplog[pxyp == 0] = 0

    pxym = np.zeros((bins.size, glcm.shape[2]))
    ki = np.abs(ix - iy)
    ki = ki[:, :, 0]
    for angle in range(glcm.shape[2]):
        for k in range(0, bins.size):
            glcm_view = glcm[..., angle]
            pxym[k, :] = np.sum(glcm_view[ki == k])

    pxymlog = np.log2(pxym+1e-7)
    pxymlog[pxym == 0] = 0

    inverse_variance = 0

    for angle in range(glcm.shape[2]):
        glcm_view = glcm[:,:,angle]
        index = ix != iy
        diff = ix-iy
        diff = diff[...,0]
        index = index[...,0]
        inverse_variance += np.sum(glcm_view[index]/(diff[index])**2)
    inverse_variance /= glcm.shape[2]


    features = {"contrast": np.mean(np.sum((ix-iy)**2*glcm,axis=(0,1))),
                "correlation": np.mean(np.sum((ix*iy*glcm-ux*uy)/(sigma_x*sigma_y+1e-6),axis=(0,1))),
                "dissimilarity": np.mean(np.sum(np.abs(ix-iy)*glcm,axis=(0,1))),
                "homogeneity": np.mean(np.sum(glcm/(1+np.abs(ix-iy)),axis=(0,1)))
                }

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
    segmented_features = segmentation_features(img_lesion, [1, 1, 1])

    # re-scale the grayscale image to be same size as the mask
    # and halve the dimensions to decrease processing time
    img = zoom(img, (0.5*mask.shape[0]/img.shape[0], 0.5, 0.5), order=1)
    mask = zoom(mask, (0.5, 0.5, 0.5), order=1)

    # extract features from the grayscale image
    glcm_features = gray_level_cooccurrence_features(img, mask)

    # extract features from both the grayscale and the segmented images
    intensity_features = intensity_features(img, mask)

    features = (list(segmented_features.values())
                + list(intensity_features.values())
                + list(glcm_features.values()))

    return features



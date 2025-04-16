# imports
import os
import pandas as pd
import numpy as np
from skimage.measure import label, regionprops_table, regionprops, marching_cubes, mesh_surface_area
from skimage.morphology import remove_small_objects
from scipy.spatial.distance import  pdist
from scipy.ndimage import zoom
from joblib import Parallel, delayed
from configparser import ConfigParser
import logging
import warnings
from datetime import datetime

from code.utils import helpers as hlp

warnings.filterwarnings("ignore", category=UserWarning, module='skimage')


class ProcessData:
    def __init__(self, config: ConfigParser, logger: logging.Logger,
                 clinical1: pd.DataFrame, clinical2: pd.DataFrame,
                 image_data: dict):
        
        self.output_path = config['output_path']
        self.LOGGER = logger
        self.clinical1 = clinical1
        self.clinical2 = clinical2
        self.image_data = image_data

    
    def process_data(self) -> None:
        """
        Process the clinical data and (PET)CT images
        ready for training the models
        """
        
        #self.LOGGER.info('Processing the first clinical data')
        #self.clinical1_clean = self.process_single_clinical_data(self.clinical1)
        
        #self.LOGGER.info('Processing the second clinical data')
        #self.clinical2_clean = self.process_single_clinical_data(self.clinical2)
        
        #self.LOGGER.info('Joining the processed clinical datasets')
        #self.clinical_clean = self.process_clinical_data()
        
        self.LOGGER.info('Processing the CT images')
        self.image_features_df = self.process_images()

        return None

    
    def process_clinical1_data(self) -> pd.DataFrame:
        """
        Process both clinical datasets individually.
        The processing steps are:


        Finally, the processed data are saved to the clean data folder.
        """

        pass

    
    def process_clinical2_data(self) -> pd.DataFrame:
        """
        Process both clinical datasets individually.
        The processing steps are:


        Finally, the processed data are saved to the clean data folder.
        """

        pass

    
    def process_clinical_data(self) -> pd.DataFrame:
        """
        Join the processed clinical datasets together.
        The processing steps are:


        Finally, the processed data are saved to the clean data folder.
        """

        pass


    def process_images(self) -> pd.DataFrame:
        """
        Process the original CT images to make a single
        3D image for each patient.
        Extract tumour properties for training the models
        and save the processed data as a CSV file.
        """

        output_dir = os.path.join(self.output_path, 'images')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # create 3D grayscale image for each patient
        # and save the segmented and 3D images to the clean data folder
        image_dict = {}
        for patient_id, images in self.image_data.items():
            image_3d = hlp.create_3d_image(images['ct'])

            image_dict[patient_id] = {
                'segmented': images['segmented'][0],
                'ct': image_3d
            }

            seg_out_path = os.path.join(output_dir, patient_id, 'segmented_image.tif')
            ct_out_path = os.path.join(output_dir, patient_id, 'grayscale_image.tif')
            hlp.save_medical_image(images['segmented'][0], "segmented image", seg_out_path, self.LOGGER)
            hlp.save_medical_image(image_3d, "3D CT image", ct_out_path, self.LOGGER)

        msg = ("3D grayscale images created successfully "
               f"and saved to {output_dir}")
        self.LOGGER.info(msg)
        
        # extract tumour properties for each patient
        feature_data = []
        for patient_id, images in image_dict.items():
            img = images['ct']
            mask = images['segmented']

            features = self.extract_tumour_properties(img, mask)
            feature_data.append([patient_id] + features)
        
        # create a dataframe for the tumour features
        columns = (['patient_id'] +
                   ["n_tumours", "maximum_diameter", "surface_area", # segmented features
                    "surface_to_volume_ratio", "volume", "radius"] +
                   ["mean_intensity", "std_intensity", "min_intensity", # intensity features
                    "max_intensity", "median_intensity"] +
                   ["contrast", "correlation", "dissimilarity", "homogeneity"]) # grayscale features
        df = pd.DataFrame(data=feature_data, columns=columns)
        
        self.LOGGER.info("Tumour features extracted successfully")

        # save the tumour features to the clean data folder
        output_filepath = os.path.join(output_dir, 'tumour_features.csv')
        hlp.save_csv(df, "tumour features", output_filepath, self.LOGGER)

        return df


    def extract_tumour_properties(self, img: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
        """
        Extract tumour properties using the
        grayscale and segmented images
        """

        # extract features from segmented image
        img_lesion = mask > 0
        segmented_features = self.segmentation_features(img_lesion, [1, 1, 1])
        
        # re-scale the grayscale image to be same size as the mask
        # and halve the dimensions to decrease processing time
        img = zoom(img, (0.5*mask.shape[0]/img.shape[0], 0.5, 0.5), order=1)
        mask = zoom(mask, (0.5, 0.5, 0.5), order=1)

        # extract features from the grayscale image
        glcm_features = self.gray_level_cooccurrence_features(img, mask)

        # extract features from both the grayscale and the segmented images
        intensity_features = self.intensity_features(img, mask)

        features = (list(segmented_features.values())
                    + list(intensity_features.values())
                    + list(glcm_features.values()))
        
        return features


    def segmentation_features(self, tumour_array: np.ndarray, voxel_size: list) -> dict:
        """
        Extract tumour features from the segmented images
        """
        
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

    
    def gray_level_cooccurrence_features(self, img: np.ndarray, mask: np.ndarray) -> dict:
        """
        Extract features from the original 3D CT images
        """

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
    

    def intensity_features(self, img: np.ndarray, mask: np.ndarray) -> dict:
        """
        Extract tumour intensity features using both the
        grayscale and segmented images
        """
        
        tumour_pixels = img[mask > 0]
        features = {
            "mean_intensity": np.mean(tumour_pixels),
            "std_intensity": np.std(tumour_pixels),
            "min_intensity": np.min(tumour_pixels),
            "max_intensity": np.max(tumour_pixels),
            "median_intensity": np.median(tumour_pixels)
        }
        return features


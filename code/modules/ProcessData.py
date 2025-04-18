# imports
import os
import pandas as pd
import numpy as np
from skimage.measure import label, marching_cubes, mesh_surface_area
from scipy.spatial.distance import  pdist
from scipy.ndimage import zoom
from joblib import Parallel, delayed
from configparser import ConfigParser
import logging
import warnings

from code.utils import helpers as hlp

warnings.filterwarnings("ignore", category=UserWarning, module='skimage')
pd.set_option('future.no_silent_downcasting', True)


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
        
        self.LOGGER.info('Processing the first clinical dataset')
        self.clinical1_clean = self.process_clinical1_data(self.clinical1)
        
        self.LOGGER.info('Processing the second clinical data')
        self.clinical2_clean = self.process_clinical2_data(self.clinical2)
        
        self.LOGGER.info('Joining the processed clinical datasets')
        self.clinical_joined = self.join_clinical_data()
        
        self.LOGGER.info('Processing the CT images')
        self.image_features_df = self.process_images()

        # join image data to clinical data
        self.full_data = self.clinical_joined.merge(self.image_features_df,
                                                    on='patient_id',
                                                    how='left')
        # save the data
        output_path = os.path.join(self.output_path, 'full_clean_data.csv')
        hlp.save_csv(self.full_data, 'full cleaned data', output_path, self.LOGGER)

        return None

    
    def process_clinical1_data(self) -> pd.DataFrame:
        """
        Process first clinical dataset.
        The processing steps are:

        1) clean column names
        2) rename columns
        3) categorise data
        4) derive 2 year survival feature

        Finally, the processed data are saved to the clean data folder.
        """

        df = self.clinical1.copy()

        df = hlp.clean_column_names(df)

        rename_cols = {
            'patientid': 'patient_id'
        }
        df = df.rename(columns=rename_cols)

        stage_categories = {
            'I': 1, 'II': 2, 'IIIa': 3, 'IIIb': 3
        }
        df.replace(stage_categories, inplace=True)

        # for ordered categories replace null values with -1
        df['clinical_t_stage'] = df['clinical_t_stage'].fillna(-1)
        df['clinical_n_stage'] = df['clinical_n_stage'].fillna(-1)
        df['clinical_m_stage'] = df['clinical_m_stage'].fillna(-1)
        df['overall_stage'] = df['overall_stage'].fillna(-1)

        df['histology'] = df['histology'].str.replace(' ', '_')

        # N stages go from N0-N3, therefore mark N4 as null
        df['clinical_n_stage'] = df['clinical_n_stage'].replace({4: np.nan})

        # derive 2 year survival
        df['survivaltime_yrs'] = df['survival_time']/364.25
        # remove people who have are still alive but their survival
        # time is less than 2 years
        df = df[~((df['survivaltime_yrs']<2)&(df['deadstatus_event']==0))].reset_index(drop=True).copy()

        df['death_2years'] = [1 if ((df.loc[i, 'survivaltime_yrs']<2)&(df.loc[i, 'deadstatus_event']==1))
                              else 0 for i in range(len(df))]
        

        # drop columns that wouldn't be available in clinical setting
        drop_cols = ['survival_time', 'deadstatus_event', 'survivaltime_yrs']
        df.drop(columns=drop_cols, inplace=True)

        # save cleaned dataset
        output_path = os.path.join(self.output_path, 'clinical1_clean.csv')
        hlp.save_csv(df, 'cleaned clinical1 data', output_path, self.LOGGER)

        return df

    
    def process_clinical2_data(self) -> pd.DataFrame:
        """
        Process second clinical dataset.
        The processing steps are:

        1) clean column names
        2) rename columns
        3) categorise data
        4) derive 2 year survival feature

        Finally, the processed data are saved to the clean data folder.
        """

        df = self.clinical2

        df = hlp.clean_column_names(df)

        rename_cols = {
            'case_id': 'patient_id',
            'patient_affiliation': 'affiliation',
            'age_at_histological_diagnosis': 'age',
            'weight_lbs': 'weight',
            'gg': 'gg_percentage',
            'pathological_t_stage': 'clinical_t_stage',
            'pathological_n_stage': 'clinical_n_stage',
            'pathological_m_stage': 'clinical_m_stage',
            'histopathological_grade': 'overall_stage',
            'pleural_invasion_elastic_visceral_or_parietal': 'pleural_invasion'
        }
        df = df.rename(columns=rename_cols)

        # drop columns that wouldn't be available in clinical setting
        drop_cols = ['recurrance', 'recurrence_location',
                     'date_of_recurrence', 'quit_smoking_year']
        df.drop(columns=drop_cols, inplace=True)

        # make the string values lower case
        obj_cols = [col for col in df.columns if df[col].dtype == 'object']

        for col in obj_cols:
            if col=='patient_id':
                continue
            df[col] = df[col].str.lower()
        
        # if someone has never smoked then make their pack years 0
        # so they don't get imputed with 'impossible' values during model training
        df['pack_years'] = [0 if df.loc[i, 'smoking_status']=='nonsmoker'
                            else df.loc[i, 'pack_years'] for i in range(len(df))]
        
        missing_value_replacement = {
            'notassessed': np.nan, 'not recorded in database': np.nan,
            'notcollected': np.nan, 'unknown': np.nan
        }

        binary_value_replacement = {
            'dead': 1, 'alive': 0
        }

        gg_replacement = {
            '0%': 0, '>0 - 25%': 1, '25 - 50%': 2, '50 - 75%': 3, '75 - < 100%': 4, '100%': 5
        }

        t_stage_replacement = {
            'tis': 0, 't1a': 1, 't1b': 1, 't2a':2, 't2b': 2, 't3': 3, 't4': 4
        }

        n_stage_replacement = {
            'n0': 0, 'n1': 1, 'n2': 2, 'n3': 3
        }

        m_stage_replacement = {
            'm0': 0, 'm1a': 1, 'm1b': 2, 'm1c': 3
        }

        hist_grade_replacement = {
            'g1 well differentiated': 1,
            'other, type i: well to moderately differentiated': 1,
            'g2 moderately differentiated': 2,
            'other, type ii: moderately to poorly differentiated': 2,
            'g3 poorly differentiated': 3
        }

        word_replacements = {
            'nsclc nos (not otherwise specified)': 'nos',
            'african-american': 'african_american',
            'hispanic/latino': 'hispanic_latino',
            'native hawaiian/pacific islander': 'native_islander'
        }

        replacements = [missing_value_replacement, binary_value_replacement, gg_replacement,
                        t_stage_replacement, n_stage_replacement, m_stage_replacement,
                        hist_grade_replacement, word_replacements]
        replacement_dict = {}
        for dictionary in replacements:
            for key, value in dictionary.items():
                replacement_dict[key] = value

        df.replace(replacement_dict, inplace=True)

        # for ordered categories replace null values with -1
        df['gg_percentage'] = df['gg_percentage'].fillna(-1)
        df['clinical_t_stage'] = df['clinical_t_stage'].fillna(-1)
        df['clinical_n_stage'] = df['clinical_n_stage'].fillna(-1)
        df['clinical_m_stage'] = df['clinical_m_stage'].fillna(-1)
        df['overall_stage'] = df['overall_stage'].fillna(-1)

        for col in obj_cols:
            if col=='patient_id':
                continue
            df[col] = df[col].str.replace(' ', '_')

        # derive a variable for survival time
        df['date_of_last_known_alive'] = pd.to_datetime(df['date_of_last_known_alive'])
        df['ct_date'] = pd.to_datetime(df['ct_date'])
        df['survival_time_dv'] = ((df['date_of_last_known_alive'] - df['ct_date']).dt.days
                                  - df['days_between_ct_and_surgery'])
        df['survival_time_dv'] = [df.loc[i, 'survival_time'] if pd.isna(df.loc[i, 'survival_time_dv'])
                                  else df.loc[i, 'survival_time_dv'] for i in df.index]
        df['survivaltime_yrs'] = df['survival_time_dv']/364.25

        # remove people who have are still alive but their survival
        # time is less than 2 years
        df = df[~((df['survivaltime_yrs']<2)&(df['survival_status']==0))].reset_index(drop=True).copy()

        df['death_2years'] = [1 if ((df.loc[i, 'survivaltime_yrs']<2)&(df.loc[i, 'survival_status']==1))
                              else 0 for i in range(len(df))]
        

        # drop columns that wouldn't be available in clinical setting
        drop_cols = ['date_of_last_known_alive', 'survival_status', 'date_of_death',
                     'time_to_death_days', 'ct_date', 'days_between_ct_and_surgery',
                     'pet_date', 'survival_time_dv', 'survivaltime_yrs']
        df.drop(columns=drop_cols, inplace=True)

        # save cleaned dataset
        output_path = os.path.join(self.output_path, 'clinical2_clean.csv')
        hlp.save_csv(df, 'cleaned clinical2 data', output_path, self.LOGGER)

        return df

    
    def join_clinical_data(self) -> pd.DataFrame:
        """
        Join the processed clinical datasets together and
        save the joined dataset to the clean data folder.
        """

        df = pd.concat([self.clinical1_clean, self.clinical2_clean], axis=0)

        # mark null categorical values as unknown or not_applicable
        # for variables that are in one of the datasets but not the other
        cols_to_mark_unkown = ['affiliation', 'smoking_status', 'ethnicity']
        df[cols_to_mark_unkown] = df[cols_to_mark_unkown].fillna('unknown')

        df['pack_years'] = df['pack_years'].fillna(-1)

        # save the joined dataset
        output_path = os.path.join(self.output_path, 'clinical_joined.csv')
        hlp.save_csv(df, 'joined clinical data', output_path, self.LOGGER)

        return df


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
        patient_data = []
        for patient_id, images in self.image_data.items():
            patient_data.append((patient_id, images))

        image_dict_list = Parallel(n_jobs=-1)(delayed(self.clean_images)(id_, imgs) for id_, imgs in patient_data)
        
        image_dict = {}
        for dictionary in image_dict_list:
            for patient_id, images in dictionary.items():
                image_dict[patient_id] = images

        msg = ("3D grayscale images created successfully "
               f"and saved to {output_dir}")
        self.LOGGER.info(msg)
        
        # extract tumour properties for each patient
        patient_data = []
        for patient_id, images in image_dict.items():
            patient_data.append((patient_id, images))
        
        feature_data = Parallel(n_jobs=-1)(delayed(self.tumour_features)(id_, imgs) for id_, imgs in patient_data)
        
        # create a dataframe for the tumour features
        columns = (['patient_id'] +
                   ["n_tumours", "maximum_diameter", "surface_area", # segmented features
                    "surface_to_volume_ratio", "volume", "radius"] +
                   ["mean_intensity", "std_intensity", "min_intensity", # intensity features
                    "max_intensity", "median_intensity"] +
                   ["contrast", "correlation", "dissimilarity", "homogeneity"]) # grayscale features
        df = pd.DataFrame(data=feature_data, columns=columns)
        
        self.LOGGER.info("Tumour features extracted successfully")

        # join the clinical data to get the outcome for each patient
        clinical = self.clinical_joined.copy()
        clinical_out = clinical[['patient_id', 'death_2years']]
        df = df.merge(right=clinical_out, on='patient_id', how='left')

        # save the tumour features to the clean data folder
        output_filepath = os.path.join(self.output_path, 'tumour_features.csv')
        hlp.save_csv(df, "tumour features", output_filepath, self.LOGGER)

        return df
    

    def clean_images(self, patient_id: str, images: dict) -> dict:
        """
        Function to allow for parallelisation of image cleaning
        """

        output_dir = os.path.join(self.output_path, 'images')

        image_dict = {}

        image_3d = hlp.create_3d_image(images['ct'])

        image_dict[patient_id] = {
            'segmented': images['segmented'][0],
            'ct': image_3d
        }

        seg_out_path = os.path.join(output_dir, patient_id, 'segmented_image.tif')
        ct_out_path = os.path.join(output_dir, patient_id, 'grayscale_image.tif')
        hlp.save_medical_image(images['segmented'][0], "segmented image", seg_out_path, self.LOGGER)
        hlp.save_medical_image(image_3d, "3D CT image", ct_out_path, self.LOGGER)

        return image_dict
    

    def tumour_features(self, patient_id: str, images: dict) -> list:
        """
        Function to allow for parallelisation of image processing
        """
        img = images['ct']
        mask = images['segmented']

        features = self.extract_tumour_properties(img, mask)

        return [patient_id] + features



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


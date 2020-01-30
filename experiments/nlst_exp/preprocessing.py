"""
Preprocessing script for NLST experiment. After applying preprocessing, images are saved as numpy arrays and the meta
information for the corresponding patient is stored as a line in the dataframe saved as info_df.pickle.
"""
import os
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool
import pandas as pd
import numpy.testing as npt
from skimage.transform import resize
import subprocess
import pickle
import configs


def resample_array(src_imgs, src_spacing, target_spacing):
    """
    Resampling taken from lidc experiment MDT.
    """
    src_spacing = np.round(src_spacing, 3)
    target_shape = [int(src_imgs.shape[ix] * src_spacing[::-1][ix] / target_spacing[::-1][ix]) for ix in range(len(src_imgs.shape))]
    for i in range(len(target_shape)):
        try:
            assert target_shape[i] > 0
        except:
            raise AssertionError("AssertionError:", src_imgs.shape, src_spacing, target_spacing)

    img = src_imgs.astype(float)
    resampled_img = resize(img, target_shape, order=1, clip=True, mode='edge').astype('float32')

    return resampled_img


def intensity_normalization(np_image):
    '''
    Normalize image: clip to expected HU bounds [−1200,600] and normalize to [0., 1.] with type float32.
    np_image: the image to be clipped and normalized.
    '''
    np_image = np.clip(np_image, -1200, 600)
    np_image = (np_image - np.min(np_image)) / np.ptp(np_image)

    return np_image.astype(np.float32)


def pp_patient(inputs):
    """
    Preprocessing of a patient.
    imgs are not yet resampled, note that you should also resample segmenations if you resample.
    """
    # image processing
    ix, path = inputs
    pid = path.split('/')[-1][:-4]
    img = sitk.ReadImage(path)
    np_image = sitk.GetArrayFromImage(img)
    np_origin = np.array(list(reversed(img.GetOrigin())))
    np_spacing = np.array(list(reversed(img.GetSpacing())))

    print('Processing {}'.format(pid), np_spacing, np_image.shape)
    print('Image direction: {}'.format(img.GetDirection()))
    # np_image = resample_array(np_image, img.GetSpacing(), cf.target_spacing)
    np_image = intensity_normalization(np_image)

    # roi processing
    final_rois = np.zeros_like(np_image, dtype=np.uint8)
    seg_path = os.path.join(cf.root_dir, 'training_data_segmentations')
    lesion_paths = ([os.path.join(seg_path, ii) for ii in os.listdir(seg_path) if pid in ii and 'mhd' in ii])

    for lp in lesion_paths:
        lesion_id = lp[-7]
        roi = sitk.ReadImage(lp)
        print('Roi direction: {}'.format(roi.GetDirection()))
        np_roi = sitk.GetArrayFromImage(roi).astype(np.uint8)
        np_roi_origin = np.array(list(reversed(roi.GetOrigin())))
        np_roi_spacing = np.array(list(reversed(roi.GetSpacing())))

        # compute offset
        offset = np.rint(abs((np_roi_origin - np_origin)) / np_roi_spacing).astype(int)
        z, y, x = offset
        a, b, c = np_roi.shape

        # put the lesion segmentation in the right place
        try:
            if img.GetDirection() == roi.GetDirection():
                final_rois[z:z+a, y:y+b, x:x+c] = np_roi * int(lesion_id)
            else:
                final_rois[z:z + a, y:y + b, x:x + c] = np.flipud(np_roi) * int(lesion_id)

        except ValueError:
            print('Roi went out of the image. PID: {}, LesionID: {}'.format(pid, lesion_id))
            print('Image origin: {}, Roi origin {}, spacing: {}'.format(np_origin, np_roi_origin, np_spacing))

    # save img and final rois
    np.save(os.path.join(cf.pp_dir, '{}_rois.npy'.format(pid)), final_rois)
    np.save(os.path.join(cf.pp_dir, '{}_img.npy'.format(pid)), np_image)

    # stuff for meta info, malignancy all set to 1?
    fg_slices = [ii for ii in np.unique(np.argwhere(final_rois != 0)[:, 0])]
    mal_labels = np.ones(len(lesion_paths))

    with open(os.path.join(cf.pp_dir, 'meta_info_{}.pickle'.format(pid)), 'wb') as handle:
        meta_info_dict = {'pid': pid, 'class_target': mal_labels, 'spacing': img.GetSpacing(), 'fg_slices': fg_slices}
        pickle.dump(meta_info_dict, handle)


if __name__ == "__main__":
    cf = configs.configs()
    paths = [os.path.join(cf.raw_data_dir, ii) for ii in os.listdir(cf.raw_data_dir) if 'mhd' in ii]

    if not os.path.exists(cf.pp_dir):
        os.mkdir(cf.pp_dir)

    for i in enumerate(paths):
        pp_patient(i)

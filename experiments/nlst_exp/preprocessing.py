"""
Preprocessing script for NLST experiment. After applying preprocessing, images are saved as numpy arrays and the meta
information for the corresponding patient is stored as a line in the dataframe saved as info_df.pickle.
"""
import os
import SimpleITK as sitk
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import pandas as pd
import numpy.testing as npt
from skimage.transform import resize
import subprocess
import pickle
import configs
import argparse


def resample_array(src_imgs, src_spacing, target_spacing):
    """
    Resampling taken from lidc experiment MDT.
    """
    src_spacing = np.round(src_spacing, 3)
    target_shape = [int(src_imgs.shape[ix] * src_spacing[::-1][ix] / target_spacing[::-1][ix]) for ix in
                    range(len(src_imgs.shape))]
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

    print('Processing number {} with pid: {}'.format(ix, pid))
    # print('Image direction: {}'.format(img.GetDirection()))
    # np_image = resample_array(np_image, img.GetSpacing(), cf.target_spacing)
    np_image = intensity_normalization(np_image)

    # roi processing
    final_rois = np.zeros_like(np_image, dtype=np.uint8)
    seg_path = os.path.join(cf.root_dir, 'training_data_segmentations')
    lesion_paths = ([os.path.join(seg_path, ii) for ii in os.listdir(seg_path) if pid in ii and 'mhd' in ii])

    rix = 1
    for lp in lesion_paths:
        roi = sitk.ReadImage(lp)
        # print('Roi direction: {}'.format(roi.GetDirection()))
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
                final_rois[z:z + a, y:y + b, x:x + c] += np_roi * rix
            else:
                if z < a:
                    final_rois[:z, y:y + b, x:x + c] += np.flipud(np_roi)[-z:, :, :] * rix
                else:
                    final_rois[z - a:z, y:y + b, x:x + c] += np.flipud(np_roi) * rix
        except ValueError:
            print('Roi went out of the image. PID: {}, LesionID: {}'.format(pid, lesion_id))
            print('Image origin: {}, Roi origin {}, spacing: {}'.format(np_origin, np_roi_origin, np_spacing))

        rix += 1

    # stuff for meta info, set malignancy to 4, this is binarized later on to malignent (all are malignent)
    fg_slices = [ii for ii in np.unique(np.argwhere(final_rois != 0)[:, 0])]
    class_targets = [0.] * len(lesion_paths)
    assert len(class_targets) + 1 == len(np.unique(final_rois)), [len(class_targets), np.unique(final_rois), pid]

    # save img and final rois
    np.save(os.path.join(cf.pp_dir, '{}_rois.npy'.format(pid)), final_rois)
    np.save(os.path.join(cf.pp_dir, '{}_img.npy'.format(pid)), np_image)

    with open(os.path.join(cf.pp_dir, 'meta_info_{}.pickle'.format(pid)), 'wb') as handle:
        meta_info_dict = {'pid': pid, 'class_target': class_targets, 'spacing': img.GetSpacing(), 'fg_slices': fg_slices}
        pickle.dump(meta_info_dict, handle)


def aggregate_meta_info(exp_dir):
    files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if 'meta_info' in f]
    df = pd.DataFrame(columns=['pid', 'class_target', 'spacing', 'fg_slices'])
    for f in files:
        with open(f, 'rb') as handle:
            df.loc[len(df)] = pickle.load(handle)

    df.to_pickle(os.path.join(exp_dir, 'info_df.pickle'))
    print("aggregated meta info to df with length", len(df))


if __name__ == "__main__":
    cf = configs.configs()
    parser = argparse.ArgumentParser(description='Preprocessing mhd to numpy.')
    parser.add_argument("--n", type=int, help="number of patients to process")
    args = parser.parse_args()

    if args.n:
        paths = [os.path.join(cf.raw_data_dir, ii) for ii in os.listdir(cf.raw_data_dir) if 'mhd' in ii][:args.n]
    else:
        paths = [os.path.join(cf.raw_data_dir, ii) for ii in os.listdir(cf.raw_data_dir) if 'mhd' in ii]

    if not os.path.exists(cf.pp_dir):
        os.mkdir(cf.pp_dir)

    for p in enumerate(paths):
        pp_patient(p)

    # with ProcessPoolExecutor() as executor:
    #     executor.map(pp_patient, enumerate(paths), chunksize=1)

    # pool = Pool(processes=12)
    # p1 = pool.map(pp_patient, enumerate(paths), chunksize=1)
    # pool.close()
    # pool.join()

    aggregate_meta_info(cf.pp_dir)
    subprocess.call('cp {} {}'.format(os.path.join(cf.pp_dir, 'info_df.pickle'), os.path.join(cf.pp_dir,
                                                                                              'info_df_bk.pickle')),
                    shell=True)

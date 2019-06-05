import os
from glob import glob
import skimage
import imageio
import numpy as np
from skimage.transform import resize

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def benchmark(output_dir, gt_dir, img_size=(128,128), margin=5):
    '''Benchmark the output of a model to ground truth.
    '''
    out_paths = sorted(glob_imgs(output_dir))
    gt_paths = sorted(glob_imgs(gt_dir))

    if len(out_paths) != len(gt_paths):
        print(len(out_paths), len(gt_paths))
        print("Not the same number of images!")
        return None

    l1_ssim_psnr = np.zeros((len(out_paths), 3))
    for i, (out_path, gt_path) in enumerate(zip(out_paths, gt_paths)):
        out_img = imageio.imread(out_path)
        gt_img = imageio.imread(gt_path)
        #print(gt_img.shape)
        gt_img = gt_img[:, 420:-420, :]
        gt_img = resize(gt_img, img_size, anti_aliasing=False)
        gt_img = gt_img[5:-5, 5:-5, :]
        gt_img = skimage.img_as_float32(gt_img)
        out_img = skimage.img_as_float32(out_img)

        l1 = np.mean(np.abs(out_img - gt_img))
        range = np.abs(np.max(gt_img) - np.min(gt_img))
        #print(np.max(gt_img))
        #print(np.min(gt_img))
        #print(np.max(out_img))
        #print(np.min(out_img))
        ssim = skimage.measure.compare_ssim(out_img, gt_img, data_range=range, multichannel=True)
        psnr = skimage.measure.compare_psnr(out_img, gt_img, data_range=range)

        print(str(i) + ": " + str(ssim) + " " + str(psnr))

        l1_ssim_psnr[i, :] = np.array([l1, ssim, psnr])

    # Write into a report file
    #report_path = output_dir.strip('/').split('/')[-1] + '_VS_' + gt_dir.strip('/').split('/')[-1] + '.txt'
    #with open(report_path, 'w') as report_file:
    #    report_file.write("out_name gt_name l1 ssim psnr")

        # Write results for each image
    #    for i in range(99):
      #      item = (os.path.basename(out_paths[i]),
        #            os.path.basename(gt_paths[i]),
          #          l1_ssim_psnr[i, 0],
            #        l1_ssim_psnr[i, 1],
              #      l1_ssim_psnr[i, 2])

           # string = ' '.join(map(str, item)) + '\n'
           # report_file.write(string)

        # Last line is the mean
        #report_file.write('\n')
        #report_file.write(' '.join(map(str, np.mean(l1_ssim_psnr, axis=0).tolist())) + '\n')
        #print(' '.join(map(str, np.mean(l1_ssim_psnr, axis=0).tolist())) + '\n')

    return np.mean(l1_ssim_psnr, axis=0).tolist()

out_dir = "./throwaway/test_traj/06_05/05-28-40_21-24-00_all_200.00_l1_weight_2_trgt__gl_train_model-epoch_13_iter_5000.pth_gl_test"
#out_dir = "./logging/test_traj/06_05/04-00-56_19-28-10_all_200.00_l1_weight_2_trgt__gl_train_model-epoch_13_iter_5000.pth_gl_test"
gt_dir = "./gl_test"
#out_dir = "./logging/test_traj/05_16/04-34-02_02-57-02_all_200.00_l1_weight_2_trgt__vase_model-epoch_4_iter_2200.pth_vase"
#gt_dir = "./test/vase/rgb"

print(benchmark(out_dir, gt_dir))

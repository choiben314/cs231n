import os
from glob import glob
import skimage
import imageio 

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def benchmark(output_dir, gt_dir, img_size=(128,128)):
    '''Benchmark the output of a model to ground truth.
    '''
    out_paths = sorted(data_util.glob_imgs(output_dir))
    gt_paths = sorted(data_util.glob_imgs(gt_dir))

    if len(out_paths) != len(gt_paths):
        print(len(out_paths), len(gt_paths))
        print("Not the same number of images!")
        return None

    l1_ssim_psnr = np.zeros((len(out_paths), 3))
    for i, (out_path, gt_path) in enumerate(zip(out_paths, gt_paths)):
        out_img = imageio.imread(out_path)
        gt_img = imageio.imread(gt_path)

        l1 = np.mean(np.abs(out_img - gt_img))
        ssim = skimage.measure.compare_ssim(out_img, gt_img, multichannel=True)
        psnr = skimage.measure.compare_psnr(out_img, gt_img)

        l1_ssim_psnr[i, :] = np.array([l1, ssim, psnr])

    # Write into a report file
    report_path = output_dir.strip('/').split('/')[-1] + '_VS_' + gt_dir.strip('/').split('/')[-1] + '.txt'
    with open(report_path, 'w') as report_file:
        report_file.write("out_name gt_name l1 ssim psnr")

        # Write results for each image
        for i in range(len(out_paths)):
            item = (os.path.basename(out_paths[i]),
                    os.path.basename(gt_paths[i]),
                    l1_ssim_psnr[i, 0],
                    l1_ssim_psnr[i, 1],
                    l1_ssim_psnr[i, 2])

            string = ' '.join(map(str, item)) + '\n'
            report_file.write(string)

        # Last line is the mean
        report_file.write('\n')
        report_file.write(' '.join(map(str, np.mean(l1_ssim_psnr, axis=0).tolist())) + '\n')
        print(' '.join(map(str, np.mean(l1_ssim_psnr, axis=0).tolist())) + '\n')

    return np.mean(l1_ssim_psnr, axis=0).tolist()

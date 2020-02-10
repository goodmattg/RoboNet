from robonet.datasets.util.hdf5_loader import *
from robonet.datasets.util.misc_utils import override_dict
from dotmap import DotMap

import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":

    import argparse
    import robonet.datasets as datasets
    import random
    import matplotlib.pyplot as plt

    override_hparams = {
        "cams_to_load": [0, 1, 2, 3],
        "target_adim": 5,
        "img_size": [240, 320],
        "color_augmentation": 0.0,
    }

    parser = argparse.ArgumentParser(
        description="tests hdf5 data loader without tensorflow dataset wrapper"
    )
    parser.add_argument("file", type=str, help="path to hdf5 you want to load")

    # TODO: Untested. Test and clean up annotations when download full dataset with annotations.
    parser.add_argument(
        "--load_annotations", action="store_true", help="loads annotations if supplied"
    )
    parser.add_argument(
        "--load_steps",
        type=int,
        default=0,
        help="loads <load_steps> steps from the dataset instead of everything",
    )
    args = parser.parse_args()

    assert "hdf5" in args.file
    data_folder = "/".join(args.file.split("/")[:-1])
    meta_data = datasets.load_metadata(data_folder)

    hparams = DotMap(override_dict(default_loader_hparams(), override_hparams))
    hparams.load_T = args.load_steps

    imgs, actions, states = load_data(
        args.file, meta_data.get_file_metadata(args.file), hparams
    )

    out = np.vstack(
        [
            np.hstack(
                [
                    cv2.resize(imgs[im, vp], (160, 120), interpolation=cv2.INTER_AREA)
                    for vp in range(imgs.shape[1])
                ]
            )
            for im in range(6)
        ]
    )

    cv2.imshow("all", out)
    cv2.waitKey(0)

    #     out.append(np.hstack([matches_img, out_img]))

    # out = np.concatenate(out, axis=0)

    # for i in range(5):
    #     for j in range(4):
    #         fname = "frame{}_view{}.png".format(i, j)
    #         imageio.imwrite(fname, imgs[i, j])

    # INDEX = 8

    # im_src = imgs[10, SRC_VIEWPOINT]
    # im_dst = imgs[10, DST_VIEWPOINT]

    # gray = cv2.cvtColor(im_src, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, 127, 255, 0)
    # # imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # # ret, thresh = cv2.threshold(imgray, 40, 255, 0)
    # im2, contours, hierarchy = cv2.findContours(
    #     thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    # )
    # # cnt = contours[0]
    # # for i in contours:
    # #     area = cv2.contourArea(cnt)
    # #     if area > largest_area:
    # #         largest_area = area
    # #         largest_contour_index = i
    # #         bounding_rect = cv2.boundingRect(contours[i])
    # # rect = im_src(bounding_rect).clone()
    # cv2.drawContours(thresh, contours, -1, (0, 255, 0), 3)
    # cv2.imshow("largest contour ", thresh)

    # # cv2.imshow("Out", out)
    # # cv2.imshow("Keypoint Matches", img_matches)

    # # cv2.imshow("corners", im_src)
    # cv2.waitKey(0)
    # # pdb.set_trace()

    # print("actions", actions.shape)
    # print("states", states.shape)
    # print("images", imgs.shape)

    # for ix in range(imgs.shape[0]):

    #     # pdb.set_trace()

    #     # Higher blur seems to help
    #     im_src = cv2.GaussianBlur(im_src, (11, 11), 0)
    #     im_dst = cv2.GaussianBlur(im_dst, (11, 11), 0)

    #     # cv2.imshow("blah", filtered)
    #     # cv2.waitKey(0)

    #     descriptor = cv2.xfeatures2d.SIFT_create()

    #     gray = cv2.cvtColor(im_src, cv2.COLOR_BGR2GRAY)
    #     corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    #     pdb.set_trace()

    #     (kps_src, features_src) = descriptor.detectAndCompute(im_src, None)
    #     (kps_dst, features_dst) = descriptor.detectAndCompute(im_dst, None)

    #     all_src_pts.append(kps_src)
    #     all_dst_pts.append(kps_dst)

    #     # FLANN_INDEX_KDTREE = 1
    #     # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    #     # search_params = dict(checks=50)
    #     # flann = cv2.FlannBasedMatcher(index_params, search_params)
    #     # matches = flann.knnMatch(features_src, features_dst, k=2)

    #     bf = cv2.BFMatcher()
    #     matches = bf.knnMatch(features_src, features_dst, k=2)
    #     all_matches.append(matches)

    #     matchesMask = [[0, 0] for i in range(len(matches))]
    #     # store all the good matches as per Lowe's ratio test.

    #     # NOTE: I typically have this at 0.7
    #     good = []
    #     for i, (m, n) in enumerate(matches):
    #         if m.distance < 0.70 * n.distance:
    #             matchesMask[i] = [1, 0]
    #             good.append(m)

    #     src_pts = np.float32([kps_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    #     dst_pts = np.float32([kps_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    #     all_good_src_pts.append(src_pts)
    #     all_good_dst_pts.append(dst_pts)

    #     match_masks.append(matchesMask)

    # all_good_src_pts = np.concatenate(all_good_src_pts, axis=0)
    # all_good_dst_pts = np.concatenate(all_good_dst_pts, axis=0)

    # # Calculate Homography
    # h, status = cv2.findHomography(all_good_src_pts, all_good_dst_pts, cv2.RANSAC, 5.0)

    # out = []
    # for iy in np.sort(np.random.randint(0, imgs.shape[0], size=5)):

    #     im_src = imgs[ix, SRC_VIEWPOINT]
    #     im_dst = imgs[ix, DST_VIEWPOINT]
    #     mmask = match_masks[ix]
    #     kps_src = all_src_pts[ix]
    #     kps_dst = all_dst_pts[ix]

    #     draw_params = dict(
    #         matchesMask=mmask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    #     )

    #     matches_img = cv2.resize(
    #         cv2.drawMatchesKnn(
    #             im_src, kps_src, im_dst, kps_dst, all_matches[ix], None, **draw_params
    #         ),
    #         (320, 120),
    #         interpolation=cv2.INTER_AREA,
    #     )

    #     src_img = cv2.resize(im_src, (160, 120), interpolation=cv2.INTER_AREA)
    #     dst_img = cv2.resize(im_dst, (160, 120), interpolation=cv2.INTER_AREA)

    #     out_img = cv2.resize(
    #         cv2.warpPerspective(im_src, h, (320, 240)),
    #         (160, 120),
    #         interpolation=cv2.INTER_AREA,
    #     )

    #     out.append(np.hstack([matches_img, out_img]))

    # out = np.concatenate(out, axis=0)
    # # # Warp source image to destination based on homography
    # # im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))

    # # Display images
    # # cv2.imshow("Source Image", im_src)
    # # cv2.imshow("Destination Image", im_dst)
    # # cv2.imshow("Warped Source Image", im_out)
    # cv2.imshow("Out", out)
    # cv2.imshow("Keypoint Matches", img_matches)

    # cv2.waitKey(0)
    # pdb.set_trace()

    # Helpful to place the cameras one after the other in gif instead of alternating
    # imageio.mimsave(
    #     "out3.gif", np.reshape(np.swapaxes(imgs, 0, 1), (-1, *hparams.img_size, 3))
    # )


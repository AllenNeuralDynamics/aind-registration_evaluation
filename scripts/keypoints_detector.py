"""
Script to test the keypoint
detector and matching
"""

import numpy as np
import tifffile as tif
import zarr
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from aind_registration_evaluation.metric import (
    compute_feature_space_distances, get_pairs_from_distances)
from aind_registration_evaluation.sample import *
from aind_registration_evaluation.util.visualization import plot_matches


def get_quadrant_2d(matrix_size, row, col):
    mid = matrix_size // 2

    if col < mid:
        if row < mid:
            return 0
        else:
            return 1
    else:
        if row < mid:
            return 2
        else:
            return 3


def get_quadrant_3d(matrix_size, depth, row, col):
    mid = matrix_size // 2

    if col < mid:
        if row < mid:
            if depth < mid:
                return 0
            else:
                return 1
        else:
            if depth < mid:
                return 2
            else:
                return 3
    else:
        if row < mid:
            if depth < mid:
                return 4
            else:
                return 5
        else:
            if depth < mid:
                return 6
            else:
                return 7


def test_fft_max_min_keypoints(img_1, img_2):
    pad_width = np.min(img_1.shape) // 4
    (
        max_img_1_keypoints,
        min_img_1_keypoints,
        filtered_img_1,
    ) = kd_fft_keypoints(image=img_1, pad_width=pad_width)
    (
        max_img_2_keypoints,
        min_img_2_keypoints,
        filtered_img_2,
    ) = kd_fft_keypoints(image=img_2, pad_width=pad_width)

    # comparison img1 filters
    f, axarr = plt.subplots(1, 2)
    f.suptitle("Image 1", fontsize=20)
    axarr[0].set_title("Original")
    axarr[0].imshow(img_1)
    axarr[0].plot(
        max_img_1_keypoints[:, 1], max_img_1_keypoints[:, 0], "r."
    )  # max points
    axarr[0].plot(
        min_img_1_keypoints[:, 1], min_img_1_keypoints[:, 0], "b."
    )  # min points

    axarr[1].set_title("Max Filtered")
    axarr[1].imshow(filtered_img_1)
    axarr[1].plot(
        max_img_1_keypoints[:, 1], max_img_1_keypoints[:, 0], "r."
    )  # max points
    axarr[1].plot(
        min_img_1_keypoints[:, 1], min_img_1_keypoints[:, 0], "b."
    )  # max points

    plt.tight_layout()
    plt.show()

    # comparison img2 filters

    f, axarr = plt.subplots(1, 2)

    f.suptitle("Image 2", fontsize=20)
    axarr[0].set_title("Original")
    axarr[0].imshow(img_2)
    axarr[0].plot(
        max_img_2_keypoints[:, 1], max_img_2_keypoints[:, 0], "r."
    )  # max points
    axarr[0].plot(
        min_img_2_keypoints[:, 1], min_img_2_keypoints[:, 0], "b."
    )  # min points

    axarr[1].set_title("Max Filtered")
    axarr[1].imshow(filtered_img_2)
    axarr[1].plot(
        max_img_2_keypoints[:, 1], max_img_2_keypoints[:, 0], "r."
    )  # max points
    axarr[1].plot(
        min_img_2_keypoints[:, 1], min_img_2_keypoints[:, 0], "b."
    )  # max points
    plt.tight_layout()
    plt.show()

    # comparison img1 - img2

    f, axarr = plt.subplots(1, 2)

    f.suptitle("Images", fontsize=20)
    axarr[0].set_title("Img 1")
    axarr[0].imshow(img_1)
    axarr[0].plot(
        max_img_1_keypoints[:, 1], max_img_1_keypoints[:, 0], "r."
    )  # max points
    axarr[0].plot(
        min_img_1_keypoints[:, 1], min_img_1_keypoints[:, 0], "b."
    )  # min points

    axarr[1].set_title("Img 2")
    axarr[1].imshow(img_2)
    axarr[1].plot(
        max_img_2_keypoints[:, 1], max_img_2_keypoints[:, 0], "r."
    )  # max points
    axarr[1].plot(
        min_img_2_keypoints[:, 1], min_img_2_keypoints[:, 0], "b."
    )  # max points
    plt.tight_layout()
    plt.show()

    # comparison filtered1 - filtered2

    f, axarr = plt.subplots(1, 2)

    f.suptitle("FFT-Butterworth maximum filter", fontsize=20)
    axarr[0].set_title("Img 1")
    axarr[0].imshow(filtered_img_1)
    axarr[0].plot(
        max_img_1_keypoints[:, 1], max_img_1_keypoints[:, 0], "r."
    )  # max points
    axarr[0].plot(
        min_img_1_keypoints[:, 1], min_img_1_keypoints[:, 0], "b."
    )  # min points

    axarr[1].set_title("Img 2")
    axarr[1].imshow(filtered_img_2)
    axarr[1].plot(
        max_img_2_keypoints[:, 1], max_img_2_keypoints[:, 0], "r."
    )  # max points
    axarr[1].plot(
        min_img_2_keypoints[:, 1], min_img_2_keypoints[:, 0], "b."
    )  # max points
    plt.tight_layout()
    plt.show()


def generate_key_features_per_img2d(img_2d, n_keypoints, mode="energy"):
    pad_width = np.min(img_2d.shape) // 6

    if mode == "energy":
        img_2d_keypoints_energy, img_response = kd_fft_energy_keypoints(
            image=img_2d,
            pad_width=pad_width,
            n_keypoints=n_keypoints,
        )
    else:
        # maximum
        img_2d_keypoints_energy, img_response = kd_fft_keypoints(
            image=img_2d,
            pad_width=pad_width,
            n_keypoints=n_keypoints,
        )

    dy_val, dx_val = derivate_image_axis(
        gaussian_filter(img_2d, sigma=8), [0, 1]
    )

    # img_2d_dy = np.zeros(img_2d.shape, dtype=img_2d.dtype)
    # img_2d_dx = np.zeros(img_2d.shape, dtype=img_2d.dtype)

    # dy_val = np.sqrt(dy_val)
    # dx_val = np.sqrt(dx_val)

    # f, ax = plt.subplots(1,2)

    # ax[0].imshow(dy_val, cmap="gray")#vmin=0, vmax=0.2)
    # ax[1].imshow(dx_val, cmap="gray") #vmin=0, vmax=0.2)
    # plt.show()

    # img_2d_dy[:-1, :] = np.float32(dy_val)
    # img_2d_dx[:, :-1] = np.float32(dx_val)

    (
        gradient_magnitude,
        gradient_orientation,
        gradient_orientation_polar,
    ) = kd_gradient_magnitudes_and_orientations(
        derivated_images=[dy_val, dx_val]  # [img_2d_dy, img_2d_dx]
    )

    img_keypoints_features = [
        kd_compute_keypoints_hog(
            image_gradient_magnitude=gradient_magnitude,
            image_gradient_orientation=[gradient_orientation],
            keypoint=keypoint,
            n_dims=2,
            window_size=16,
            bins=[8],
        )
        for keypoint in img_2d_keypoints_energy
    ]

    keypoints = []
    features = []
    for key_feat in img_keypoints_features:
        # print(f"Keypoint {key_feat['keypoint']} feat shape: {key_feat['feature_vector'].shape}")
        keypoints.append(key_feat["keypoint"])
        features.append(key_feat["feature_vector"])

    return {
        "keypoints": np.array(keypoints),
        "features": np.array(features),
        "response_img": img_response,
    }


def test_fft_max_keypoints(img_1, img_2):
    n_keypoints = 200
    img_1_dict = generate_key_features_per_img2d(
        img_1, n_keypoints=n_keypoints, mode="max"
    )
    img_2_dict = generate_key_features_per_img2d(
        img_2, n_keypoints=n_keypoints, mode="max"
    )

    feature_vector_img_1 = (img_1_dict["features"], img_1_dict["keypoints"])
    feature_vector_img_2 = (img_2_dict["features"], img_2_dict["keypoints"])

    distances = compute_feature_space_distances(
        feature_vector_img_1, feature_vector_img_2, feature_weight=0.3
    )

    point_matches_pruned = get_pairs_from_distances(
        distances=distances, delete_points=True
    )

    point_matches_not_pruned = get_pairs_from_distances(
        distances=distances, delete_points=False
    )

    print(
        f"N keypoints img_1: {img_1_dict['keypoints'].shape} img_2: {img_2_dict['keypoints'].shape}"
    )

    # Showing only points
    # comparison img1 filters
    print("\n Keypoint confidence img 1")
    for key_idx in range(len(img_1_dict["keypoints"])):
        print(
            f"Confidence for point {key_idx} is: ",
            img_1_dict["response_img"][
                img_1_dict["keypoints"][key_idx][0],
                img_1_dict["keypoints"][key_idx][1],
            ],
        )

    print("\n Keypoint confidence img 2")
    for key_idx in range(len(img_2_dict["keypoints"])):
        print(
            f"Confidence for point {key_idx} is: ",
            img_2_dict["response_img"][
                img_2_dict["keypoints"][key_idx][0],
                img_2_dict["keypoints"][key_idx][1],
            ],
        )

    f, axarr = plt.subplots(1, 2)
    f.suptitle("Image 1", fontsize=20)
    axarr[0].set_title("Original")
    axarr[0].imshow(img_1)
    axarr[0].plot(
        img_1_dict["keypoints"][:, 1], img_1_dict["keypoints"][:, 0], "r."
    )  # max points

    axarr[1].set_title("FFT-Butterworth Max Filtered")
    axarr[1].imshow(img_1_dict["response_img"])
    axarr[1].plot(
        img_1_dict["keypoints"][:, 1], img_1_dict["keypoints"][:, 0], "r."
    )  # max points

    plt.tight_layout()
    plt.show()
    # comparison img2 filters

    f, axarr = plt.subplots(1, 2)

    f.suptitle("Image 2", fontsize=20)
    axarr[0].set_title("Original")
    axarr[0].imshow(img_2)
    axarr[0].plot(
        img_2_dict["keypoints"][:, 1], img_2_dict["keypoints"][:, 0], "r."
    )  # max points

    axarr[1].set_title("FFT-Butterworth Max Filtered")
    axarr[1].imshow(img_2_dict["response_img"])
    axarr[1].plot(
        img_2_dict["keypoints"][:, 1], img_2_dict["keypoints"][:, 0], "r."
    )  # max points
    plt.tight_layout()
    plt.show()

    # comparison img1 - img2
    f, axarr = plt.subplots(1, 2)

    f.suptitle("FFT-Butterworth Max Filtered", fontsize=20)
    axarr[0].set_title("Image 1")
    axarr[0].imshow(img_1_dict["response_img"])
    axarr[0].plot(
        img_1_dict["keypoints"][:, 1], img_1_dict["keypoints"][:, 0], "r."
    )  # max points

    axarr[1].set_title("Image 2")
    axarr[1].imshow(img_2_dict["response_img"])
    axarr[1].plot(
        img_2_dict["keypoints"][:, 1], img_2_dict["keypoints"][:, 0], "r."
    )  # max points
    plt.tight_layout()
    plt.show()

    f, axarr = plt.subplots(1, 2)

    f.suptitle("Points in images", fontsize=20)
    axarr[0].set_title("Img 1")
    axarr[0].imshow(img_1)
    axarr[0].plot(
        img_1_dict["keypoints"][:, 1], img_1_dict["keypoints"][:, 0], "r."
    )  # max points

    axarr[1].set_title("Img 2")
    axarr[1].imshow(img_2)
    axarr[1].plot(
        img_2_dict["keypoints"][:, 1], img_2_dict["keypoints"][:, 0], "r."
    )  # max points
    plt.tight_layout()
    plt.show()

    f, axarr = plt.subplots(1, 2, figsize=(10, 5))
    f.suptitle("Matched points", fontsize=20)
    # Set titles and labels
    axarr[0].set_xlabel("X")
    axarr[0].set_ylabel("Y")
    axarr[0].set_title("1-N Match")

    axarr[1].set_title("1-1 Match")
    axarr[1].set_xlabel("X")
    axarr[1].set_ylabel("Y")

    idxs1, idxs2 = list(point_matches_not_pruned.keys()), list(
        point_matches_not_pruned.values()
    )
    matches_not_pruned = np.column_stack((idxs1, idxs2))

    idxs1, idxs2 = list(point_matches_pruned.keys()), list(
        point_matches_pruned.values()
    )
    matches_pruned = np.column_stack((idxs1, idxs2))

    plot_matches(
        ax=axarr[0],
        image1=img_1,
        image2=img_2,
        keypoints1=img_1_dict["keypoints"],
        keypoints2=img_2_dict["keypoints"],
        keypoints_color="red",
        matches=matches_not_pruned,
        matches_color="red",
        only_matches=False,
    )

    plot_matches(
        ax=axarr[1],
        image1=img_1,
        image2=img_2,
        keypoints1=img_1_dict["keypoints"],
        keypoints2=img_2_dict["keypoints"],
        keypoints_color="red",
        matches=matches_pruned,
        matches_color="red",
        only_matches=True,
    )

    plt.tight_layout()

    # Show the plot
    plt.show()

    f, axarr = plt.subplots(1, 2, figsize=(10, 5))
    f.suptitle("Matched points", fontsize=20)
    # Set titles and labels
    axarr[0].set_xlabel("X")
    axarr[0].set_ylabel("Y")
    axarr[0].set_title("1-N Match")

    axarr[1].set_title("1-1 Match")
    axarr[1].set_xlabel("X")
    axarr[1].set_ylabel("Y")

    idxs1, idxs2 = list(point_matches_not_pruned.keys()), list(
        point_matches_not_pruned.values()
    )
    matches_not_pruned = np.column_stack((idxs1, idxs2))

    idxs1, idxs2 = list(point_matches_pruned.keys()), list(
        point_matches_pruned.values()
    )
    matches_pruned = np.column_stack((idxs1, idxs2))

    plot_matches(
        ax=axarr[0],
        image1=img_1_dict["response_img"],
        image2=img_2_dict["response_img"],
        keypoints1=img_1_dict["keypoints"],
        keypoints2=img_2_dict["keypoints"],
        keypoints_color="red",
        matches=matches_not_pruned,
        matches_color="red",
        only_matches=False,
    )

    plot_matches(
        ax=axarr[1],
        image1=img_1_dict["response_img"],
        image2=img_2_dict["response_img"],
        keypoints1=img_1_dict["keypoints"],
        keypoints2=img_2_dict["keypoints"],
        keypoints_color="red",
        matches=matches_pruned,
        matches_color="red",
        only_matches=True,
    )

    plt.tight_layout()

    # Show the plot
    plt.show()


def test_fft_energy_keypoints(img_1, img_2):
    n_keypoints = 200
    img_1_dict = generate_key_features_per_img2d(
        img_1, n_keypoints=n_keypoints
    )
    img_2_dict = generate_key_features_per_img2d(
        img_2, n_keypoints=n_keypoints
    )

    feature_vector_img_1 = (img_1_dict["features"], img_1_dict["keypoints"])
    feature_vector_img_2 = (img_2_dict["features"], img_2_dict["keypoints"])

    distances = compute_feature_space_distances(
        feature_vector_img_1, feature_vector_img_2, feature_weight=0.2
    )

    point_matches_pruned = get_pairs_from_distances(
        distances=distances, delete_points=True
    )

    point_matches_not_pruned = get_pairs_from_distances(
        distances=distances, delete_points=False
    )

    print(
        f"N keypoints img_1: {img_1_dict['keypoints'].shape} img_2: {img_2_dict['keypoints'].shape}"
    )

    # Showing only points
    # comparison img1 filters
    print("\n Keypoint confidence img 1")
    for key_idx in range(len(img_1_dict["keypoints"])):
        print(
            f"Confidence for point {key_idx} is: ",
            img_1_dict["response_img"][
                img_1_dict["keypoints"][key_idx][0],
                img_1_dict["keypoints"][key_idx][1],
            ],
        )

    print("\n Keypoint confidence img 2")
    for key_idx in range(len(img_2_dict["keypoints"])):
        print(
            f"Confidence for point {key_idx} is: ",
            img_2_dict["response_img"][
                img_2_dict["keypoints"][key_idx][0],
                img_2_dict["keypoints"][key_idx][1],
            ],
        )

    f, axarr = plt.subplots(1, 2)
    f.suptitle("Image 1", fontsize=20)
    axarr[0].set_title("Original")
    axarr[0].imshow(img_1)
    axarr[0].plot(
        img_1_dict["keypoints"][:, 1], img_1_dict["keypoints"][:, 0], "r."
    )  # max points

    axarr[1].set_title("FFT-Butterworth Gauss Laplaced Filtered")
    axarr[1].imshow(img_1_dict["response_img"])
    axarr[1].plot(
        img_1_dict["keypoints"][:, 1], img_1_dict["keypoints"][:, 0], "r."
    )  # max points

    plt.tight_layout()
    plt.show()
    # comparison img2 filters

    f, axarr = plt.subplots(1, 2)

    f.suptitle("Image 2", fontsize=20)
    axarr[0].set_title("Original")
    axarr[0].imshow(img_2)
    axarr[0].plot(
        img_2_dict["keypoints"][:, 1], img_2_dict["keypoints"][:, 0], "r."
    )  # max points

    axarr[1].set_title("FFT-Butterworth Gauss Laplaced Filtered")
    axarr[1].imshow(img_2_dict["response_img"])
    axarr[1].plot(
        img_2_dict["keypoints"][:, 1], img_2_dict["keypoints"][:, 0], "r."
    )  # max points
    plt.tight_layout()
    plt.show()

    # comparison img1 - img2
    f, axarr = plt.subplots(1, 2)

    f.suptitle("FFT-Butterworth Gaussian Laplaced Filtered", fontsize=20)
    axarr[0].set_title("Image 1")
    axarr[0].imshow(img_1_dict["response_img"])
    axarr[0].plot(
        img_1_dict["keypoints"][:, 1], img_1_dict["keypoints"][:, 0], "r."
    )  # max points

    axarr[1].set_title("Image 2")
    axarr[1].imshow(img_2_dict["response_img"])
    axarr[1].plot(
        img_2_dict["keypoints"][:, 1], img_2_dict["keypoints"][:, 0], "r."
    )  # max points
    plt.tight_layout()
    plt.show()

    f, axarr = plt.subplots(1, 2)

    f.suptitle("Points in images", fontsize=20)
    axarr[0].set_title("Img 1")
    axarr[0].imshow(img_1)
    axarr[0].plot(
        img_1_dict["keypoints"][:, 1], img_1_dict["keypoints"][:, 0], "r."
    )  # max points

    axarr[1].set_title("Img 2")
    axarr[1].imshow(img_2)
    axarr[1].plot(
        img_2_dict["keypoints"][:, 1], img_2_dict["keypoints"][:, 0], "r."
    )  # max points
    plt.tight_layout()
    plt.show()

    f, axarr = plt.subplots(1, 2, figsize=(10, 5))
    f.suptitle("Matched points", fontsize=20)
    # Set titles and labels
    axarr[0].set_xlabel("X")
    axarr[0].set_ylabel("Y")
    axarr[0].set_title("1-N Match")

    axarr[1].set_title("1-1 Match")
    axarr[1].set_xlabel("X")
    axarr[1].set_ylabel("Y")

    idxs1, idxs2 = list(point_matches_not_pruned.keys()), list(
        point_matches_not_pruned.values()
    )
    matches_not_pruned = np.column_stack((idxs1, idxs2))

    idxs1, idxs2 = list(point_matches_pruned.keys()), list(
        point_matches_pruned.values()
    )
    matches_pruned = np.column_stack((idxs1, idxs2))

    plot_matches(
        ax=axarr[0],
        image1=img_1,
        image2=img_2,
        keypoints1=img_1_dict["keypoints"],
        keypoints2=img_2_dict["keypoints"],
        keypoints_color="red",
        matches=matches_not_pruned,
        matches_color="red",
        only_matches=False,
    )

    plot_matches(
        ax=axarr[1],
        image1=img_1,
        image2=img_2,
        keypoints1=img_1_dict["keypoints"],
        keypoints2=img_2_dict["keypoints"],
        keypoints_color="red",
        matches=matches_pruned,
        matches_color="red",
        only_matches=True,
    )

    plt.tight_layout()

    # Show the plot
    plt.show()

    f, axarr = plt.subplots(1, 2, figsize=(10, 5))
    f.suptitle("Matched points", fontsize=20)
    # Set titles and labels
    axarr[0].set_xlabel("X")
    axarr[0].set_ylabel("Y")
    axarr[0].set_title("1-N Match")

    axarr[1].set_title("1-1 Match")
    axarr[1].set_xlabel("X")
    axarr[1].set_ylabel("Y")

    idxs1, idxs2 = list(point_matches_not_pruned.keys()), list(
        point_matches_not_pruned.values()
    )
    matches_not_pruned = np.column_stack((idxs1, idxs2))

    idxs1, idxs2 = list(point_matches_pruned.keys()), list(
        point_matches_pruned.values()
    )
    matches_pruned = np.column_stack((idxs1, idxs2))

    plot_matches(
        ax=axarr[0],
        image1=img_1_dict["response_img"],
        image2=img_2_dict["response_img"],
        keypoints1=img_1_dict["keypoints"],
        keypoints2=img_2_dict["keypoints"],
        keypoints_color="red",
        matches=matches_not_pruned,
        matches_color="red",
        only_matches=False,
    )

    plot_matches(
        ax=axarr[1],
        image1=img_1_dict["response_img"],
        image2=img_2_dict["response_img"],
        keypoints1=img_1_dict["keypoints"],
        keypoints2=img_2_dict["keypoints"],
        keypoints_color="red",
        matches=matches_pruned,
        matches_color="red",
        only_matches=True,
    )

    plt.tight_layout()

    # Show the plot
    plt.show()


def test_img_3d_orientations(img_3d):
    pad_width_3d = np.min(img_3d.shape) // 4
    # Getting keypoints
    img_3d_keypoints_energy, response_image = kd_fft_energy_keypoints(
        image=img_3d,
        pad_width=pad_width_3d,
    )

    f, ax = plt.subplots(1, 2)
    f.suptitle("Original image vs response image")

    slide = 0
    ax[0].imshow(img_3d[slide, :, :])
    ax[1].imshow(response_image[slide, :, :])
    plt.show()

    # Getting derivatives in each axis
    dz_val, dy_val, dx_val = derivate_image_axis(img_3d, [0, 1, 2])

    # Getting image derivatives
    filtered_img_3d_dz = np.zeros(img_3d.shape, dtype=img_3d.dtype)
    filtered_img_3d_dy = np.zeros(img_3d.shape, dtype=img_3d.dtype)
    filtered_img_3d_dx = np.zeros(img_3d.shape, dtype=img_3d.dtype)

    filtered_img_3d_dz[:-1, :, :] = np.float32(dz_val)
    filtered_img_3d_dy[:, :-1, :] = np.float32(dy_val)
    filtered_img_3d_dx[:, :, :-1] = np.float32(dx_val)

    # Getting gradient magnitude, and orientations in YX plane and polar coordinates
    (
        gradient_magnitude_3d,
        gradient_orientation_yx_3d,
        gradient_orientation_z_yx_3d,
    ) = kd_gradient_magnitudes_and_orientations(
        derivated_images=[
            filtered_img_3d_dz,
            filtered_img_3d_dy,
            filtered_img_3d_dx,
        ]
    )

    img_3d_keypoint_energy_orientations = []
    for keypoint in img_3d_keypoints_energy:
        print(f"Computing hog for keypoint {keypoint}")
        img_3d_res = kd_compute_keypoints_hog(
            image_gradient_magnitude=gradient_magnitude_3d,
            image_gradient_orientation=[
                gradient_orientation_z_yx_3d,
                gradient_orientation_yx_3d,
            ],
            keypoint=keypoint,
            n_dims=3,
            window_size=16,
            bins=[8, 4],
        )
        img_3d_keypoint_energy_orientations.append(img_3d_res)

    for img_3d_ori in img_3d_keypoint_energy_orientations:
        print("orientations: ", img_3d_ori)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Set plot labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Dominant 3D Point Orientation")

    # Plot the dominant orientation as a line
    for keypoints_ori_dict in img_3d_keypoint_energy_orientations:
        keypoint = keypoints_ori_dict["keypoint"]
        orientation = keypoints_ori_dict["dominant_orientation"]
        x, y, z = keypoint[2], keypoint[1], keypoint[0]

        ax.scatter(zs=z, ys=y, xs=x, c="blue", marker="o", s=50)

        angle_rad = np.radians(orientation)
        length = 15
        z2 = z + length * np.cos(angle_rad[0])
        x2 = x + length * np.sin(angle_rad[1])
        y2 = y + length * np.cos(angle_rad[1])

        ax.plot(zs=[z, z2], ys=[y, y2], xs=[x, x2], c="r", linestyle="-")

        ax.plot(x, y, z, color="red", linewidth=2)

    # Show the 3D plot
    plt.show()


def main():
    BASE_PATH = "/Users/camilo.laiton/Documents/images/"

    img_1_path = BASE_PATH + "Ex_488_Em_525_468770_468770_830620_012820.zarr/0"
    img_2_path = BASE_PATH + "Ex_488_Em_525_501170_501170_830620_012820.zarr/0"

    img_3D_path = BASE_PATH + "block_10.tif"

    img_1 = zarr.open(img_1_path, "r")[0, 0, 0, :, 1800:]
    img_2 = zarr.open(img_2_path, "r")[0, 0, 0, :, :200]
    img_3d = tif.imread(img_3D_path)[120:184, 200:456, 200:456]
    print("3D image shape: ", img_3d.shape)

    # test_img_2d_orientations(img_3d[10, :, :])# (img_1)#
    # test_img_3d_orientations(img_3d)
    test_fft_energy_keypoints(img_1, img_2)
    # test_fft_max_keypoints(img_1, img_2)


if __name__ == "__main__":
    main()

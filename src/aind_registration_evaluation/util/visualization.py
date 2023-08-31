"""
Module to plot visualizations
"""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from aind_registration_evaluation._shared.types import ArrayLike


class SliceTracker:
    """
    Slice Tracker class to visualize 3D images
    """

    def __init__(
        self,
        fig_axes,
        image_1_data,
        image_2_data,
        points,
        selected_ponts,
        # rectangles,
        vmin=0,
        vmax=200,
        alpha=0.7,
        color="bone",
    ) -> None:
        """
        Class constructor
        """
        self.axes = fig_axes
        self.image_1_data = image_1_data
        self.image_2_data = image_2_data
        self.vmin = vmin
        self.vmax = vmax
        self.alpha = alpha
        self.color = color
        self.points = points
        self.selected_points = selected_ponts
        self.points_1 = self.points_2 = None
        self.plot_image_1 = self.plot_image_2 = None

        # Setting up message
        self.axes.set_title("Scroll to move to slices")

        self.__slices, self.__rows, self.__cols = image_1_data.shape

        # Starting visualization in the middle of the 3D block
        self.idx = self.__slices // 2
        self.get_current_slice()
        self.update_slice()

    def search_points_in_slice(self):
        """
        Searches for points in a slice
        """

        def search_points(points: List):
            """
            Helper function to search points
            in a slice

            Parameters
            -----------
            points: List[List[int]]
                Points in 3D space where
                the 3 lists are in order
                Z Y X
            """

            points_x = []
            points_y = []

            for z_point_pos in range(len(points[0])):
                if self.idx == points[0][z_point_pos]:
                    points_y.append(points[1][z_point_pos])
                    points_x.append(points[2][z_point_pos])

            return points_x, points_y

        points_x, points_y = search_points(self.points)
        sel_points_x, sel_points_y = search_points(self.selected_points)

        return [points_y, points_x], [sel_points_y, sel_points_x]

    def on_scroll(self, event):
        """
        Event to scroll in the image volume
        """
        if event.button == "up":
            self.idx = (self.idx + 1) % self.__slices
        else:
            self.idx = (self.idx - 1) % self.__slices
        self.update_slice()

    def update_slice(self):
        """
        Updates the current slice based on
        the scroll
        """

        self.update_points()
        self.get_current_slice()

        self.axes.set_ylabel("Slice %s" % self.idx)
        self.axes.figure.canvas.draw()

    def update_points(self):
        """
        Updates the points in a slice based on
        the scroll
        """
        pts, sl_pts = self.search_points_in_slice()

        if self.points_1 and self.points_2:
            self.points_1.remove()
            self.points_2.remove()

        self.points_1 = self.axes.scatter(y=pts[0], x=pts[1], c="r", s=20)
        self.points_2 = self.axes.scatter(
            y=sl_pts[0], x=sl_pts[1], c="#5DD9A7", s=20
        )

    def get_current_slice(self):
        """
        Gets the current slice and shows it
        in the renderer
        """
        if self.plot_image_1 and self.plot_image_2:
            self.plot_image_1.remove()
            self.plot_image_2.remove()

        self.plot_image_1 = self.axes.imshow(
            self.image_1_data[self.idx, :, :],
            alpha=self.alpha,
            cmap=self.color,
            vmin=self.vmin,
            vmax=self.vmax,
        )

        self.plot_image_2 = self.axes.imshow(
            self.image_2_data[self.idx, :, :],
            alpha=self.alpha,
            cmap=self.color,
            vmin=self.vmin,
            vmax=self.vmax,
        )


def visualize_images(
    image_1_data: ArrayLike,
    image_2_data: ArrayLike,
    bounds: List[np.ndarray],
    pruned_points: ArrayLike,
    selected_pruned_points: ArrayLike,
    transform: np.matrix,
    metric_name: str,
) -> None:
    """
    Function that plots an image to help visualize the
    intersection area, sampled points and images.

    Parameters
    ------------------------

    image_1_data: ArrayLike
        Array with the data of the image 1.

    image_2_data: ArrayLike
        Array with the data of the image 1.

    bounds: List[np.ndarray]
        List with the boundaries for the intersection area of the images.

    pruned_points: ArrayLike
        Points inside the intersection area that fit based on a window size.

    selected_pruned_points: ArrayLike
        Points inside the intersection area that were used for the metric.

    transform: np.matrix
        Transformation matrix applied to the images

    metric_name: str
        Metric name for the set of points
    """

    def generate_new_boundaries():
        """
        Generates new boundaries to plot
        two images in the same coordinate
        system given an image transformation
        """
        lower_bound_1, upper_bound_1 = None, None
        lower_bound_2, upper_bound_2 = None, None

        if ty >= 0:
            lower_bound_1 = [bounds_1[0][0]]
            upper_bound_1 = [bounds_1[1][0]]

            lower_bound_2 = [bounds_2[0][0]]
            upper_bound_2 = [bounds_2[1][0]]

        else:
            lower_bound_1 = [abs(ty)]
            upper_bound_1 = [bounds_2[1][0] + abs(ty) * 2]

            lower_bound_2 = [bounds_1[0][0]]
            upper_bound_2 = [bounds_1[1][0]]

        if tx >= 0:
            lower_bound_1.append(bounds_1[0][1])
            upper_bound_1.append(bounds_1[1][1])

            lower_bound_2.append(bounds_2[0][1])
            upper_bound_2.append(bounds_2[1][1])

        else:
            lower_bound_1 = [bounds_2[0][1]]
            upper_bound_1 = [bounds_2[1][1]]

            lower_bound_2 = [bounds_1[0][1]]
            upper_bound_2 = [bounds_1[1][1]]

        img_1_bounds = [lower_bound_1, upper_bound_1]
        img_2_bounds = [lower_bound_2, upper_bound_2]

        return img_1_bounds, img_2_bounds

    if image_1_data.ndim != image_2_data.ndim:
        raise ValueError("Images should have the same shape")

    if image_1_data.ndim > 3:
        raise ValueError("Only 2D/3D images are supported")

    bounds_1 = bounds[0]
    bounds_2 = bounds[1]
    # from scipy.ndimage import rotate

    if image_1_data.ndim == 2:
        # plot directly the images and grid

        # Getting max boundaries in X and Y
        ty = transform[0, -1]
        tx = transform[1, -1]

        size_y = image_1_data.shape[0] + abs(ty)
        size_x = image_1_data.shape[1] + abs(tx)

        # size_x = max(bounds_1[1][0], bounds_2[1][0])
        # size_y = max(bounds_1[1][1], bounds_2[1][1])

        img_1_bounds, img_2_bounds = generate_new_boundaries()

        # Image within same coordinate system
        adjusted_img_1 = np.ones((size_y, size_x)) * 255
        adjusted_img_2 = np.ones((size_y, size_x)) * 255

        # flake8: noqa: E203
        # Getting data from the images to common coordinate image
        points_orig_img = (
            np.array(
                [np.arange(0, image_1_data.shape[0], step=1, dtype=np.intp)]
            ),
            np.array(
                [np.arange(0, image_1_data.shape[1], step=1, dtype=np.intp)]
            ),
        )

        points_img_1 = (
            np.array(
                [
                    np.arange(
                        img_1_bounds[0][0],
                        img_1_bounds[1][0],
                        step=1,
                        dtype=np.intp,
                    )
                ]
            ),
            np.array(
                [
                    np.arange(
                        img_1_bounds[0][1],
                        img_1_bounds[1][1],
                        step=1,
                        dtype=np.intp,
                    )
                ]
            ),
        )

        points_img_2 = (
            np.array(
                [
                    np.arange(
                        img_2_bounds[0][0],
                        img_2_bounds[1][0],
                        step=1,
                        dtype=np.intp,
                    )
                ]
            ),
            np.array(
                [
                    np.arange(
                        img_2_bounds[0][1],
                        img_2_bounds[1][1],
                        step=1,
                        dtype=np.intp,
                    )
                ]
            ),
        )

        xv_img_1, yv_img_1 = np.meshgrid(*points_img_1, indexing="ij")
        xv_img_2, yv_img_2 = np.meshgrid(*points_img_2, indexing="ij")
        xv_orig, yv_orig = np.meshgrid(*points_orig_img, indexing="ij")

        # vindex takes some time when processing with dask
        adjusted_img_1[xv_img_1, yv_img_1] = (
            image_1_data[xv_orig, yv_orig]
            if isinstance(image_1_data, np.ndarray)
            else image_1_data.vindex[xv_orig, yv_orig]
        )
        adjusted_img_2[xv_img_2, yv_img_2] = (
            image_2_data[xv_orig, yv_orig]
            if isinstance(image_2_data, np.ndarray)
            else image_2_data.vindex[xv_orig, yv_orig]
        )

        fig, ax = plt.subplots()

        fig.suptitle(f"Metric: {metric_name}", fontsize=16)

        # Setting X,Y positions of points within the grid
        y_points = [point[0] for point in pruned_points]
        x_points = [point[1] for point in pruned_points]

        # Setting X,Y positions of points within the
        # grid that were used to metric estimation
        selected_y_points = [point[0] for point in selected_pruned_points]
        selected_x_points = [point[1] for point in selected_pruned_points]

        # Scattering points within image in common coordinate system
        plt.scatter(x=x_points, y=y_points, c="r", s=10)
        plt.scatter(x=selected_x_points, y=selected_y_points, c="g", s=10)

        # Rectangles to divide images
        rectangle_image_1 = Rectangle(
            xy=(img_1_bounds[0][1], img_1_bounds[0][0]),
            width=img_1_bounds[1][1],
            height=img_1_bounds[1][0],
            linewidth=1,
            edgecolor="#FF2D00",
            facecolor="none",
            linestyle=":",
        )

        rectangle_image_2 = Rectangle(
            xy=(img_2_bounds[0][1], img_2_bounds[0][0]),
            width=img_1_bounds[1][1],
            height=img_1_bounds[1][0],
            linewidth=1,
            edgecolor="#13FF00",
            facecolor="none",
            linestyle="--",
        )

        # Alpha for overlaying
        alpha = 0.5
        vmin_1, vmax_1 = np.percentile(image_1_data.flatten(), (0.2, 99))
        vmin_2, vmax_2 = np.percentile(image_2_data.flatten(), (0.2, 99))
        ax.imshow(
            adjusted_img_1, alpha=alpha, vmin=vmin_1, vmax=vmax_1, cmap="Blues"
        )
        ax.imshow(
            adjusted_img_2,
            alpha=alpha,
            vmin=vmin_2,
            vmax=vmax_2,
            cmap="Oranges",
        )

        ax.add_patch(rectangle_image_1)
        ax.add_patch(rectangle_image_2)

        plt.show()

    else:
        # 3D image, find a way to render 3D images with 3D grid

        size_x = max(bounds_1[1][0], bounds_2[1][0])
        size_y = max(bounds_1[1][1], bounds_2[1][1])
        size_z = max(bounds_1[1][2], bounds_2[1][2])

        # Image within same coordinate system
        adjusted_img_1 = np.ones((size_x, size_y, size_z)) * 255
        adjusted_img_2 = np.ones((size_x, size_y, size_z)) * 255

        # Getting data from the images to common coordinate image
        adjusted_img_1[
            : bounds_1[1][0], : bounds_1[1][1], : bounds_1[1][2]
        ] = image_1_data
        adjusted_img_2[
            bounds_2[0][0] :, bounds_2[0][1] :, bounds_2[0][2] :
        ] = image_2_data

        # Setting Z, Y, X positions of points within the grid
        z_points = [point[0] for point in pruned_points]
        y_points = [point[1] for point in pruned_points]
        x_points = [point[2] for point in pruned_points]

        # Setting Z,Y,X positions of points within the grid
        # that were used to metric estimation
        selected_z_points = [point[0] for point in selected_pruned_points]
        selected_y_points = [point[1] for point in selected_pruned_points]
        selected_x_points = [point[2] for point in selected_pruned_points]

        points = [z_points, y_points, x_points]
        selected_points = [
            selected_z_points,
            selected_y_points,
            selected_x_points,
        ]

        # visualize_image_3D(combined_volume, points, selected_points)

        # Rectangles to divide images
        rectangle_image_1 = Rectangle(
            xy=(bounds_1[0][2], bounds_1[0][1]),
            width=bounds_1[1][2],
            height=bounds_1[1][1],
            linewidth=1,
            edgecolor="#FF2D00",
            facecolor="none",
            linestyle=":",
        )

        rectangle_image_2 = Rectangle(
            xy=(bounds_2[0][2], bounds_2[0][1]),
            width=bounds_1[1][2],
            height=bounds_1[1][1],
            linewidth=1,
            edgecolor="#13FF00",
            facecolor="none",
            linestyle="--",
        )

        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.suptitle(f"Metric: {metric_name}", fontsize=16)

        tracker = SliceTracker(
            fig_axes=ax,
            image_1_data=adjusted_img_1,
            image_2_data=adjusted_img_2,
            points=points,
            selected_ponts=selected_points,
            # rectangles=[rectangle_image_1, rectangle_image_2]
        )

        ax.add_patch(rectangle_image_1)
        ax.add_patch(rectangle_image_2)
        fig.canvas.mpl_connect("scroll_event", tracker.on_scroll)
        fig.tight_layout()
        plt.show()


def visualize_misalignment_images(
    image_1_data: ArrayLike,
    image_2_data: ArrayLike,
    bounds: List[np.ndarray],
    keypoints_img_1: ArrayLike,
    keypoints_img_2: ArrayLike,
    transform: np.matrix,
    metric_name: str,
) -> None:
    """
    Function that plots an image to help visualize the
    intersection area, sampled points and images.

    Parameters
    ------------------------

    image_1_data: ArrayLike
        Array with the data of the image 1.

    image_2_data: ArrayLike
        Array with the data of the image 1.

    bounds: List[np.ndarray]
        List with the boundaries for the intersection area of the images.

    pruned_points: ArrayLike
        Points inside the intersection area that fit based on a window size.

    selected_pruned_points: ArrayLike
        Points inside the intersection area that were used for the metric.

    transform: np.matrix
        Transformation matrix applied to the images

    metric_name: str
        Metric name for the set of points
    """

    def generate_new_boundaries():
        """
        Generates new boundaries to plot
        two images in the same coordinate
        system given an image transformation
        """
        lower_bound_1, upper_bound_1 = None, None
        lower_bound_2, upper_bound_2 = None, None

        if ty >= 0:
            lower_bound_1 = [bounds_1[0][0]]
            upper_bound_1 = [bounds_1[1][0]]

            lower_bound_2 = [bounds_2[0][0]]
            upper_bound_2 = [bounds_2[1][0]]

        else:
            lower_bound_1 = [abs(ty)]
            upper_bound_1 = [bounds_2[1][0] + abs(ty) * 2]

            lower_bound_2 = [bounds_1[0][0]]
            upper_bound_2 = [bounds_1[1][0]]

        if tx >= 0:
            lower_bound_1.append(bounds_1[0][1])
            upper_bound_1.append(bounds_1[1][1])

            lower_bound_2.append(bounds_2[0][1])
            upper_bound_2.append(bounds_2[1][1])

        else:
            lower_bound_1 = [bounds_2[0][1]]
            upper_bound_1 = [bounds_2[1][1]]

            lower_bound_2 = [bounds_1[0][1]]
            upper_bound_2 = [bounds_1[1][1]]

        img_1_bounds = [lower_bound_1, upper_bound_1]
        img_2_bounds = [lower_bound_2, upper_bound_2]

        return img_1_bounds, img_2_bounds

    if image_1_data.ndim != image_2_data.ndim:
        raise ValueError("Images should have the same shape")

    if image_1_data.ndim > 3:
        raise ValueError("Only 2D/3D images are supported")

    bounds_1 = bounds[0]
    bounds_2 = bounds[1]
    # from scipy.ndimage import rotate

    if image_1_data.ndim == 2:
        # plot directly the images and grid

        # Getting max boundaries in X and Y
        ty = transform[0, -1]
        tx = transform[1, -1]

        size_y = image_1_data.shape[0] + abs(ty)
        size_x = image_1_data.shape[1] + abs(tx)

        # size_x = max(bounds_1[1][0], bounds_2[1][0])
        # size_y = max(bounds_1[1][1], bounds_2[1][1])

        img_1_bounds, img_2_bounds = generate_new_boundaries()

        # Image within same coordinate system
        adjusted_img_1 = np.ones((size_y, size_x)) * 255
        adjusted_img_2 = np.ones((size_y, size_x)) * 255

        # flake8: noqa: E203
        # Getting data from the images to common coordinate image
        points_orig_img = (
            np.array(
                [np.arange(0, image_1_data.shape[0], step=1, dtype=np.intp)]
            ),
            np.array(
                [np.arange(0, image_1_data.shape[1], step=1, dtype=np.intp)]
            ),
        )

        points_img_1 = (
            np.array(
                [
                    np.arange(
                        img_1_bounds[0][0],
                        img_1_bounds[1][0],
                        step=1,
                        dtype=np.intp,
                    )
                ]
            ),
            np.array(
                [
                    np.arange(
                        img_1_bounds[0][1],
                        img_1_bounds[1][1],
                        step=1,
                        dtype=np.intp,
                    )
                ]
            ),
        )

        points_img_2 = (
            np.array(
                [
                    np.arange(
                        img_2_bounds[0][0],
                        img_2_bounds[1][0],
                        step=1,
                        dtype=np.intp,
                    )
                ]
            ),
            np.array(
                [
                    np.arange(
                        img_2_bounds[0][1],
                        img_2_bounds[1][1],
                        step=1,
                        dtype=np.intp,
                    )
                ]
            ),
        )

        xv_img_1, yv_img_1 = np.meshgrid(*points_img_1, indexing="ij")
        xv_img_2, yv_img_2 = np.meshgrid(*points_img_2, indexing="ij")
        xv_orig, yv_orig = np.meshgrid(*points_orig_img, indexing="ij")

        # vindex takes some time when processing with dask
        adjusted_img_1[xv_img_1, yv_img_1] = (
            image_1_data[xv_orig, yv_orig]
            if isinstance(image_1_data, np.ndarray)
            else image_1_data.vindex[xv_orig, yv_orig]
        )
        adjusted_img_2[xv_img_2, yv_img_2] = (
            image_2_data[xv_orig, yv_orig]
            if isinstance(image_2_data, np.ndarray)
            else image_2_data.vindex[xv_orig, yv_orig]
        )

        fig, ax = plt.subplots()

        fig.suptitle(f"Metric: {metric_name}", fontsize=12)

        # Setting X,Y positions of points within the grid

        y_points_img_1 = [point[0] for point in keypoints_img_1]
        x_points_img_1 = [point[1] for point in keypoints_img_1]

        y_points_img_2 = [point[0] for point in keypoints_img_2]
        x_points_img_2 = [point[1] for point in keypoints_img_2]

        # Scattering points within image in common coordinate system
        plt.scatter(x=x_points_img_1, y=y_points_img_1, c="r", s=10)
        plt.scatter(x=x_points_img_2, y=y_points_img_2, c="r", s=10)
        ax.plot(
            (keypoints_img_1[:, 1], keypoints_img_2[:, 1]),
            (keypoints_img_1[:, 0], keypoints_img_2[:, 0]),
            "-",
            color="red",
        )

        # Rectangles to divide images
        rectangle_image_1 = Rectangle(
            xy=(img_1_bounds[0][1], img_1_bounds[0][0]),
            width=img_1_bounds[1][1],
            height=img_1_bounds[1][0],
            linewidth=1,
            edgecolor="#FF2D00",
            facecolor="none",
            linestyle=":",
        )

        rectangle_image_2 = Rectangle(
            xy=(img_2_bounds[0][1], img_2_bounds[0][0]),
            width=img_1_bounds[1][1],
            height=img_1_bounds[1][0],
            linewidth=1,
            edgecolor="#13FF00",
            facecolor="none",
            linestyle="--",
        )

        # Alpha for overlaying
        alpha = 0.5
        vmin_1, vmax_1 = np.percentile(image_1_data, (0.2, 99))
        vmin_2, vmax_2 = np.percentile(image_2_data, (0.2, 99))
        ax.imshow(
            adjusted_img_1, alpha=alpha, vmin=vmin_1, vmax=vmax_1, cmap="Blues"
        )
        ax.imshow(
            adjusted_img_2,
            alpha=alpha,
            vmin=vmin_2,
            vmax=vmax_2,
            cmap="Oranges",
        )

        ax.add_patch(rectangle_image_1)
        ax.add_patch(rectangle_image_2)

        plt.tight_layout()
        plt.show()

    else:
        pass


def plot_matches(
    ax,
    image1,
    image2,
    keypoints1,
    keypoints2,
    matches,
    keypoints_color="k",
    matches_color=None,
    only_matches=False,
    alignment="horizontal",
):
    """Plot matched features.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matches and image are drawn in this ax.
    image1 : (N, M [, 3]) array
        First grayscale or color image.
    image2 : (N, M [, 3]) array
        Second grayscale or color image.
    keypoints1 : (K1, 2) array
        First keypoint coordinates as ``(row, col)``.
    keypoints2 : (K2, 2) array
        Second keypoint coordinates as ``(row, col)``.
    matches : (Q, 2) array
        Indices of corresponding matches in first and second set of
        descriptors, where ``matches[:, 0]`` denote the indices in the first
        and ``matches[:, 1]`` the indices in the second set of descriptors.
    keypoints_color : matplotlib color, optional
        Color for keypoint locations.
    matches_color : matplotlib color, optional
        Color for lines which connect keypoint matches. By default the
        color is chosen randomly.
    only_matches : bool, optional
        Whether to only plot matches and not plot the keypoint locations.
    alignment : {'horizontal', 'vertical'}, optional
        Whether to show images side by side, ``'horizontal'``, or one above
        the other, ``'vertical'``.

    """
    image1 = image1.astype(np.float16)
    image2 = image2.astype(np.float16)

    new_shape1 = list(image1.shape)
    new_shape2 = list(image2.shape)

    if image1.shape[0] < image2.shape[0]:
        new_shape1[0] = image2.shape[0]
    elif image1.shape[0] > image2.shape[0]:
        new_shape2[0] = image1.shape[0]

    if image1.shape[1] < image2.shape[1]:
        new_shape1[1] = image2.shape[1]
    elif image1.shape[1] > image2.shape[1]:
        new_shape2[1] = image1.shape[1]

    if new_shape1 != image1.shape:
        new_image1 = np.zeros(new_shape1, dtype=image1.dtype)
        new_image1[: image1.shape[0], : image1.shape[1]] = image1
        image1 = new_image1

    if new_shape2 != image2.shape:
        new_image2 = np.zeros(new_shape2, dtype=image2.dtype)
        new_image2[: image2.shape[0], : image2.shape[1]] = image2
        image2 = new_image2

    offset = np.array(image1.shape)
    if alignment == "horizontal":
        image = np.concatenate([image1, image2], axis=1)
        offset[0] = 0
    elif alignment == "vertical":
        image = np.concatenate([image1, image2], axis=0)
        offset[1] = 0
    else:
        mesg = (
            f"plot_matches accepts either 'horizontal' or 'vertical' for "
            f"alignment, but '{alignment}' was given. See "
            f"https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.plot_matches "  # noqa
            f"for details."
        )
        raise ValueError(mesg)

    if not only_matches:
        ax.scatter(
            keypoints1[:, 1],
            keypoints1[:, 0],
            facecolors="none",
            edgecolors=keypoints_color,
        )
        ax.scatter(
            keypoints2[:, 1] + offset[1],
            keypoints2[:, 0] + offset[0],
            facecolors="none",
            edgecolors=keypoints_color,
        )

    ax.imshow(image)  # , cmap='gray')
    ax.axis((0, image1.shape[1] + offset[1], image1.shape[0] + offset[0], 0))

    rng = np.random.default_rng()

    for i in range(matches.shape[0]):
        idx1 = matches[i, 0]
        idx2 = matches[i, 1]

        if matches_color is None:
            color = rng.random(3)
        else:
            color = matches_color

        ax.plot(
            (keypoints1[idx1, 1], keypoints2[idx2, 1] + offset[1]),
            (keypoints1[idx1, 0], keypoints2[idx2, 0] + offset[0]),
            "-",
            color=color,
        )

        ax.annotate(
            str(idx1),
            (keypoints1[idx1, 1], keypoints1[idx1, 0]),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
            color=color,
        )

        ax.annotate(
            str(idx2),
            (keypoints2[idx2, 1] + offset[1], keypoints2[idx2, 0] + offset[0]),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=8,
            color=color,
        )

import sys
import scipy
import numpy as np
from matplotlib import pyplot as plt
import random
import elliptic_fitting.utils as utils


def elliptic_fitting_by_weighted_repetition(noise_x, noise_y, f):
    W = np.ones(len(noise_x), dtype="float64")
    theta_zero = np.zeros(6, dtype="float64")
    diff = 10000.0
    # f=1

    while diff > 1e-10:
        xi_sum = np.zeros((6, 6))
        for i in range(len(noise_x)):
            x = noise_x[i]
            y = noise_y[i]
            xi = np.array([[x**2, 2 * x * y, y**2, 2 * f * x, 2 * f * y, f * f]])
            # xi = np.array([[x**2, 2 * x * y, x**2, 2 * f * x, 2 * f * y, f * f]])  # make C = A
            xi_sum += np.dot(np.dot(W[i], xi.T), xi)

        M = xi_sum / len(noise_x)
        w, v = np.linalg.eig(M)
        theta = v[:, np.argmin(w)]

        if np.dot(theta, theta_zero) < 0:
            theta = np.dot(-1, theta)
        diff = np.sum(np.abs(theta_zero - theta))
        # print("diff: ", diff)
        if diff <= 1e-10:
            break

        for i in range(len(noise_x)):
            x = noise_x[i]
            y = noise_y[i]
            # V0_xi = 4 * np.array(
            #     [
            #         [x**2, x * y, 0, f * x, 0, 0],
            #         [x * y, x**2 + x**2, x * y, f * y, f * x, 0],  # changed y**2 to x**2
            #         [0, x * y, x**2, 0, f * y, 0],  # changed y**2 to x**2
            #         [f * x, f * y, 0, f**2, 0, 0],
            #         [0, f * x, f * y, 0, f**2, 0],
            #         [0, 0, 0, 0, 0, 0],
            #     ]
            # )
            V0_xi = 4 * np.array(
                [
                    [x ** 2, x * y, 0, f * x, 0, 0],
                    [x * y, x ** 2 + y ** 2, x * y, f * y, f * x, 0],
                    [0, x * y, y ** 2, 0, f * y, 0],
                    [f * x, f * y, 0, f ** 2, 0, 0],
                    [0, f * x, f * y, 0, f ** 2, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            )
            W = np.insert(W, i, 1 / (np.dot(theta, np.dot(V0_xi, theta))))
        theta_zero = theta

    return theta


def elliptic_fitting_by_fns(noise_x, noise_y, f):
    W = np.ones(len(noise_x), dtype="float64")
    theta_zero = np.zeros(6, dtype="float64")
    diff = 10000.0

    while diff > 1e-10:
        xi_sum = np.zeros((6, 6))
        L_sum = np.zeros((6, 6))
        V0_xi_list = []
        for i in range(len(noise_x)):
            x = noise_x[i]
            y = noise_y[i]
            xi = np.array([[x**2, 2 * x * y, y**2, 2 * f * x, 2 * f * y, f * f]])
            xi_sum += np.dot(W[i], np.dot(xi.T, xi))
            V0_xi = 4 * np.array(
                [
                    [x**2, x * y, 0, f * x, 0, 0],
                    [x * y, x**2 + y**2, x * y, f * y, f * x, 0],
                    [0, x * y, y**2, 0, f * y, 0],
                    [f * x, f * y, 0, f**2, 0, 0],
                    [0, f * x, f * y, 0, f**2, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            )
            V0_xi_list.append(V0_xi)
            L_sum += np.dot(
                np.dot(W[i] ** 2, np.dot(xi.T[:, 0], theta_zero) ** 2), V0_xi
            )

        M = xi_sum / len(noise_x)
        L = L_sum / len(noise_x)
        X = M - L
        eig_val, eig_vec = np.linalg.eig(X)
        theta = eig_vec[:, np.argmin(eig_val)]

        if np.dot(theta, theta_zero) < 0:
            theta = np.dot(-1, theta)
        diff = np.sum(np.abs(theta_zero - theta))
        if diff <= 1e-10:
            break

        for i in range(len(noise_x)):
            W = np.insert(W, i, 1 / (np.dot(theta, np.dot(V0_xi_list[i], theta))))
        theta_zero = theta

    return theta


def elliptic_fitting_by_least_squares(noise_x, noise_y, f):
    xi_sum = np.zeros((6, 6))
    for i in range(len(noise_x)):
        x = noise_x[i]
        y = noise_y[i]
        xi = np.array([[x**2, 2 * x * y, y**2, 2 * f * x, 2 * f * y, f * f]])
        xi_sum += np.dot(xi.T, xi)

    M = xi_sum / len(noise_x)
    w, v = np.linalg.eig(M)
    min_eig_vec = v[:, np.argmin(w)]

    return min_eig_vec


def elliptic_fitting_by_renormalization(noise_x, noise_y, f):
    W = np.ones(len(noise_x), dtype="float64")
    theta_zero = np.zeros(6, dtype="float64")
    diff = 10000.0

    while diff > 1e-10:
        xi_sum = np.zeros((6, 6))
        N_sum = np.zeros((6, 6))
        V0_xi_list = []
        for i in range(len(noise_x)):
            x = noise_x[i]
            y = noise_y[i]
            xi = np.array([[x**2, 2 * x * y, y**2, 2 * f * x, 2 * f * y, f * f]])
            xi_sum += np.dot(np.dot(W[i], xi.T), xi)
            V0_xi = 4 * np.array(
                [
                    [x**2, x * y, 0, f * x, 0, 0],
                    [x * y, x**2 + y**2, x * y, f * y, f * x, 0],
                    [0, x * y, y**2, 0, f * y, 0],
                    [f * x, f * y, 0, f**2, 0, 0],
                    [0, f * x, f * y, 0, f**2, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            )
            V0_xi_list.append(V0_xi)
            N_sum += np.dot(W[i], V0_xi)

        M = xi_sum / len(noise_x)
        N = N_sum / len(noise_x)
        eig_val, eig_vec = scipy.linalg.eig(N, M)
        theta = eig_vec[:, np.argmax(eig_val)]

        if np.dot(theta, theta_zero) < 0:
            theta = np.dot(-1, theta)
        diff = np.sum(np.abs(theta_zero - theta))
        print("diff: ", diff)
        if diff <= 1e-10:
            break

        for i in range(len(noise_x)):
            W = np.insert(W, i, 1 / (np.dot(theta, np.dot(V0_xi_list[i], theta))))
        theta_zero = theta

    return theta


def removed_outlier(noise_x, noise_y, theta, f, threshold):
    removed_x = []
    removed_y = []
    for i in range(len(noise_x)):
        x = noise_x[i]
        y = noise_y[i]
        xi = np.array([[x**2, 2 * x * y, y**2, 2 * f * x, 2 * f * y, f * f]])
        V0_xi = 4 * np.array(
            [
                [x**2, x * y, 0, f * x, 0, 0],
                [x * y, x**2 + y**2, x * y, f * y, f * x, 0],
                [0, x * y, y**2, 0, f * y, 0],
                [f * x, f * y, 0, f**2, 0, 0],
                [0, f * x, f * y, 0, f**2, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        d = (np.dot(xi, theta)) ** 2 / (np.dot(theta, np.dot(V0_xi, theta)))
        if d[0] < threshold:
            removed_x.append(x)
            removed_y.append(y)

    return removed_x, removed_y


def remove_outlier_by_ransac(noise_x, noise_y, f):
    inliner_count_max = 0
    no_update_count = 0
    desired_theta = np.zeros(6, dtype="float64")
    threshold = 1
    while no_update_count < 5000:
        random_list = random.sample(range(len(noise_x)), k=5)
        M = np.zeros((6, 6))
        for idx in random_list:
            x = noise_x[idx]
            y = noise_y[idx]
            xi = np.array([[x**2, 2 * x * y, y**2, 2 * f * x, 2 * f * y, f * f]])
            M += np.dot(xi.T, xi)

        eig_val, eig_vec = np.linalg.eig(M)
        theta = eig_vec[:, np.argmin(eig_val)]

        removed_x = []
        removed_y = []
        inlier_count = 0
        for i in range(len(noise_x)):
            x = noise_x[i]
            y = noise_y[i]
            xi = np.array([[x**2, 2 * x * y, y**2, 2 * f * x, 2 * f * y, f * f]])
            V0_xi = 4 * np.array(
                [
                    [x**2, x * y, 0, f * x, 0, 0],
                    [x * y, x**2 + y**2, x * y, f * y, f * x, 0],
                    [0, x * y, y**2, 0, f * y, 0],
                    [f * x, f * y, 0, f**2, 0, 0],
                    [0, f * x, f * y, 0, f**2, 0],
                    [0, 0, 0, 0, 0, 0],
                ]
            )
            d = (np.dot(xi, theta)) ** 2 / (np.dot(theta, np.dot(V0_xi, theta)))
            if d[0] < threshold:
                inlier_count += 1

        if inlier_count > inliner_count_max:
            inliner_count_max = inlier_count
            desired_theta = theta
            no_update_count = 0
        else:
            no_update_count += 1

    removed_x, removed_y = removed_outlier(
        noise_x, noise_y, desired_theta, f, threshold
    )

    return removed_x, removed_y


def main():
    utils.plot_base()
    corr_x, corr_y, noise_x, noise_y = utils.get_elliptic_points_with_tilt(45)

    f_0 = 20
    w_theta = elliptic_fitting_by_weighted_repetition(noise_x, noise_y, f_0)
    w_fit_x, w_fit_y = utils.solve_fitting(w_theta, corr_x, f_0)

    weighted_diff, weighted_diff_avg = utils.eval_pos_diff(
        corr_x, corr_y, w_fit_x, w_fit_y
    )
    # print("weighted_diff_avg: ", weighted_diff_avg)

    plt.scatter(
        corr_x, corr_y, marker="o", c="black", s=20, alpha=0.4, label="Correct input"
    )
    plt.scatter(
        noise_x, noise_y, marker="o", c="blue", s=20, alpha=0.4, label="Noise input"
    )

    plt.scatter(
        w_fit_x,
        w_fit_y,
        marker="o",
        c="green",
        s=20,
        alpha=0.4,
        label="Weighted Repetition",
    )
    plt.legend()


if __name__ == "__main__":
    main()
    if len(sys.argv) == 2 and sys.argv[1] == "NotShow":
        print("It shows nothing")
    else:
        plt.show()
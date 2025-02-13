// This file is for the thin-plate spline 2d point matching, a specific form of non-rigid transformation.
//
// Copyright (C) 2019 Yang Zhenjie <amazingzhen@foxmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "rpm.h"

#include <iostream>
#include <chrono>

#include "data.h"

using std::cout;
using std::endl;
using namespace rpm;

// Annealing params
double rpm::T_start = 1;
double rpm::T_end = T_start * 1e-4;
double rpm::r = 0.90, rpm::I0 = 5, rpm::epsilon0 = 1e-2;
double rpm::alpha = 0.1; // 5 * 5
// Softassign params
double rpm::I1 = 10, rpm::epsilon1 = 1e-4;
// Thin-plate spline params
double rpm::lambda_start = T_start;

double rpm::scale = 300;

//#define USE_SVD_SOLVER

namespace {
    inline bool _matrices_equal(
            const MatrixXd &m1,
            const MatrixXd &m2,
            const double tol = 1e-3) {
        if (m1.rows() != m2.rows() || m1.cols() != m2.cols()) {
            return false;
        }

        return ((m1 - m2).cwiseAbs().maxCoeff() <= tol);
    }

    inline void _soft_assign(
            MatrixXd &assignment_matrix) {
        int iter = 0;
        while (iter++ < I1) {
            // normalizing across all rows
#pragma omp parallel for
            for (int r = 0; r < assignment_matrix.rows() - 1; r++) {
                double row_sum = assignment_matrix.row(r).sum();
                if (row_sum < epsilon1) {
                    continue;
                }
                assignment_matrix.row(r) /= row_sum;
            }

            // normalizing across all cols
#pragma omp parallel for
            for (int c = 0; c < assignment_matrix.cols() - 1; c++) {
                double col_sum = assignment_matrix.col(c).sum();
                if (col_sum < epsilon1) {
                    continue;
                }
                assignment_matrix.col(c) /= col_sum;
            }
        }
    }

    inline double _distance(const MatrixXd &Y_, const MatrixXd &M, const rpm::ThinPlateSplineParams &params) {
        MatrixXd Y = rpm::apply_correspondence(Y_, M);
        MatrixXd XT = params.applyTransform(true);

        if (XT.rows() != Y.rows() || XT.cols() != Y.cols()) {
            throw std::invalid_argument("X size not same as Y in _distance!");
        }

        MatrixXd diff = (Y - XT).cwiseAbs();
        return diff.maxCoeff();
    }
}

void rpm::set_T_start(double T, double scale) {
    T *= scale;

    T_start = T;
    T_end = T * 1e-3;
    lambda_start = T;

    cout << "Set T_start : " << T_start << endl;
    //getchar();
}

bool rpm::estimate(
        const MatrixXd &X_,
        const MatrixXd &Y_,
        MatrixXd &M,
        ThinPlateSplineParams &params,
        const vector<pair<int, int> > &matched_point_indices) {
    auto t1 = std::chrono::high_resolution_clock::now();

    try {
        if (X_.cols() != rpm::D || Y_.cols() != rpm::D) {
            throw std::invalid_argument("rpm::estimate() only support 2d points!");
        }

        MatrixXd X = X_, Y = Y_;

        data_process::preprocess(X, Y);
        data_process::homo(X);
        data_process::homo(Y);

        params = ThinPlateSplineParams(X);

        double max_dist = 0, average_dist = 0;
        int K = X.rows(), N = Y.rows();
        for (int k = 0; k < K; k++) {
            const VectorXd &x = X.row(k);
            for (int n = 0; n < N; n++) {
                const VectorXd &y = Y.row(n);
                double dist = (y - x).squaredNorm();

                max_dist = max(max_dist, dist);
                average_dist += dist;
            }
        }
        average_dist /= (K * N);
        std::cout << "max_dist : " << max_dist << std::endl;
        std::cout << "average_dist : " << average_dist << std::endl;
        set_T_start(average_dist, 1);
        //rpm::alpha = average_dist * 0.1;

        double T_cur = T_start;
        double lambda = lambda_start;

        if (!init_params(X, Y, T_start, M, params)) {
            throw std::runtime_error("init params failed!");
        }

        //char file[256];
        //if (data_visualize::save_intermediate_result) {
        //	sprintf_s(file, "%s/data_%.8f.png", data_visualize::res_dir.c_str(), T_cur);
        //	Mat result_image = data_visualize::visualize(params.applyTransform(), Y);
        //	imwrite(file, result_image);
        //}

        int indi = 0;
        while (T_cur >= T_end) {
//            printf("indi= %d, T : %.2f, ",indi, T_cur);
//            printf("lambda : %.2f ", lambda);
//            std:: cout << " " <<   std::  endl;


            int iter = 0;

            while (iter++ < I0) {
                //printf("	Annealing iter : %d\n", iter);
                MatrixXd M_prev = M;
                ThinPlateSplineParams params_prev = params;
                if (!estimate_correspondence(X, Y, matched_point_indices, params, T_cur, T_start, M)) {
                    throw std::runtime_error("estimate correspondence failed!");
                }

                if (!estimate_transform(X, Y, M, lambda, params)) {
                    throw std::runtime_error("estimate transform failed!");
                }

                std::cout << "indi= " << indi << ",iter = " << iter << ",T_cur = " << T_cur << ",T_end=" << T_end
                          << std::endl;

                //if (_matrices_equal(M_prev, M, epsilon0)) {  // hack!!!
                //	//M = M_prev;
                //	//params = params_prev;
                //	break;
                //}
            }
            indi++;

            //if (data_visualize::save_intermediate_result) {
            //	sprintf_s(file, "%s/data_%.8f.png", data_visualize::res_dir.c_str(), T_cur);
            //	Mat result_image = data_visualize::visualize(params.applyTransform(), Y, scale);
            //	imwrite(file, result_image);
            //}

            T_cur *= r;
            lambda *= r;
        }

        // Re-estimate real ThinPlateSplineParams on unnormalized data.

        //MatrixXd M_binary = MatrixXd::Zero(K, N);
        //for (int k = 0; k < K; k++) {
        //	Eigen::Index n;
        //	double max_coeff = M.row(k).maxCoeff(&n);
        //	if (max_coeff > 1.0 / N) {
        //		M_binary(k, n) = 1;
        //	}
        //}
        //M = M_binary;

        //params = ThinPlateSplineParams(X_);
        //	estimate_transform(X_, Y_, M, lambda, params);
    }
    catch (const std::exception &e) {
        std::cout << e.what();
        return false;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    auto timespan = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "TPS-RPM estimate time: " << timespan.count() << " seconds.\n";

    return true;
}

bool rpm::init_params(
        const MatrixXd &X,
        const MatrixXd &Y,
        const double T,
        MatrixXd &M,
        ThinPlateSplineParams &params) {
    const int K = X.rows(), N = Y.rows();

    //estimate_transform(X, X, MatrixXd::Identity(K + 1, K + 1), T, lambda, params);

    //estimate_correspondence(X, Y, params, T, T, M);

    //M = Eigen::MatrixXd::Identity(K + 1, N + 1);

    //	const double beta = 1.0 / T;
    //	M = Eigen::MatrixXd::Zero(K + 1, N + 1);
    //#pragma omp parallel for
    //	for (int k = 0; k < K; k++) {
    //		const VectorXd& x = X.row(k);
    //		for (int n = 0; n < N; n++) {
    //			const VectorXd& y = Y.row(n);
    //
    //			//assignment_matrix(p_i, v_i) = -((p[p_i] - v[v_i]).squaredNorm() - alpha);
    //			double dist = ((y - x).squaredNorm());
    //
    //			//assignment_matrix(p_i, v_i) = dist < alpha ? std::exp(-(1.0 / T) * dist) : 0;
    //			M(k, n) = std::exp(beta * -dist);
    //		}
    //	};
    //
    //	_soft_assign(M);
    //cout << "M" << endl;
    //cout << M << endl;

    return true;
}

bool rpm::estimate_correspondence(
        const MatrixXd &X,
        const MatrixXd &Y,
        const vector<pair<int, int> > &matched_point_indices,
        const ThinPlateSplineParams &params,
        const double T,
        const double T0,
        MatrixXd &M) {
    if (X.cols() != D + 1 || Y.cols() != D + 1) {
        throw std::invalid_argument("Current only support 3d homogeneou points!");
    }

    const int K = X.rows(), N = Y.rows();
    const double beta = 1.0 / T;

    M = MatrixXd::Zero(K + 1, N + 1);

    MatrixXd XT = params.applyTransform();

#pragma omp parallel for
    for (int k = 0; k < K; k++) {
        const Vector3d &x = XT.row(k);
        for (int n = 0; n < N; n++) {
            const Vector3d &y = Y.row(n);

            //assignment_matrix(p_i, v_i) = -((p[p_i] - v[v_i]).squaredNorm() - alpha);
            double dist = ((y - x).squaredNorm());

            //assignment_matrix(p_i, v_i) = dist < alpha ? std::exp(-(1.0 / T) * dist) : 0;
            M(k, n) = std::exp(beta * (alpha - dist));
        }
    };

    for (auto point_pair : matched_point_indices) {
        int k = point_pair.first, n = point_pair.second;
        if (k < 0 || k >= K || n < 0 || n >= N) {
            continue;
        }

        M.row(k).setZero();
        M.col(n).setZero();
        M(k, n) = 1;
    }

    //Vector3d center_x(XT.col(0).mean(), XT.col(1).mean(), XT.col(2).mean());
    //Vector3d center_y(Y.col(0).mean(), Y.col(1).mean(), Y.col(2).mean());

    //	const double beta_start = 1.0 / T0;
    //#pragma omp parallel for
    //	for (int k = 0; k < K; k++) {
    //		const Vector3d& x = XT.row(k);
    //		double dist = ((center_y - x).squaredNorm());
    //		M(k, N) = beta_start * std::exp(beta_start * -dist);
    //	}
    //
    //#pragma omp parallel for
    //	for (int n = 0; n < N; n++) {
    //		const Vector3d& y = Y.row(n);
    //		double dist = ((y - center_x).squaredNorm());
    //		M(K, n) = beta_start * std::exp(beta_start * -dist);
    //	}

    M.row(K).setConstant(1.0 / (N + 1));
    M.col(N).setConstant(1.0 / (K + 1));

    _soft_assign(M);

    M.conservativeResize(K, N);

    return true;
}

bool rpm::estimate_transform(
        const MatrixXd &X,
        const MatrixXd &Y_,
        const MatrixXd &M,
        const double lambda,
        ThinPlateSplineParams &params) {
    //auto t1 = std::chrono::high_resolution_clock::now();

    try {
        if (X.cols() != D + 1 || Y_.cols() != D + 1) {
            throw std::invalid_argument("Current only support 3d homogeneou points!");
        }

        const int K = X.rows(), N = Y_.rows();
        if (M.rows() != K || M.cols() != N) {
            throw std::invalid_argument("Matrix M size not same as X and Y!");
        }

        int dim = D + 1;
        MatrixXd Y = apply_correspondence(Y_, M);

        const MatrixXd &phi = params.get_phi();
        const MatrixXd &Q = params.get_Q();
        const MatrixXd &R_ = params.get_R();

        MatrixXd Q1 = Q.block(0, 0, K, dim), Q2 = Q.block(0, dim, K, K - dim);
        MatrixXd R = R_.block(0, 0, dim, dim);

#ifdef RPM_USE_BOTHSIDE_OUTLIER_REJECTION
        MatrixXd W = MatrixXd::Zero(K, K);
        for (int k = 0; k < K; k++) {
            W(k, k) = 1.0 / std::max(M.row(k).sum(), epsilon1);
        }

        MatrixXd T = phi + N * lambda * W;

        LDLT<MatrixXd> solver;
        MatrixXd L_mat = Q2.transpose() * T * Q2;

        solver.compute(L_mat.transpose() * L_mat);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Param w ldlt decomposition failed!");
        }

        MatrixXd b_mat = Q2.transpose() * Y;
        MatrixXd gamma = solver.solve(L_mat.transpose() * b_mat);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Param w ldlt solve failed!");
        }

        params.w = Q2 * gamma;


#ifdef RPM_REGULARIZE_AFFINE_PARAM  // Add regular term lambdaI * d = lambdaI * I
        double lambda_d = N * lambda * 0.01;

        L_mat = MatrixXd(R.rows() * 2, R.cols());
        L_mat << R,
                MatrixXd::Identity(R.rows(), R.cols()) * lambda_d;
#else
        L_mat = R;
#endif // RPM_REGULARIZE_AFFINE_PARAM

        solver.compute(L_mat.transpose() * L_mat);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Param d ldlt decomposition failed!");
        }

#ifdef RPM_REGULARIZE_AFFINE_PARAM
        b_mat = MatrixXd(R.rows() * 2, R.cols());
        b_mat << Q1.transpose() * (Y - T * params.w),
                MatrixXd::Identity(R.rows(), R.cols()) * lambda_d;
#else
        b_mat = Q1.transpose() * (Y - K * params.w);
#endif // RPM_REGULARIZE_AFFINE_PARAM

        params.d = solver.solve(L_mat.transpose() * b_mat);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Param d ldlt solve failed!");
        }
#else
        LDLT<MatrixXd> solver;
        MatrixXd L_mat = (Q2.transpose() * phi * Q2 + (MatrixXd::Identity(K - dim, K - dim) * K * lambda));

        solver.compute(L_mat.transpose() * L_mat);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Param w ldlt decomposition failed!");
        }

        MatrixXd b_mat = Q2.transpose() * Y;
        MatrixXd gamma = solver.solve(L_mat.transpose() * b_mat);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Param w ldlt solve failed!");
        }

        params.w = Q2 * gamma;


#ifdef RPM_REGULARIZE_AFFINE_PARAM  // Add regular term lambdaI * d = lambdaI * I
        double lambda_d = K * lambda * 0.01;

        L_mat = MatrixXd(R.rows() * 2, R.cols());
        L_mat << R,
                MatrixXd::Identity(R.rows(), R.cols()) * lambda_d;
#else
        L_mat = R;
#endif // RPM_REGULARIZE_AFFINE_PARAM

        solver.compute(L_mat.transpose() * L_mat);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Param d ldlt decomposition failed!");
        }

#ifdef RPM_REGULARIZE_AFFINE_PARAM
        b_mat = MatrixXd(R.rows() * 2, R.cols());
        b_mat << Q1.transpose() * (Y - phi * params.w),
                MatrixXd::Identity(R.rows(), R.cols()) * lambda_d;
#else
        b_mat = Q1.transpose() * (Y - phi * params.w);
#endif // RPM_REGULARIZE_AFFINE_PARAM

        params.d = solver.solve(L_mat.transpose() * b_mat);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Param d ldlt solve failed!");
        }

        // Another form of regularize d.
        //MatrixXd A = (R.transpose() * R + 0.01 * lambda * MatrixXd::Identity(dim, dim)).inverse()
        //	* (R.transpose() * ((Q1.transpose() * (Y - phi * params.w)) - R));
        //params.d = A + MatrixXd::Identity(dim, dim);

#endif // RPM_USE_BOTHSIDE_OUTLIER_REJECTION
    }
    catch (const std::exception &e) {
        std::cout << e.what() << std::endl;

        return false;
    }

    //auto t2 = std::chrono::high_resolution_clock::now();

    //auto span = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1);
    //std::cout << "Thin-plate spline params estimating time: " << span.count() << " seconds.\n";

    return true;
}

MatrixXd rpm::apply_correspondence(const MatrixXd &Y, const MatrixXd &M) {
    if (Y.cols() != rpm::D + 1) {
        throw std::invalid_argument("input must be 3d homogeneou points!");
    }

    MatrixXd MY = M * Y;
#ifdef RPM_USE_BOTHSIDE_OUTLIER_REJECTION
    for (int k = 0; k < M.rows(); k++) {
        MY.row(k) /= std::max(M.row(k).sum(), epsilon1);
    }
#endif // RPM_USE_BOTHSIDE_OUTLIER_REJECTION

    return MY;
}

rpm::ThinPlateSplineParams::ThinPlateSplineParams(const MatrixXd &X_) {
    X = X_;
    data_process::homo(X);

    const int K = X.rows();

    phi = MatrixXd::Zero(K, K);  // phi(a, b) = || Xb - Xa || ^ 2 * log(|| Xb - Xa ||);
#pragma omp parallel for
    for (int a_i = 0; a_i < K; a_i++) {
        VectorXd a = X.row(a_i);

        for (int b_i = 0; b_i < K; b_i++) {
            if (b_i == a_i) {
                continue;
            }

            VectorXd b = X.row(b_i);

            phi(b_i, a_i) = ((b - a).squaredNorm() * log((b - a).norm()));
        }
    }

    HouseholderQR<MatrixXd> qr;
    qr.compute(X);

    Q = qr.householderQ();
    R = qr.matrixQR().triangularView<Upper>();

    w = MatrixXd::Zero(X.rows(), rpm::D + 1);
    d = MatrixXd::Identity(rpm::D + 1, rpm::D + 1);
}

rpm::ThinPlateSplineParams::ThinPlateSplineParams(const ThinPlateSplineParams &other) {
    d = other.d;
    w = other.w;
    X = other.X;
    phi = other.phi;
    Q = other.Q;
    R = other.R;
}

MatrixXd rpm::ThinPlateSplineParams::applyTransform(bool hnormalize) const {
    MatrixXd XT = X * d + phi * w;

    if (hnormalize) {
        data_process::hnorm(XT);
    }

    return XT;
}

MatrixXd rpm::ThinPlateSplineParams::applyTransform(const MatrixXd &P_, bool hnormalize) const {
    MatrixXd P = P_;
    data_process::homo(P);

    const int N = P.rows();
    const int K = X.rows();

    MatrixXd phi_px = MatrixXd::Zero(N, K);  // phi(a, b) = || Xb - Xa || ^ 2 * log(|| Xb - Xa ||);
#pragma omp parallel for
    for (int p_i = 0; p_i < N; p_i++) {
        const Vector3d &p = P.row(p_i);

        for (int x_i = 0; x_i < K; x_i++) {
            const Vector3d &x = X.row(x_i);

            double dist = (p - x).norm();
            if (dist > 1e-5) {
                phi_px(p_i, x_i) = (dist * dist) * log(dist);
            }
        }
    }

    MatrixXd PT = P * d + phi_px * w;
    if (hnormalize) {
        data_process::hnorm(PT);
    }
    return PT;
}

Vector2d rpm::ThinPlateSplineParams::applyTransform(const Vector2d &p, bool hnormalize) const {
    Vector3d P = p.homogeneous();

    const int K = X.rows();
    VectorXd phi_px = VectorXd::Zero(K);  // phi(a, b) = || Xb - Xa || ^ 2 * log(|| Xb - Xa ||);
#pragma omp parallel for
    for (int x_i = 0; x_i < K; x_i++) {
        const Vector3d &x = X.row(x_i);

        double dist = (P - x).norm();
        if (dist > 1e-5) {
            phi_px(x_i) = (dist * dist) * log(dist);
        }
    }

    Vector3d PT = d.transpose() * P + w.transpose() * phi_px;
    return PT.hnormalized();
}

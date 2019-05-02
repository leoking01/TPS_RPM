// This file is for the thin-plate spline 2d point matching, a specific form of non-rigid transformation.
//
// Copyright (C) 2019 Yang Zhenjie <amazingzhen@foxmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include <Eigen/Dense>

using namespace Eigen;

namespace rpm {
	const static int D = 2;
	// Annealing params
	const static double T_start = 1500, T_end = T_start * 0.001;
	const static double r = 0.93, I0 = 5, epsilon0 = 2 * 1e-2;
	const static double alpha = 25.0; // 5 * 5
	// Softassign params
	const static double I1 = 30, epsilon1 = 1e-3;
	// Thin-plate spline params
	const static double lambda_start = T_start * 0.2;

	class ThinPLateSplineParams {
	public:
		ThinPLateSplineParams(const MatrixXd &X);

		// (D + 1) * (D + 1) matrix representing the affine transformation.
		MatrixXd d;
		// K * (D + 1) matrix representing the non-affine deformation.
		MatrixXd w;

		MatrixXd applyTransform(bool homo) const;
		VectorXd applyTransform(int x_i) const;

		MatrixXd get_phi() { return phi; };
		MatrixXd get_Q() { return Q; };
		MatrixXd get_R() { return R; };

	private:
		MatrixXd X;

		// K * K matrix
		MatrixXd phi;

		// Q, R
		MatrixXd Q, R;
	};

	// Compute the thin-plate spline params and 2d point correspondence from two point sets.
	//
	// Input:
	//   X, Y		source and target points set.
	// Output:
	//	 M			correspondence between X and Y
	//	 params		thin-plate spline params
	// Returns true on success, false on failure
	//
	bool estimate(
		const MatrixXd& X,
		const MatrixXd& Y,
		MatrixXd& M,
		ThinPLateSplineParams& params
	);

	bool init_params(
		const MatrixXd& X,
		const MatrixXd& Y,
		const double T,
		MatrixXd& M,
		ThinPLateSplineParams& params
	);

	// Compute the thin-plate spline parameters from two point sets.
	//
	// Input:
	//   X, Y		source and target points set.
	//	 params		thin-plate spline params
	//	 T			temperature
	// Output:
	//	 M			correspondence between X and Y
	// Returns true on success, false on failure
	//
	bool estimate_correspondence(
		const MatrixXd& X,
		const MatrixXd& Y,
		const ThinPLateSplineParams& params,
		const double T,
		const double T0,
		MatrixXd& M
	);

	// Compute the thin-plate spline parameters from two point sets.
	//
	// Input:
	//   X, Y		source and target points set.
	//	 M			correspondence between X and Y
	// Output:
	//	 params		thin-plate spline params
	// Returns true on success, false on failure
	//
	bool estimate_transform(
		const MatrixXd& X,
		const MatrixXd& Y,
		const MatrixXd& M,
		const double T,
		const double lambda,
		ThinPLateSplineParams& params
	);

	MatrixXd apply_correspondence(
		const MatrixXd& Y,
		const MatrixXd& M);
}



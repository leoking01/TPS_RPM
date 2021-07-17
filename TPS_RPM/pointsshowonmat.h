#ifndef POINTSSHOWONMAT_H
#define POINTSSHOWONMAT_H

#include <iostream>
#include <random>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

class PointsShowOnMat {
public:
    PointsShowOnMat() {
        _dstWid = 300;
        _dstHei = 300;
        _m_imgShow = cv::Mat(_dstHei + _dstHei + _dstHei, _dstWid + _dstWid + _dstWid, CV_8UC3,
                             cv::Scalar(255, 255, 255));
    }

public:
    PointsShowOnMat(int dstWid, int dstHei) {
        _dstWid = dstWid;
        _dstHei = dstHei;
        _m_imgShow = cv::Mat(_dstHei + _dstHei + _dstHei, _dstWid + _dstWid + _dstWid, CV_8UC3,
                             cv::Scalar(255, 255, 255));
    }

    void show_grid_on_image(cv::Scalar color = cv::Scalar(128, 128, 128)) {
        for (int i = -_dstWid; i < _dstWid + _dstWid; i += 50) {
            grid_vec_x.push_back(i + _dstWid);
        }
        for (int j = -_dstHei; j < _dstHei + _dstHei; j += 50) {
            grid_vec_y.push_back(j + _dstHei);
        }

        for (int j = 0; j < grid_vec_y.size() - 1; j += 1) {
            for (int i = 0; i < grid_vec_x.size() - 1; i += 1) {
                cv::line(_m_imgShow, cv::Point2d(grid_vec_x[i], grid_vec_x[j]),
                         cv::Point2d(grid_vec_x[i], grid_vec_x[j + 1]), color, 1);
                cv::line(_m_imgShow, cv::Point2d(grid_vec_x[i], grid_vec_x[j]),
                         cv::Point2d(grid_vec_x[i + 1], grid_vec_x[j]), color, 1);
            }
        }
    }

    std::vector<int> grid_vec_x;
    std::vector<int> grid_vec_y;


    void show_pts_on_image(Eigen::MatrixXd X_origin__t, cv::Scalar color, double radius) {
        Eigen::MatrixXd X_origin = X_origin__t / 2.0;

        for (int i = 0; i < X_origin.rows(); i++) {
            cv::Point2d pt = getDstPt(X_origin.row(i).x(), X_origin.row(i).y());
            cv::circle(_m_imgShow, pt, radius, color, 1);
        }
    }

    cv::Point2d getDstPt(double x, double y) {
        return cv::Point2d(x * _dstWid + _dstWid, y * _dstHei + _dstHei);
    }

public:
    cv::Mat _m_imgShow;
    int _dstWid;
    int _dstHei;
};

#endif // POINTSSHOWONMAT_H

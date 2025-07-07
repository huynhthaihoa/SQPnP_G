#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include "../sqpnp/sqpnp.h"
#include "../sqpnp/unified_sqpnp.h"

namespace py = pybind11;

// Helper function to convert numpy array to vector of points
std::vector<sqpnp::_Point> numpy_to_points_3d(const Eigen::Ref<const Eigen::MatrixXd>& points) {
    std::vector<sqpnp::_Point> result;
    result.reserve(points.rows());
    
    for (int i = 0; i < points.rows(); ++i) {
        if (points.cols() >= 3) {
            result.emplace_back(points(i, 0), points(i, 1), points(i, 2));
        }
    }
    return result;
}

// Helper function to convert numpy array to vector of projections
std::vector<sqpnp::_Projection> numpy_to_projections(const Eigen::Ref<const Eigen::MatrixXd>& points) {
    std::vector<sqpnp::_Projection> result;
    result.reserve(points.rows());
    
    for (int i = 0; i < points.rows(); ++i) {
        if (points.cols() >= 2) {
            result.emplace_back(points(i, 0), points(i, 1));
        }
    }
    return result;
}

// Helper function to normalize coordinates based on camera parameters
std::vector<sqpnp::_Projection> normalize_coordinates(const std::vector<sqpnp::_Projection>& projections, 
                                                     const Eigen::Ref<const Eigen::VectorXd>& camera_params) {
    int num_params = camera_params.size();
    std::vector<sqpnp::_Projection> normalized;
    normalized.reserve(projections.size());
    
    if (num_params == 4) {
        // Simple pinhole camera: [fx, fy, cx, cy]
        double fx = camera_params(0), fy = camera_params(1);
        double cx = camera_params(2), cy = camera_params(3);
        
        for (const auto& proj : projections) {
            double x = (proj.vector(0) - cx) / fx;
            double y = (proj.vector(1) - cy) / fy;
            normalized.emplace_back(x, y);
        }
    } else if (num_params == 8) {
        // Fisheye camera (KB model): [fx, fy, cx, cy, k1, k2, k3, k4]
        double fx = camera_params(0), fy = camera_params(1);
        double cx = camera_params(2), cy = camera_params(3);
        double k1 = camera_params(4), k2 = camera_params(5);
        double k3 = camera_params(6), k4 = camera_params(7);
        
        for (const auto& proj : projections) {
            double x = (proj.vector(0) - cx) / fx;
            double y = (proj.vector(1) - cy) / fy;
            
            // Apply fisheye distortion correction (simplified)
            double r2 = x*x + y*y;
            double r4 = r2*r2;
            double distortion = 1.0 + k1*r2 + k2*r4 + k3*r2*r4 + k4*r4*r4;
            
            normalized.emplace_back(x / distortion, y / distortion);
        }
    } else if (num_params == 10) {
        // Pinhole with distortion: [fx, fy, cx, cy, k1, k2, k3, k4, p1, p2]
        double fx = camera_params(0), fy = camera_params(1);
        double cx = camera_params(2), cy = camera_params(3);
        double k1 = camera_params(4), k2 = camera_params(5);
        double k3 = camera_params(6), k4 = camera_params(7);
        double p1 = camera_params(8), p2 = camera_params(9);
        
        for (const auto& proj : projections) {
            double x = (proj.vector(0) - cx) / fx;
            double y = (proj.vector(1) - cy) / fy;
            
            // Apply radial and tangential distortion correction
            double r2 = x*x + y*y;
            double r4 = r2*r2;
            double radial = 1.0 + k1*r2 + k2*r4 + k3*r2*r4 + k4*r4*r4;
            
            double tangential_x = 2*p1*x*y + p2*(r2 + 2*x*x);
            double tangential_y = p1*(r2 + 2*y*y) + 2*p2*x*y;
            
            normalized.emplace_back((x - tangential_x) / radial, (y - tangential_y) / radial);
        }
    } else {
        throw std::runtime_error("Invalid number of camera parameters. Expected 4, 8, or 10 parameters.");
    }
    
    return normalized;
}

// Python wrapper class for SQPnP results
class SQPnPResult {
public:
    bool success;
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    double error;
    int num_solutions;
    
    SQPnPResult(bool s, const Eigen::Matrix3d& R, const Eigen::Vector3d& t, 
                double e, int ns) : success(s), rotation(R), translation(t), error(e), num_solutions(ns) {}
};

// Main SQPnP solver class
class SQPnPSolver {
public:
    SQPnPSolver() = default;
    
    // Solve PnP with simple pinhole camera
    SQPnPResult solve_pinhole(const Eigen::Ref<const Eigen::MatrixXd>& points_3d,
                             const Eigen::Ref<const Eigen::MatrixXd>& points_2d,
                             const Eigen::Ref<const Eigen::VectorXd>& camera_params) {
        try {
            auto points3d = numpy_to_points_3d(points_3d);
            auto points2d_raw = numpy_to_projections(points_2d);
            auto points2d = normalize_coordinates(points2d_raw, camera_params);
            
            sqpnp::PnPSolver solver(points3d, points2d);
            bool success = solver.Solve();
            
            if (success && solver.NumberOfSolutions() > 0) {
                const auto& solution = solver.SolutionPtr(0);
                // Convert 9x1 rotation vector to 3x3 matrix
                Eigen::Matrix3d R = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(solution->r_hat.data());
                Eigen::Vector3d t = solution->t;
                double error = solver.AverageSquaredProjectionErrors()[0];
                
                return SQPnPResult(success, R, t, error, solver.NumberOfSolutions());
            } else {
                return SQPnPResult(false, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), std::numeric_limits<double>::max(), 0);
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("SQPnP error: ") + e.what());
        }
    }
    
    // Solve PnP with RANSAC for robust estimation
    SQPnPResult solve_ransac(const Eigen::Ref<const Eigen::MatrixXd>& points_3d,
                            const Eigen::Ref<const Eigen::MatrixXd>& points_2d,
                            const Eigen::Ref<const Eigen::VectorXd>& camera_params,
                            int max_iterations = 1000,
                            double threshold = 2.0,
                            double confidence = 0.99) {
        try {
            auto points3d = numpy_to_points_3d(points_3d);
            auto points2d_raw = numpy_to_projections(points_2d);
            auto points2d = normalize_coordinates(points2d_raw, camera_params);
            
            // Use UnifiedPnPSolver for RANSAC
            sqpnp::RansacParameters ransac_params;
            ransac_params.max_iterations = max_iterations;
            ransac_params.outlier_threshold = threshold;
            ransac_params.inlier_percentage = confidence;
            
            sqpnp::UnifiedPnPSolver solver(points3d, points2d, ransac_params);
            
            if (solver.isValid()) {
                sqpnp::UnifiedPoseResult result = solver.solve();
                
                if (result.success) {
                    return SQPnPResult(true, result.rotation, result.translation, 
                                     result.reprojection_error, 1);
                } else {
                    return SQPnPResult(false, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), 
                                     std::numeric_limits<double>::max(), 0);
                }
            } else {
                return SQPnPResult(false, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), 
                                 std::numeric_limits<double>::max(), 0);
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("SQPnP RANSAC error: ") + e.what());
        }
    }
    
    std::string get_camera_type(const Eigen::Ref<const Eigen::VectorXd>& camera_params) {
        int num_params = camera_params.size();
        if (num_params == 4) return "pinhole";
        else if (num_params == 8) return "fisheye";
        else if (num_params == 10) return "pinhole_with_distortion";
        else return "unknown";
    }
};

PYBIND11_MODULE(sqpnp_python, m) {
    m.doc() = "Python bindings for SQPnP - Efficient Perspective-n-Point Algorithm";
    
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "Manoj Pandey";
    m.attr("__description__") = "Enhanced SQPnP with multi-camera support and RANSAC integration";
    
    py::class_<SQPnPResult>(m, "SQPnPResult")
        .def_readonly("success", &SQPnPResult::success, "Whether the PnP solution was successful")
        .def_readonly("rotation", &SQPnPResult::rotation, "3x3 rotation matrix")
        .def_readonly("translation", &SQPnPResult::translation, "3D translation vector")
        .def_readonly("error", &SQPnPResult::error, "Average squared projection error")
        .def_readonly("num_solutions", &SQPnPResult::num_solutions, "Number of solutions found")
        .def("__repr__", [](const SQPnPResult& self) {
            return "SQPnPResult(success=" + std::to_string(self.success) + 
                   ", error=" + std::to_string(self.error) + 
                   ", num_solutions=" + std::to_string(self.num_solutions) + ")";
        });
    
    py::class_<SQPnPSolver>(m, "SQPnPSolver")
        .def(py::init<>(), "Initialize SQPnP solver")
        .def("solve_pinhole", &SQPnPSolver::solve_pinhole, 
             py::arg("points_3d"), py::arg("points_2d"), py::arg("camera_params"),
             "Solve PnP with a specified camera model.")
        .def("solve_ransac", &SQPnPSolver::solve_ransac,
             py::arg("points_3d"), py::arg("points_2d"), py::arg("camera_params"),
             py::arg("max_iterations") = 1000, py::arg("threshold") = 2.0, 
             py::arg("confidence") = 0.99,
             "Solve PnP with RANSAC for robust estimation.")
        .def("get_camera_type", &SQPnPSolver::get_camera_type,
             py::arg("camera_params"),
             "Get camera type from parameter count.")
        .def("__repr__", [](const SQPnPSolver&) {
            return "SQPnPSolver()";
        });
    
    m.def("solve_pnp", [](const Eigen::Ref<const Eigen::MatrixXd>& points_3d,
                         const Eigen::Ref<const Eigen::MatrixXd>& points_2d,
                         const Eigen::Ref<const Eigen::VectorXd>& camera_params) {
        SQPnPSolver solver;
        return solver.solve_pinhole(points_3d, points_2d, camera_params);
    }, py::arg("points_3d"), py::arg("points_2d"), py::arg("camera_params"),
       "Convenience function to solve PnP with a specified camera model.");
    
    m.def("solve_pnp_ransac", [](const Eigen::Ref<const Eigen::MatrixXd>& points_3d,
                                const Eigen::Ref<const Eigen::MatrixXd>& points_2d,
                                const Eigen::Ref<const Eigen::VectorXd>& camera_params,
                                int max_iterations = 1000,
                                double threshold = 2.0,
                                double confidence = 0.99) {
        SQPnPSolver solver;
        return solver.solve_ransac(points_3d, points_2d, camera_params, 
                                 max_iterations, threshold, confidence);
    }, py::arg("points_3d"), py::arg("points_2d"), py::arg("camera_params"),
       py::arg("max_iterations") = 1000, py::arg("threshold") = 2.0, 
       py::arg("confidence") = 0.99,
       "Convenience function to solve PnP with RANSAC.");
}
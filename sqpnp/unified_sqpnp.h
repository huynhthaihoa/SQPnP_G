//
// unified_sqpnp.h
//
// Unified SQPnP interface providing both simple and robust (RANSAC) PnP solvers
// under a single namespace for clean, minimal integration
//

#ifndef UNIFIED_SQPnP_H__
#define UNIFIED_SQPnP_H__

#include "types.h"
#include "sqpnp.h"
#include "RansacLib/ransac.h"
#include "RansacLib/sampling.h"
#include "RansacLib/utils.h"
#include <vector>
#include <memory>
#include <chrono>

namespace sqpnp {

// Unified pose result structure
struct UnifiedPoseResult {
    Eigen::Matrix3d rotation;           // 3x3 rotation matrix
    Eigen::Vector3d translation;        // 3x1 translation vector
    double reprojection_error;          // Average reprojection error
    int num_inliers;                    // Number of inliers (for robust methods)
    int num_outliers;                   // Number of outliers (for robust methods)
    std::vector<int> inlier_indices;    // Indices of inlier points
    std::vector<int> outlier_indices;   // Indices of outlier points
    bool success;                       // Whether the solution was successful
    double execution_time_us;           // Execution time in microseconds
    
    UnifiedPoseResult() : reprojection_error(0.0), num_inliers(0), num_outliers(0), 
                         success(false), execution_time_us(0.0) {}
};

// RANSAC parameters structure
struct RansacParameters {
    int min_iterations = 100;           // Minimum RANSAC iterations
    int max_iterations = 1000;          // Maximum RANSAC iterations
    double inlier_percentage = 0.8;     // Expected inlier percentage (0.0-1.0)
    double outlier_threshold = 0.1;     // Outlier threshold in pixels
    int min_sample_size = 6;            // Minimum sample size for RANSAC
    int non_minimal_sample_size = 20;   // Non-minimal sample size for refinement
    
    RansacParameters() = default;
    
    RansacParameters(int min_iter, int max_iter, double inlier_pct, double threshold)
        : min_iterations(min_iter), max_iterations(max_iter), 
          inlier_percentage(inlier_pct), outlier_threshold(threshold) {}
};

// Pose as [R t] for RANSAC
typedef Eigen::Matrix<double, 3, 4> Matrix34d;

// Internal RANSAC solver class (not exposed to user)
class InternalRansacSolver {
public:
    InternalRansacSolver(const std::vector<_Point>* points3d, 
                        const std::vector<_Projection>* points2d,
                        int min_sample_size, int non_minimal_sample_size)
        : points3d_(points3d), points2d_(points2d), 
          min_sample_size_(min_sample_size), non_minimal_sample_size_(non_minimal_sample_size) {}
    
    // Required interface for RansacLib
    int min_sample_size() const { return min_sample_size_; }
    int non_minimal_sample_size() const { return non_minimal_sample_size_; }
    int num_data() const { return static_cast<int>(points3d_->size()); }
    
    // Minimal solver (required by RansacLib)
    int MinimalSolver(const std::vector<int>& sample, std::vector<Matrix34d>* poses) const {
        const int nsample = sample.size();
        if (nsample < min_sample_size_) return 0;
        
        // Extract sample points
        std::vector<_Point> points(nsample);
        std::vector<_Projection> projections(nsample);
        
        for (int i = 0; i < nsample; ++i) {
            int j = sample[i];
            points[i] = (*points3d_)[j];
            projections[i] = (*points2d_)[j];
        }
        
        // Solve with SQPnP
        SolverParameters params;
        std::vector<double> weights(nsample, 1.0);
        PnPSolver solver(points, projections, weights, params);
        
        if (solver.IsValid()) {
            solver.Solve();
            poses->resize(solver.NumberOfSolutions());
            
            for (int i = 0; i < solver.NumberOfSolutions(); i++) {
                const SQPSolution* sol = solver.SolutionPtr(i);
                (*poses)[i].block<3,3>(0,0) = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(sol->r_hat.data());
                (*poses)[i].block<3,1>(0,3) = sol->t;
            }
            return solver.NumberOfSolutions();
        }
        return 0;
    }
    
    // Non-minimal solver (required by RansacLib)
    int NonMinimalSolver(const std::vector<int>& sample, Matrix34d* pose) const {
        const int npts = sample.size();
        if (npts < non_minimal_sample_size_) return 0;
        
        std::vector<Matrix34d> poses;
        int n = MinimalSolver(sample, &poses);
        
        if (n > 0) {
            *pose = poses[0]; // Take first solution
            return 1;
        }
        return 0;
    }
    
    // Evaluate model on point (required by RansacLib)
    double EvaluateModelOnPoint(const Matrix34d& pose, int i) const {
        if (i < 0 || i >= static_cast<int>(points3d_->size())) return std::numeric_limits<double>::max();
        
        const _Point& p3d = (*points3d_)[i];
        const _Projection& p2d = (*points2d_)[i];
        
        // Transform 3D point
        Eigen::Vector3d p3d_eigen(p3d.vector);
        Eigen::Vector3d p_cam = pose.block<3,3>(0,0) * p3d_eigen + pose.block<3,1>(0,3);
        
        // Project to 2D
        if (p_cam(2) <= 0) return std::numeric_limits<double>::max();
        
        double x_proj = p_cam(0) / p_cam(2);
        double y_proj = p_cam(1) / p_cam(2);
        
        // Compute squared error
        double dx = x_proj - p2d.vector(0);
        double dy = y_proj - p2d.vector(1);
        return dx*dx + dy*dy;
    }
    
    // Least squares refinement (optional)
    void LeastSquares(const std::vector<int>& sample, Matrix34d* pose) const {
        NonMinimalSolver(sample, pose);
    }

private:
    const std::vector<_Point>* points3d_;
    const std::vector<_Projection>* points2d_;
    int min_sample_size_;
    int non_minimal_sample_size_;
};

// Unified SQPnP solver class
class UnifiedPnPSolver {
public:
    // Constructor for simple PnP
    template <class Point3D, class Projection2D, typename Pw = double>
    UnifiedPnPSolver(const std::vector<Point3D>& points3d,
                     const std::vector<Projection2D>& points2d,
                     const std::vector<Pw>& weights = std::vector<Pw>(),
                     const SolverParameters& params = SolverParameters())
        : points3d_(), points2d_(), weights_(), solver_params_(params), use_ransac_(false) {
        
        const size_t n = points3d.size();
        if (n != points2d.size() || n < 3) {
            return;
        }
        
        // Convert to internal format
        points3d_.reserve(n);
        points2d_.reserve(n);
        
        for (size_t i = 0; i < n; ++i) {
            points3d_.emplace_back(points3d[i]);
            points2d_.emplace_back(points2d[i]);
        }
        
        if (!weights.empty()) {
            if (n == weights.size()) {
                weights_ = weights;
            }
        } else {
            weights_.resize(n, 1.0);
        }
        
        is_valid_ = true;
    }
    
    // Constructor for robust PnP with RANSAC
    template <class Point3D, class Projection2D, typename Pw = double>
    UnifiedPnPSolver(const std::vector<Point3D>& points3d,
                     const std::vector<Projection2D>& points2d,
                     const RansacParameters& ransac_params,
                     const std::vector<Pw>& weights = std::vector<Pw>(),
                     const SolverParameters& params = SolverParameters())
        : points3d_(), points2d_(), weights_(), solver_params_(params), 
          ransac_params_(ransac_params), use_ransac_(true) {
        
        const size_t n = points3d.size();
        if (n != points2d.size() || n < 3) {
            return;
        }
        
        // Convert to internal format
        points3d_.reserve(n);
        points2d_.reserve(n);
        
        for (size_t i = 0; i < n; ++i) {
            points3d_.emplace_back(points3d[i]);
            points2d_.emplace_back(points2d[i]);
        }
        
        if (!weights.empty()) {
            if (n == weights.size()) {
                weights_ = weights;
            }
        } else {
            weights_.resize(n, 1.0);
        }
        
        is_valid_ = true;
    }
    
    // Solve PnP (simple or robust based on constructor)
    UnifiedPoseResult solve() {
        UnifiedPoseResult result;
        
        if (!is_valid_) {
            return result;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (use_ransac_) {
            result = solveRobust();
        } else {
            result = solveSimple();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        result.execution_time_us = static_cast<double>(duration.count());
        
        return result;
    }
    
    // Check if solver is valid
    bool isValid() const { return is_valid_; }
    
    // Get number of points
    size_t getNumPoints() const { return points3d_.size(); }
    
    // Set RANSAC parameters (for switching to robust mode)
    void setRansacParameters(const RansacParameters& params) {
        ransac_params_ = params;
        use_ransac_ = true;
    }
    
    // Set simple mode (disable RANSAC)
    void setSimpleMode() {
        use_ransac_ = false;
    }

private:
    // Solve simple PnP without RANSAC
    UnifiedPoseResult solveSimple() {
        UnifiedPoseResult result;
        
        PnPSolver solver(points3d_, points2d_, weights_, solver_params_);
        
        if (solver.IsValid()) {
            solver.Solve();
            
            if (solver.NumberOfSolutions() > 0) {
                const SQPSolution* solution = solver.SolutionPtr(0);
                
                // Extract rotation and translation
                result.rotation = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(solution->r_hat.data());
                result.translation = solution->t;
                result.reprojection_error = solver.AverageSquaredProjectionErrors()[0];
                result.num_inliers = static_cast<int>(points3d_.size());
                result.num_outliers = 0;
                result.success = true;
                
                // All points are inliers for simple mode
                result.inlier_indices.resize(points3d_.size());
                for (size_t i = 0; i < points3d_.size(); ++i) {
                    result.inlier_indices[i] = static_cast<int>(i);
                }
            }
        }
        
        return result;
    }
    
    // Solve robust PnP with RANSAC
    UnifiedPoseResult solveRobust() {
        UnifiedPoseResult result;
        
        // Create RANSAC solver
        InternalRansacSolver ransac_solver(&points3d_, &points2d_, 
                                          ransac_params_.min_sample_size, 
                                          ransac_params_.non_minimal_sample_size);
        
        // Set up RANSAC options
        ransac_lib::LORansacOptions options;
        options.min_num_iterations_ = static_cast<uint32_t>(ransac_params_.min_iterations);
        options.max_num_iterations_ = static_cast<uint32_t>(ransac_params_.max_iterations);
        options.squared_inlier_threshold_ = ransac_params_.outlier_threshold * ransac_params_.outlier_threshold;
        options.final_least_squares_ = true;
        
        // Run RANSAC
        ransac_lib::LocallyOptimizedMSAC<Matrix34d, std::vector<Matrix34d>, InternalRansacSolver> lomsac;
        ransac_lib::RansacStatistics ransac_stats;
        
        Matrix34d best_pose;
        int num_inliers = lomsac.EstimateModel(options, ransac_solver, &best_pose, &ransac_stats);
        
        if (num_inliers > 0) {
            // Extract rotation and translation
            result.rotation = best_pose.block<3,3>(0,0);
            result.translation = best_pose.block<3,1>(0,3);
            result.num_inliers = num_inliers;
            result.num_outliers = static_cast<int>(points3d_.size()) - num_inliers;
            result.inlier_indices = ransac_stats.inlier_indices;
            result.success = true;
            
            // Calculate reprojection error using inliers only
            if (!result.inlier_indices.empty()) {
                std::vector<_Point> inlier_points3d;
                std::vector<_Projection> inlier_points2d;
                std::vector<double> inlier_weights;
                
                for (int idx : result.inlier_indices) {
                    inlier_points3d.push_back(points3d_[idx]);
                    inlier_points2d.push_back(points2d_[idx]);
                    inlier_weights.push_back(1.0);
                }
                
                PnPSolver refined_solver(inlier_points3d, inlier_points2d, inlier_weights, solver_params_);
                if (refined_solver.IsValid()) {
                    refined_solver.Solve();
                    if (refined_solver.NumberOfSolutions() > 0) {
                        result.reprojection_error = refined_solver.AverageSquaredProjectionErrors()[0];
                    }
                }
            }
            
            // Fill outlier indices
            result.outlier_indices.clear();
            std::vector<bool> is_inlier(points3d_.size(), false);
            for (int idx : result.inlier_indices) {
                is_inlier[idx] = true;
            }
            for (size_t i = 0; i < points3d_.size(); ++i) {
                if (!is_inlier[i]) {
                    result.outlier_indices.push_back(static_cast<int>(i));
                }
            }
        }
        
        return result;
    }
    
    std::vector<_Point> points3d_;
    std::vector<_Projection> points2d_;
    std::vector<double> weights_;
    SolverParameters solver_params_;
    RansacParameters ransac_params_;
    bool use_ransac_;
    bool is_valid_;
};

// Convenience functions for easy usage

// Simple PnP solver
template <class Point3D, class Projection2D, typename Pw = double>
UnifiedPoseResult solvePnP(const std::vector<Point3D>& points3d,
                          const std::vector<Projection2D>& points2d,
                          const std::vector<Pw>& weights = std::vector<Pw>(),
                          const SolverParameters& params = SolverParameters()) {
    UnifiedPnPSolver solver(points3d, points2d, weights, params);
    return solver.solve();
}

// Robust PnP solver with RANSAC
template <class Point3D, class Projection2D, typename Pw = double>
UnifiedPoseResult solveRobustPnP(const std::vector<Point3D>& points3d,
                                const std::vector<Projection2D>& points2d,
                                const RansacParameters& ransac_params,
                                const std::vector<Pw>& weights = std::vector<Pw>(),
                                const SolverParameters& params = SolverParameters()) {
    UnifiedPnPSolver solver(points3d, points2d, ransac_params, weights, params);
    return solver.solve();
}

// Overloaded version with default RANSAC parameters
template <class Point3D, class Projection2D, typename Pw = double>
UnifiedPoseResult solveRobustPnP(const std::vector<Point3D>& points3d,
                                const std::vector<Projection2D>& points2d,
                                const std::vector<Pw>& weights = std::vector<Pw>(),
                                const SolverParameters& params = SolverParameters()) {
    RansacParameters ransac_params; // Use default parameters
    UnifiedPnPSolver solver(points3d, points2d, ransac_params, weights, params);
    return solver.solve();
}

} // namespace sqpnp

#endif // UNIFIED_SQPnP_H__ 
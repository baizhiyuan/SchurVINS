// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// This file is subject to the terms and conditions defined in the file
// 'LICENSE', which is part of this source code package.

// Modification Note: 
// This file may have been modified by the authors of SchurVINS.
// (All authors of SchurVINS are with PICO department of ByteDance Corporation)

#pragma once

#include <array>
#include <atomic>
#include <iostream>
#include <unordered_map>

#include <glog/logging.h>
#include <svo/common/types.h>

namespace svo {

/// Thread-safe point-ID provider.
class PointIdProvider
{
public:
  PointIdProvider() = delete;

  static int getNewPointId()
  {
    return last_id_.fetch_add(1);
  }

private:
   static std::atomic<int> last_id_;
};

struct KeypointIdentifier
{
  FrameWeakPtr frame;       // 使用弱指针（weak_ptr）来指向关键点所属的帧，避免循环引用
  int frame_id;             // 当前帧的 ID，用于唯一标识帧
  size_t keypoint_index_;   // 关键点在帧中的索引位置
  // 构造函数，通过传入一个指向帧的共享指针 (FramePtr) 以及关键点在帧中的索引来构造 KeypointIdentifier 对象
  KeypointIdentifier(const FramePtr& _frame, const size_t _feature_index);

  // 重载 == 运算符：用于比较两个 KeypointIdentifier 对象是否相等
  // 首先使用 frame.lock() 将弱指针 frame 转换为共享指针，确保指向的帧仍然有效
  inline bool operator ==(const KeypointIdentifier& other) const
  {
    CHECK(frame.lock() && other.frame.lock());  // 用于确保两个对象都指向有效的帧
    // 具体比较条件包括：frame 是否指向相同的帧；frame_id 是否相同；keypoint_index_ 是否相同
    return frame.lock().get() == other.frame.lock().get() &&
        frame_id == other.frame_id && keypoint_index_ == other.keypoint_index_;
  }

    /// \brief Less than operator. Compares first multiframe ID, then camera index,
    ///        then keypoint index.
    bool operator<(const KeypointIdentifier& rhs) const
    {
      if (frame_id == rhs.frame_id)
      {
          return keypoint_index_ < rhs.keypoint_index_;
      }
      return frame_id < rhs.frame_id;
    }
};
using KeypointIdentifierList = std::vector<KeypointIdentifier>;
using Matrix23d = Eigen::Matrix<double, 2, 3>;


/// A 3D point on the surface of the scene.
class Point
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // observation saves frame-id and pointer to feature
  typedef std::vector<KeypointIdentifier> KeypointIdentifierList;

  enum PointType {
    TYPE_EDGELET_SEED,
    TYPE_CORNER_SEED,
    TYPE_EDGELET,
    TYPE_CORNER
  };

  int           id_;                       //!< Unique ID of the point.
  Position      pos_;                      //!< 3d pos of the point in the world coordinate frame.
  KeypointIdentifierList  obs_;                      //!< References to keyframes which observe the point

  bool local_status_ = true;
  LocalFeatureMap local_obs_;  // state id, feature
  StateFactorMap state_factors;

  int64_t register_id_ = -1;
  Eigen::Matrix3d V;
  Eigen::Matrix3d Vinv;
  Eigen::Vector3d gv;
  Matrix6o3d W;
  bool ekf_init = false;
  Eigen::Matrix3d cov;

  Eigen::Vector3d      normal_;                   //!< Surface normal at point.
  Eigen::Matrix2d      normal_information_;       //!< Inverse covariance matrix of normal estimation.
  bool          normal_set_ = false;       //!< Flag whether the surface normal was estimated or not.
  uint64_t      last_published_ts_ = 0;    //!< Timestamp of last publishing.
  std::array<int, 8> last_projected_kf_id_;//!< Flag for the reprojection: don't reproject a pt twice in the same camera
  PointType     type_ = TYPE_CORNER;       //!< Quality of the point.
  int           n_failed_reproj_ = 0;      //!< Number of failed reprojections. Used to assess the quality of the point.
  int           n_succeeded_reproj_ = 0;   //!< Number of succeeded reprojections. Used to assess the quality of the point.
  int           last_structure_optim_ = 0; //!< Timestamp of last point optimization

  // bundle adjustment:
  bool          in_ba_graph_ = false;     //!< Was this point already added to the iSAM bundle adjustment graph?
  int64_t       last_ba_update_ = -1;     //!< Timestamp of last estimate in bundle adjustment.
  static std::atomic_uint64_t global_map_value_version_;

  /// Default constructor.
  Point(const Eigen::Vector3d& pos);

  /// Constructor with id: Only for testing!
  Point(const int id, const Eigen::Vector3d& pos);

  /// Destructor.
  ~Point();

  // no copy
  Point(const Point&) = delete;
  Point& operator=(const Point&) = delete;

  /// check
  bool CheckStatus();

  /// check
  bool CheckLocalStatus();
  bool CheckLocalStatus(const int prev_frame_id0, const int prev_frame_id1, const int curr_framebundle_id);

  /// Remove observation via frame-ID and camera-ID.
  void RemoveLocalObs(const int state_id, const int camera_id);
  
  /// Add a reference to a frame.
  void addObservation(const FramePtr& frame, const size_t feature_index);

  /// Remove observation via frame-ID.
  void removeObservation(int frame_id);

  /// Initialize point normal. The inital estimate will point towards the frame.
  void initNormal();

  /// Get Frame with similar viewpoint.
  bool getCloseViewObs(const Eigen::Vector3d& pos, FramePtr& ref_frame, size_t& ref_feature_index) const;

  /// Get parallax angle of triangulation. Useful to check if point is constrained.
  double getTriangulationParallax() const;

  /// Get frame with seed of this point.
  FramePtr getSeedFrame() const;

  /// Get number of observations.
  inline size_t nRefs() const { return obs_.size(); }

  /// Retriangulate the point from its observations.
  bool triangulateLinear();

  /// update Hessian and Gradient of point based on one observation, using unit plane.
  void updateHessianGradientUnitPlane(
      const Eigen::Ref<BearingVector>& f,
      const Eigen::Vector3d& p_in_f,
      const Eigen::Matrix3d& R_f_w,
      Eigen::Matrix3d& A,
      Eigen::Vector3d& b,
      double& new_chi2);

  /// update Hessian and Gradient of point based on one observation, using unit sphere.
  void updateHessianGradientUnitSphere(
      const Eigen::Ref<BearingVector>& f,
      const Eigen::Vector3d& p_in_f,
      const Eigen::Matrix3d& R_f_w,
      Eigen::Matrix3d& A,
      Eigen::Vector3d& b,
      double& new_chi2);

  void EkfInit();

  void EkfUpdate(const Eigen::VectorXd& dx, double obs_std);

  void EkfUpdate(const Eigen::VectorXd& dx, const double obs_std, const double focal_length, const int min_idx,
                  const int max_idx, const double huberA, const double huberB);
                  
  /// Optimize point position through minimizing the reprojection error.
  void optimize(const size_t n_iter, bool using_bearing_vector=false);

  /// Print infos about the point.
  void print(const std::string& s = "Point:") const;

  /// Return unique point identifier.
  inline int id() const { return id_; }

  /// 3D position of point in world frame.
  inline const Eigen::Vector3d& pos() const { return pos_; }

  /// Return type of point.
  inline const PointType& type() const { return type_; }

  /// Jacobian of point projection on unit plane (focal length = 1) in frame (f).
  inline static void jacobian_xyz2uv(
      const Eigen::Vector3d& p_in_f,
      const Eigen::Matrix3d& R_f_w,
      svo::Matrix23d& point_jac)
  {
    const double z_inv = 1.0/p_in_f[2];
    const double z_inv_sq = z_inv*z_inv;
    point_jac(0, 0) = z_inv;
    point_jac(0, 1) = 0.0;
    point_jac(0, 2) = -p_in_f[0] * z_inv_sq;
    point_jac(1, 0) = 0.0;
    point_jac(1, 1) = z_inv;
    point_jac(1, 2) = -p_in_f[1] * z_inv_sq;
    point_jac = - point_jac * R_f_w;
  }

  /// Jacobian of point to unit bearing vector.
  inline static void jacobian_xyz2f(
      const Eigen::Vector3d& p_in_f,
      const Eigen::Matrix3d& R_f_w,
      Eigen::Matrix3d& point_jac)
  {
    Eigen::Matrix3d J_normalize;
    double x2 = p_in_f[0]*p_in_f[0];
    double y2 = p_in_f[1]*p_in_f[1];
    double z2 = p_in_f[2]*p_in_f[2];
    double xy = p_in_f[0]*p_in_f[1];
    double yz = p_in_f[1]*p_in_f[2];
    double zx = p_in_f[2]*p_in_f[0];
    J_normalize << y2+z2, -xy, -zx,
        -xy, x2+z2, -yz,
        -zx, -yz, x2+y2;
    J_normalize *= 1.0 / std::pow(x2+y2+z2, 1.5);
    point_jac = (-1.0) * J_normalize * R_f_w;
  }
};

} // namespace svo

namespace std {

inline ostream& operator <<(ostream& stream, const svo::KeypointIdentifier& id)
{
  stream << "(" << id.frame_id << ", " << id.keypoint_index_ << ")";
  return stream;
}

}  // namespace std

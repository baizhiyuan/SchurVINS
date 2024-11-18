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

#include <svo/frame_handler_stereo.h>
#include <svo/map.h>
#include <svo/common/frame.h>
#include <svo/common/point.h>
#include <svo/pose_optimizer.h>
#include <svo/img_align/sparse_img_align.h>
#include <svo/direct/depth_filter.h>
#include <svo/stereo_triangulation.h>
#include <svo/direct/feature_detection.h>
#include <svo/direct/feature_detection_utils.h>
#include <svo/reprojector.h>
#include <svo/initialization.h>
#include <vikit/performance_monitor.h>

namespace svo {

FrameHandlerStereo::FrameHandlerStereo(
    const BaseOptions& base_options,
    const DepthFilterOptions& depth_filter_options,
    const DetectorOptions& feature_detector_options,
    const InitializationOptions& init_options,
    const StereoTriangulationOptions& stereo_options,
    const ReprojectorOptions& reprojector_options,
    const FeatureTrackerOptions& tracker_options,
    const CameraBundle::Ptr& stereo_camera)
  : FrameHandlerBase(
      base_options, reprojector_options, depth_filter_options,
      feature_detector_options, init_options, tracker_options, stereo_camera)
{
  // init initializer
  stereo_triangulation_.reset(
        new StereoTriangulation(
          stereo_options,
          feature_detection_utils::makeDetector(
            feature_detector_options, cams_->getCameraShared(0))));
}

UpdateResult FrameHandlerStereo::processFrameBundle()
{
  UpdateResult res = UpdateResult::kFailure;
  if(stage_ == Stage::kTracking)
    res = processFrame();
  else if(stage_ == Stage::kInitializing)
    res = processFirstFrame();
  return res;
}

void FrameHandlerStereo::addImages(
    const cv::Mat& img_left,
    const cv::Mat& img_right,
    const uint64_t timestamp)
{
  // TODO: deprecated
  addImageBundle({img_left, img_right}, timestamp);
}

UpdateResult FrameHandlerStereo::processFirstFrame()
{
  schurvinsForward();
  if(initializer_->addFrameBundle(new_frames_) == InitResult::kFailure)
  {
    SVO_ERROR_STREAM("Initialization failed. Not enough triangulated points.");
    return UpdateResult::kDefault;
  }

  new_frames_->at(0)->setKeyframe();
  map_->addKeyframe(new_frames_->at(0),
                    bundle_adjustment_type_==BundleAdjustmentType::kCeres);
  new_frames_->at(1)->setKeyframe();
  map_->addKeyframe(new_frames_->at(1),
                    bundle_adjustment_type_==BundleAdjustmentType::kCeres);

  frame_utils::getSceneDepth(new_frames_->at(0), depth_median_, depth_min_, depth_max_);
  depth_filter_->addKeyframe(new_frames_->at(0), depth_median_, 0.5*depth_min_, depth_median_*1.5);
  schurvinsBackward();

  LOG(INFO) << std::fixed << "InitState: " << new_frames_->getMinTimestampSeconds()
            << ", quat: " << new_frames_->get_T_W_B().getEigenQuaternion().w() << ", "
            << new_frames_->get_T_W_B().getEigenQuaternion().x() << ", "
            << new_frames_->get_T_W_B().getEigenQuaternion().y() << ", "
            << new_frames_->get_T_W_B().getEigenQuaternion().z() << ", "
            << "pos: " << new_frames_->get_T_W_B().getPosition()[0] << ", "
            << new_frames_->get_T_W_B().getPosition()[1] << ", " << new_frames_->get_T_W_B().getPosition()[2];  

  SVO_INFO_STREAM("Init: Selected first frame.");
  stage_ = Stage::kTracking;
  tracking_quality_ = TrackingQuality::kGood;
  return UpdateResult::kKeyframe;
}

UpdateResult FrameHandlerStereo::processFrame()
{
  // ---------------------------------------------------------------------------
  // tracking

  // STEP 1: Sparse Image Align
  size_t n_tracked_features = 0;
  sparseImageAlignment();

  // STEP 2: Map Reprojection & Feature Align
  n_tracked_features = projectMapInFrame();
  if(n_tracked_features < options_.quality_min_fts)
  {
    return makeKeyframe(); // force stereo triangulation to recover
  }

  // STEP 3: Pose & Structure Optimization
  if(bundle_adjustment_type_!=BundleAdjustmentType::kCeres)
  {
    n_tracked_features = optimizePose();
    if(n_tracked_features < options_.quality_min_fts)
    {
      return makeKeyframe(); // force stereo triangulation to recover
    }
    optimizeStructure(new_frames_, options_.structure_optimization_max_pts, 5);
  }
  // return if tracking bad
  setTrackingQuality(n_tracked_features);
  if(tracking_quality_ == TrackingQuality::kInsufficient)
  {
    return makeKeyframe(); // force stereo triangulation to recover
  }

  // ---------------------------------------------------------------------------
  // select keyframe
  frame_utils::getSceneDepth(new_frames_->at(0), depth_median_, depth_min_, depth_max_);
  if(!need_new_kf_(new_frames_->at(0)->T_f_w_))
  {
    for(size_t i=0; i<new_frames_->size(); ++i)
      depth_filter_->updateSeeds(overlap_kfs_.at(i), new_frames_->at(i));
    return UpdateResult::kDefault;
  }
  SVO_DEBUG_STREAM("New keyframe selected.");
  return makeKeyframe();
}

UpdateResult FrameHandlerStereo::makeKeyframe()
{
  static size_t kf_counter = 0; // 用于跟踪关键帧的数量
  // 计算得出的当前关键帧的 ID，使用模运算以确保在多摄像头系统中轮流选择相机
  const size_t kf_id = kf_counter++ % cams_->numCameras();
  // 下一个摄像头的 ID，kf_id 和 other_id 不相等，确保不同相机之间的交替
  const size_t other_id = kf_counter % cams_->numCameras();
  CHECK(kf_id != other_id);

  // ---------------------------------------------------------------------------
  // add extra features when num tracked is critically low!
  if(new_frames_->numLandmarks() < options_.kfselect_numkfs_lower_thresh)
  {
    // 特征检测器可以避免在已经被其他特征点或固定地标占据的栅格中重复检测特征，
    // 从而提高特征点的分布均匀性和检测效率
    setDetectorOccupiedCells(0, stereo_triangulation_->feature_detector_);
    // 设置关键帧，选择关键特征点
    new_frames_->at(other_id)->setKeyframe();
    // 添加关键帧
    map_->addKeyframe(new_frames_->at(other_id),
                      bundle_adjustment_type_==BundleAdjustmentType::kCeres);
    // 将当前帧中的种子点（Seeds）升级为三维空间中的特征点（Features）
    upgradeSeedsToFeatures(new_frames_->at(other_id));
    // 从两个立体相机图像中恢复特征点的三维位置
    stereo_triangulation_->compute(new_frames_->at(0), new_frames_->at(1));
  }

  // ---------------------------------------------------------------------------
  // new keyframe selected
  // 设置当前相机 ID 为 kf_id 的新帧为关键帧
  new_frames_->at(kf_id)->setKeyframe();
  // 将此关键帧添加到地图中
  map_->addKeyframe(new_frames_->at(kf_id),
                    bundle_adjustment_type_==BundleAdjustmentType::kCeres);
  
  // 使用 upgradeSeedsToFeatures() 将该帧的种子点提升为特征点
  upgradeSeedsToFeatures(new_frames_->at(kf_id));

  // init new depth-filters, set feature-detection grid-cells occupied that
  // already have a feature
  // 初始化新的深度滤波器，并将已经存在的特征点标记为占用
  {
    // 获取深度滤波器的互斥锁，保护对特征检测器的访问，避免多线程竞争
    DepthFilter::ulock_t lock(depth_filter_->feature_detector_mut_);
    // 标记当前关键帧中已经包含特征的网格单元，以防止在相同位置重复检测
    setDetectorOccupiedCells(kf_id, depth_filter_->feature_detector_);
  } // release lock 结束作用域后自动释放互斥锁
  // 向深度滤波器添加新的关键帧，并初始化深度滤波器的参数
  depth_filter_->addKeyframe(
        new_frames_->at(kf_id), depth_median_, 0.5*depth_min_, depth_median_*1.5);
  // 更新深度滤波器中的种子点，使用的参考关键帧为 overlap_kfs_（当前帧的重叠关键帧）
  depth_filter_->updateSeeds(overlap_kfs_.at(0), new_frames_->at(0));
  depth_filter_->updateSeeds(overlap_kfs_.at(1), new_frames_->at(1));

  // TEST
  // {
  if(options_.update_seeds_with_old_keyframes)
  {
    // 如果配置中允许使用旧关键帧来更新种子，则调用 updateSeeds() 函数用之前的关键帧来进一步更新当前帧中的种子点。
    // 这一步能增加深度估计的准确性和稳定性。
    // 如果关键帧数量超过限制，移除最远的
    depth_filter_->updateSeeds({ new_frames_->at(0) }, last_frames_->at(0));
    depth_filter_->updateSeeds({ new_frames_->at(0) }, last_frames_->at(1));
    depth_filter_->updateSeeds({ new_frames_->at(1) }, last_frames_->at(0));
    depth_filter_->updateSeeds({ new_frames_->at(1) }, last_frames_->at(1));
    for(const FramePtr& old_keyframe : overlap_kfs_.at(0))
    {
      depth_filter_->updateSeeds({ new_frames_->at(0) }, old_keyframe);
      depth_filter_->updateSeeds({ new_frames_->at(1) }, old_keyframe);
    }
  }
  // }

  // if limited number of keyframes, remove the one furthest apart
  // 如果关键帧数量超过限制，移除最远的
  while(map_->size() > options_.max_n_kfs && options_.max_n_kfs > 2)
  {
    if(bundle_adjustment_type_==BundleAdjustmentType::kCeres)
    {
      // deal differently with map for ceres backend
      map_->removeOldestKeyframe();
    }
    else
    {
      FramePtr furthest_frame =
          map_->getFurthestKeyframe(new_frames_->at(kf_id)->pos());
      map_->removeKeyframe(furthest_frame->id());
    }
  }
  // 返回结果
  return UpdateResult::kKeyframe;
}

void FrameHandlerStereo::resetAll()
{
  backend_scale_initialized_ = true;
  resetVisionFrontendCommon();
}

} // namespace svo

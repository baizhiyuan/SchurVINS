// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).

#include <numeric>
#include <svo/direct/matcher.h>
#include <svo/common/point.h>
#include <svo/common/frame.h>
#include <svo/stereo_triangulation.h>
#include <svo/direct/feature_detection.h>
#include <svo/tracker/feature_tracker.h>

namespace svo {

StereoTriangulation::StereoTriangulation(
    const StereoTriangulationOptions& options,
    const AbstractDetector::Ptr& feature_detector)
  : options_(options)
  , feature_detector_(feature_detector)
{ ; }

void StereoTriangulation::compute(const FramePtr& frame0,
                                  const FramePtr& frame1)
{
  // Check if there is something to do
  // 如果 frame0 中已有的地标点数量超过或等于设定的目标数量，则不再需要进行三角化，函数返回
  if(frame0->numLandmarks() >= options_.triangulate_n_features)
  {
    VLOG(5) << "Calling stereo triangulation with sufficient number of features"
        << " has no effect.";
    return;
  }

  // Detect new features.
  Keypoints new_px;
  Levels new_levels;
  Scores new_scores;
  Gradients new_grads;
  FeatureTypes new_types;
  const size_t max_n_features = feature_detector_->grid_.size();
  // 调用特征检测器 feature_detector_ 来检测新的特征点，并将结果存储在相应的变量中
  feature_detector_->detect(
        frame0->img_pyr_, frame0->getMask(), max_n_features, new_px,
        new_scores, new_levels, new_grads, new_types);
  if(new_px.cols() == 0)
  {
    SVO_ERROR_STREAM("Stereo Triangulation: No features detected.");
    return;
  }

  // 计算并归一化特征点的方向向量
  // 这些方向向量表示相机光心到特征点的单位向量，后续会用来进行三角化
  // Compute and normalize all bearing vectors.
  Bearings new_f;
  frame_utils::computeNormalizedBearingVectors(new_px, *frame0->cam(), &new_f);

  // 将新特征点添加到 frame0 中
  // Add features to first frame.
  const long n_old = static_cast<long>(frame0->numFeatures());
  const long n_new = new_px.cols();
  // 扩展 frame0 中的特征点存储空间
  frame0->resizeFeatureStorage(
        frame0->num_features_ + static_cast<size_t>(n_new));
  frame0->px_vec_.middleCols(n_old, n_new) = new_px;
  frame0->f_vec_.middleCols(n_old, n_new) = new_f;
  frame0->grad_vec_.middleCols(n_old, n_new) = new_grads;
  frame0->score_vec_.segment(n_old, n_new) = new_scores;
  frame0->level_vec_.segment(n_old, n_new) = new_levels;
  frame0->num_features_ += static_cast<size_t>(n_new);
  frame0->type_vec_.insert(
        frame0->type_vec_.begin()+n_old, new_types.cbegin(), new_types.cend());

  // 对新特征点的索引进行随机打乱，优先选择角点
  // We only want a limited number of features. Therefore, we create a random
  // vector of indices that we will process.
  std::vector<size_t> indices(static_cast<size_t>(n_new));
  std::iota(indices.begin(), indices.end(), n_old);
  // 统计新特征点中角点（kCorner）的数量
  long n_corners = std::count_if(
        new_types.begin(), new_types.end(),
        [](const FeatureType& t) { return t==FeatureType::kCorner; });

  // 对这些索引进行随机打乱，但优先处理角点（将角点的索引与其他点的索引分别打乱）
  // shuffle twice before we prefer corners!
  std::random_shuffle(indices.begin(), indices.begin()+n_corners);
  std::random_shuffle(indices.begin()+n_corners, indices.end());

  // 初始化新的三维点（种子点）并匹配特征点
  // now for all maximum corners, initialize a new seed
  size_t n_succeded = 0, n_failed = 0;
  const size_t n_desired =
      options_.triangulate_n_features - frame0->numLandmarks();
  //note: we checked already at start that n_desired will be larger than 0

  // 计算还需要多少特征点（n_desired），并为 frame1 预留存储空间
  // reserve space for features in second frame
  if(frame1->num_features_ + n_desired > frame1->landmark_vec_.size())
  {
    frame1->resizeFeatureStorage(frame1->num_features_ + n_desired);
  }

  // 初始化匹配器 matcher，设置匹配器的一些选项（如最大极线搜索步数、亚像素匹配等）
  Matcher matcher;
  matcher.options_.max_epi_search_steps = 500;
  matcher.options_.subpix_refinement = true;
  // 计算 frame1 和 frame0 之间的相对变换 T_f1f0，用于后续匹配过程中坐标系之间的转换。
  const Transformation T_f1f0 = frame1->T_cam_body_*frame0->T_body_cam_;
  // 遍历新特征点的索引并使用匹配器进行极线匹配
  // 通过 findEpipolarMatchDirect() 方法在 frame1 中查找与 frame0 中特征点对应的匹配。
  for(const size_t &i_ref : indices)
  {
    matcher.options_.align_1d = isEdgelet(frame0->type_vec_[i_ref]); // TODO(cfo): check effect
    FloatType depth = 0.0;
    FeatureWrapper ref_ftr = frame0->getFeatureWrapper(i_ref);
    Matcher::MatchResult res =
            matcher.findEpipolarMatchDirect(
                *frame0, *frame1, T_f1f0, ref_ftr, options_.mean_depth_inv,
                options_.min_depth_inv, options_.max_depth_inv, depth);

    // 成功匹配时添加新的三维点
    if(res == Matcher::MatchResult::kSuccess)
    {
      // 根据匹配深度计算三维点在世界坐标系中的位置
      const Position xyz_world = frame0->T_world_cam()
          * (frame0->f_vec_.col(static_cast<int>(i_ref)) * depth);

      // 创建一个新的 Point 对象来表示该三维点，并将其添加到 frame0 和 frame1 的地标列表中
      // 更新 frame0 和 frame1 中相关特征点的属性，例如跟踪 ID、方向向量、梯度等
      PointPtr new_point(new Point(xyz_world));
      frame0->landmark_vec_[i_ref] = new_point;
      frame0->track_id_vec_(static_cast<int>(i_ref)) = new_point->id();
      new_point->addObservation(frame0, i_ref);

      const int i_cur = static_cast<int>(frame1->num_features_);
      frame1->type_vec_[static_cast<size_t>(i_cur)] = ref_ftr.type;
      frame1->level_vec_[i_cur] = ref_ftr.level;
      frame1->px_vec_.col(i_cur) = matcher.px_cur_;
      frame1->f_vec_.col(i_cur) = matcher.f_cur_;
      frame1->score_vec_[i_cur] = ref_ftr.score;
      GradientVector g = matcher.A_cur_ref_*ref_ftr.grad;
      frame1->grad_vec_.col(i_cur) = g.normalized();
      frame1->landmark_vec_[static_cast<size_t>(i_cur)] = new_point;
      frame1->track_id_vec_(i_cur) = new_point->id();
      new_point->addObservation(frame1, static_cast<size_t>(i_cur));
      frame1->num_features_++;
      ++n_succeded;
    }
    else
    {
      ++n_failed;
    }
    // 记录成功匹配和失败匹配的数量，并在成功匹配达到 n_desired 时提前退出循环
    if(n_succeded >= n_desired)
      break;
  }
  VLOG(20) << "Stereo: Triangulated " << n_succeded << " features,"
           << n_failed << " failed.";
}

} // namespace svo


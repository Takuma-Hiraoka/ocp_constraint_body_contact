#include <ament_index_cpp/get_package_share_directory.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Eigenvalues>
#include <assimp_eigen/assimp_eigen.h>
#include <mujoco/mujoco.h>
#include <mujoco_viewer/mujoco_viewer.h>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/fwd.hpp>

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocs2_core/penalties/penalties/QuadraticPenalty.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_mpc/SystemObservation.h>
#include <ocp_constraint/contact_fix_constraint.h>
#include <ocp_constraint/joint_limit_constraint.h>
#include <ocp_constraint/joint_torque_cost.h>
#include <ocp_constraint/penalties/piece_wise_polynominal_barrier_penalty.h>
#include <ocp_constraint/swing_position_constraint.h>
#include <ocp_constraint_body_contact/assumed_surface_contact_constraint.h>
#include <ocp_constraint_body_contact/contact_point_search_constraint.h>
#include <ocp_constraint_body_contact/cop_center_constraint.h>
#include <ocp_constraint_body_contact/cop_moment_constraint.h>
#include <ocp_solver/common/quadratic_state_cost.h>
#include <ocp_solver/common/quadratic_state_input_cost.h>
#include <ocp_solver/ocp_interface.h>
#include <ocp_solver/pinocchio/pinocchio_frame_dynamics.h>
#include <ocp_solver/solver/dynamics_helper_functions.h>
#include <point_mesh_projector/point_mesh_projector.h>
#include <trajectory_logger/trajectory_logger.h>

namespace {

struct ContactTrajectoryVisualization {
  size_t contactIndex = 0;
  std::array<float, 4> currentColor = {0.1f, 0.3f, 1.0f, 1.0f};
  std::array<float, 4> targetColor = {1.0f, 0.1f, 0.1f, 1.0f};
  std::array<float, 4> errorColor = {1.0f, 0.9f, 0.1f, 1.0f};
};

struct AssumedSurfaceVisualization {
  size_t contactIndex = 0;
  const ocp_constraint_body_contact::AssumedSurfaceContactConstraint* constraint = nullptr;
};

ocs2::vector_t makeInitialState(const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter) {
  ocs2::vector_t initialState = ocs2::vector_t::Zero(stateConverter.getStateDim());
  initialState.head<7>() << 0.0, 0.0, 0.65, 0.0, 0.0, 0.0, 1.0;

  auto setJoint = [&](const std::string& jointName, double value) {
    initialState[stateConverter.getJointStartindex() + stateConverter.getJointIndex(jointName)] = value;
  };
  setJoint("left_hip_pitch_joint", -0.6);
  setJoint("left_hip_roll_joint", 0.0);
  setJoint("left_hip_yaw_joint", 0.0);
  setJoint("left_knee_joint", 1.4);
  setJoint("left_ankle_pitch_joint", -0.8);
  setJoint("left_ankle_roll_joint", 0.0);
  setJoint("right_hip_pitch_joint", -0.6);
  setJoint("right_hip_roll_joint", 0.0);
  setJoint("right_hip_yaw_joint", 0.0);
  setJoint("right_knee_joint", 1.4);
  setJoint("right_ankle_pitch_joint", -0.8);
  setJoint("right_ankle_roll_joint", 0.0);
  setJoint("waist_yaw_joint", 0.0);
  setJoint("waist_roll_joint", 0.0);
  setJoint("waist_pitch_joint", 0.0);
  setJoint("left_shoulder_pitch_joint", 0.0);
  setJoint("left_shoulder_roll_joint", 0.0);
  setJoint("left_shoulder_yaw_joint", 0.0);
  setJoint("left_elbow_joint", 0.0);
  setJoint("right_shoulder_pitch_joint", 0.0);
  setJoint("right_shoulder_roll_joint", 0.0);
  setJoint("right_shoulder_yaw_joint", 0.0);
  setJoint("right_elbow_joint", 0.0);
  setJoint("right_wrist_roll_joint", 0.0);
  setJoint("right_wrist_pitch_joint", 0.0);
  setJoint("right_wrist_yaw_joint", 0.0);
  return initialState;
}

std::vector<double> toMujocoQ(const ocs2::vector_t& state, int nq) {
  std::vector<double> q(36, 0.0);
  size_t offset = 0;
  for (int j = 0; j < nq; ++j) {
    q[j + offset] = state[j];
    if (j == 24) {
      q[j + offset + 1] = 0.0;
      q[j + offset + 2] = 0.0;
      q[j + offset + 3] = 0.0;
      offset += 3;
    }
  }
  q[3] = state[6];
  q[4] = state[3];
  q[5] = state[4];
  q[6] = state[5];
  return q;
}

void addContactFix(ocp_solver::OCPInterface& interface, size_t contactIndex, const std::string& name) {
  auto frameDynamics = std::make_unique<ocp_solver::PinocchioFrameDynamics>(interface.getStateConverter(), contactIndex);

  ocp_constraint::PositionConstraint::Config hardConfig;
  hardConfig.Aa = ocs2::matrix_t::Identity(6, 6);
  interface.getOptimalControlProblem().equalityConstraintPtr->add(
      name, std::make_unique<ocp_constraint::ContactFixConstraint>(
                *interface.getReferenceManagerPtr(), *frameDynamics, 6, hardConfig));

  ocp_constraint::PositionConstraint::Config softConfig;
  softConfig.Ax = ocs2::matrix_t::Zero(3, 6);
  softConfig.Ax.block(0, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3) * 100;
  interface.getOptimalControlProblem().softConstraintPtr->add(
      name + "_orientation",
      std::make_unique<ocs2::StateInputSoftConstraint>(
          std::make_unique<ocp_constraint::ContactFixConstraint>(
              *interface.getReferenceManagerPtr(), *frameDynamics, 3, softConfig),
          std::make_unique<ocs2::QuadraticPenalty>(1.0)));
}

std::array<mjtNum, 3> toMujocoPosition(const Eigen::Vector3d& position) {
  return {position.x(), position.y(), position.z()};
}

void appendAssumedSurfaceEllipse(mujoco_viewer::Viewer& viewer,
                                 const pinocchio::SE3& contactPose,
                                 const ocp_constraint_body_contact::AssumedSurfaceContactConstraint::SurfaceGeometry& surfaceGeometry) {
  const Eigen::Vector3d normal = surfaceGeometry.normalInContactFrame.normalized();
  const Eigen::Vector3d tangent0 = normal.unitOrthogonal();
  const Eigen::Vector3d tangent1 = normal.cross(tangent0).normalized();
  Eigen::Matrix<double, 3, 2> tangentBasis;
  tangentBasis.col(0) = tangent0;
  tangentBasis.col(1) = tangent1;

  const Eigen::Matrix2d tangentMetric =
      tangentBasis.transpose() * surfaceGeometry.ellipseMetric * tangentBasis;
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(tangentMetric);
  if (solver.info() != Eigen::Success) return;

  const Eigen::Vector2d eigenvalues = solver.eigenvalues().cwiseMax(1e-12);
  const Eigen::Matrix2d eigenvectors = solver.eigenvectors();

  constexpr int segments = 96;
  constexpr double twoPi = 6.28318530717958647692;
  const std::array<float, 4> ellipseColor = {0.1f, 0.9f, 0.25f, 1.0f};
  Eigen::Vector3d previousWorld = Eigen::Vector3d::Zero();
  for (int i = 0; i <= segments; ++i) {
    const double theta = twoPi * static_cast<double>(i) / static_cast<double>(segments);
    Eigen::Vector2d ellipsePoint;
    ellipsePoint << std::cos(theta) / std::sqrt(eigenvalues[0]),
                    std::sin(theta) / std::sqrt(eigenvalues[1]);
    const Eigen::Vector3d pointLocal =
        surfaceGeometry.geometricCenterInContactPlane + tangentBasis * (eigenvectors * ellipsePoint);
    const Eigen::Vector3d pointWorld = contactPose.translation() + contactPose.rotation() * pointLocal;
    if (i > 0) {
      mujoco_viewer::appendLine(viewer, toMujocoPosition(previousWorld), toMujocoPosition(pointWorld), 3.0, ellipseColor);
    }
    previousWorld = pointWorld;
  }
}

Eigen::Matrix<double, 6, 1> makeCopCenteredWrench(
    const pinocchio::SE3& contactPose,
    const ocp_constraint_body_contact::AssumedSurfaceContactConstraint::SurfaceGeometry& surfaceGeometry,
    double normalForce) {
  const Eigen::Vector3d normal = surfaceGeometry.normalInContactFrame.normalized();
  const Eigen::Vector3d center = surfaceGeometry.geometricCenterInContactPlane;

  Eigen::Matrix<double, 6, 1> wrench = Eigen::Matrix<double, 6, 1>::Zero();
  const Eigen::Vector3d localForce = normalForce * normal;
  const Eigen::Vector3d localMoment = -normalForce * normal.cross(center);
  wrench.head<3>() = contactPose.rotation() * localForce;
  wrench.tail<3>() = contactPose.rotation() * localMoment;
  return wrench;
}

void setViewerQ(mujoco_viewer::Viewer& viewer, const ocs2::vector_t& state, int nq) {
  const std::vector<double> viewerQ = toMujocoQ(state, nq);
  for (size_t i = 0; i < viewerQ.size() && i < static_cast<size_t>(viewer.model()->nq); ++i) {
    viewer.data()->qpos[i] = viewerQ[i];
  }
  mj_forward(viewer.model(), viewer.data());
}

void updatePinocchioKinematics(ocs2::PinocchioInterface& pinocchioInterface, const ocs2::vector_t& state) {
  pinocchio::forwardKinematics(pinocchioInterface.getModel(), pinocchioInterface.getData(),
                               state.head(pinocchioInterface.getModel().nq));
  pinocchio::updateFramePlacements(pinocchioInterface.getModel(), pinocchioInterface.getData());
}

bool findContactTarget(const std::vector<std::pair<ocp_solver::ContactCandidateIndex, pinocchio::SE3>>& contacts,
                       size_t contactIndex,
                       pinocchio::SE3& targetPose) {
  for (const auto& contact : contacts) {
    if (contact.first == contactIndex) {
      targetPose = contact.second;
      return true;
    }
  }
  return false;
}

void appendContactTrajectoryVisualization(
    mujoco_viewer::Viewer& viewer,
    ocs2::PinocchioInterface& pinocchioInterface,
    const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
    const ocs2::vector_t& state,
    const std::vector<std::pair<ocp_solver::ContactCandidateIndex, pinocchio::SE3>>& targetContacts,
    const std::vector<ContactTrajectoryVisualization>& visualizations) {
  const std::array<float, 4> parentColor = {0.75f, 0.75f, 0.75f, 1.0f};
  for (const ContactTrajectoryVisualization& visualization : visualizations) {
    const auto contactCandidate = stateConverter.getContactCandidate(state, visualization.contactIndex);
    const pinocchio::SE3 currentPose =
        ocp_solver::getContactCandidatePlacement(pinocchioInterface, contactCandidate);
    mujoco_viewer::appendSphere(viewer, toMujocoPosition(currentPose.translation()), 0.013, visualization.currentColor);

    if (contactCandidate.searchContactPoint) {
      const Eigen::Vector3d parentPosition =
          pinocchioInterface.getData().oMi[contactCandidate.parentJointIndex].translation();
      mujoco_viewer::appendSphere(viewer, toMujocoPosition(parentPosition), 0.008, parentColor);
      mujoco_viewer::appendLine(viewer, toMujocoPosition(parentPosition), toMujocoPosition(currentPose.translation()), 1.5, parentColor);
    }

    pinocchio::SE3 targetPose = pinocchio::SE3::Identity();
    if (findContactTarget(targetContacts, visualization.contactIndex, targetPose)) {
      mujoco_viewer::appendSphere(viewer, toMujocoPosition(targetPose.translation()), 0.018, visualization.targetColor);
      mujoco_viewer::appendLine(viewer, toMujocoPosition(currentPose.translation()), toMujocoPosition(targetPose.translation()), 3.0, visualization.errorColor);
    }
  }
}

void appendAssumedSurfaceVisualization(
    mujoco_viewer::Viewer& viewer,
    ocs2::PinocchioInterface& pinocchioInterface,
    const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
    ocs2::scalar_t time,
    const ocs2::vector_t& state,
    const ocs2::vector_t& input,
    const std::vector<AssumedSurfaceVisualization>& visualizations) {
  const std::array<float, 4> centerColor = {1.0f, 0.9f, 0.1f, 1.0f};
  const std::array<float, 4> copColor = {1.0f, 0.05f, 0.05f, 1.0f};
  const std::array<float, 4> normalColor = {0.95f, 0.95f, 0.95f, 1.0f};

  for (const AssumedSurfaceVisualization& visualization : visualizations) {
    if (!visualization.constraint) continue;

    const pinocchio::SE3 contactPose =
        ocp_solver::getContactCandidatePlacement(pinocchioInterface,
                                                 stateConverter.getContactCandidate(state, visualization.contactIndex));
    const Eigen::Matrix<ocs2::scalar_t, 6, 1> wrench =
        stateConverter.getContactWrench(input, visualization.contactIndex);
    const auto surfaceGeometry =
        visualization.constraint->getSurfaceGeometry(time, state, pinocchioInterface);
    const Eigen::Vector3d centerLocal =
        surfaceGeometry.geometricCenterInContactPlane;
    const Eigen::Vector3d centerWorld =
        contactPose.translation() + contactPose.rotation() * centerLocal;
    const Eigen::Vector3d copLocal =
        visualization.constraint->computePressureCenterInContactFrame(contactPose.rotation(), wrench,
                                                                      surfaceGeometry.normalInContactFrame);
    const Eigen::Vector3d copWorld = contactPose.translation() + contactPose.rotation() * copLocal;
    const Eigen::Vector3d normalWorld =
        contactPose.rotation() * surfaceGeometry.normalInContactFrame.normalized();

    appendAssumedSurfaceEllipse(viewer, contactPose, surfaceGeometry);
    mujoco_viewer::appendSphere(viewer, toMujocoPosition(centerWorld), 0.01, centerColor);
    mujoco_viewer::appendSphere(viewer, toMujocoPosition(copWorld), 0.014, copColor);
    mujoco_viewer::appendLine(viewer, toMujocoPosition(contactPose.translation()),
                              toMujocoPosition(contactPose.translation() + 0.08 * normalWorld), 2.0, normalColor);
  }
}

void visualizeOptimizationTrajectory(
    const std::string& modelPath,
    const ocs2::scalar_array_t& timeTrajectory,
    const ocs2::vector_array_t& stateTrajectory,
    const ocs2::vector_array_t& inputTrajectory,
    ocs2::PinocchioInterface& pinocchioInterface,
    const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
    const ocp_solver::SwitchedModelReferenceManager& referenceManager,
    const std::vector<ContactTrajectoryVisualization>& contactVisualizations,
    const std::vector<AssumedSurfaceVisualization>& assumedSurfaceVisualizations) {
  if (timeTrajectory.empty() || stateTrajectory.empty() || inputTrajectory.empty()) {
    return;
  }

  mujoco_viewer::Viewer viewer;
  viewer.viewModel(modelPath);
  mujoco_viewer::setRobotTransparency(viewer, 0.28f);

  size_t trajectoryIndex = 0;
  auto lastFrameTime = std::chrono::steady_clock::now();
  constexpr auto frameDuration = std::chrono::milliseconds(80);

  while (viewer.isOpen()) {
    const auto now = std::chrono::steady_clock::now();
    if (now - lastFrameTime >= frameDuration) {
      trajectoryIndex = (trajectoryIndex + 1) % stateTrajectory.size();
      lastFrameTime = now;
    }

    const ocs2::vector_t& state = stateTrajectory[trajectoryIndex];
    const ocs2::vector_t& input = inputTrajectory[std::min(trajectoryIndex, inputTrajectory.size() - 1)];
    updatePinocchioKinematics(pinocchioInterface, state);
    setViewerQ(viewer, state, pinocchioInterface.getModel().nq);
    viewer.updateScene();
    appendContactTrajectoryVisualization(viewer, pinocchioInterface, stateConverter, state,
                                         referenceManager.getContacts(timeTrajectory[trajectoryIndex]),
                                         contactVisualizations);
    appendAssumedSurfaceVisualization(viewer, pinocchioInterface, stateConverter,
                                      timeTrajectory[trajectoryIndex], state, input,
                                      assumedSurfaceVisualizations);
    viewer.drawScene();
    viewer.pollEvents();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string packageShare = ament_index_cpp::get_package_share_directory("ocp_constraint_body_contact_sample");
  const std::string robotDir = packageShare + "/unitree_ros/robots/g1_description";
  const std::string urdfFile = robotDir + "/g1_29dof.urdf";
  const std::string mujocoModelFile = robotDir + "/g1_29dof.xml";
  const std::string rhContactMeshFile = robotDir + "/meshes/right_rubber_hand.STL";
  bool visualizeTrajectory = true;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--no-viewer") {
      visualizeTrajectory = false;
    }
  }

  std::vector<ocp_solver::ContactCandidate> contactCandidates;
  {
    ocp_solver::ContactCandidate candidate;
    candidate.frameName = "rf";
    candidate.parentJointName = "right_ankle_roll_joint";
    candidate.localPose = pinocchio::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.035, 0.0, -0.035));
    contactCandidates.push_back(candidate);
  }
  {
    ocp_solver::ContactCandidate candidate;
    candidate.frameName = "lf";
    candidate.parentJointName = "left_ankle_roll_joint";
    candidate.localPose = pinocchio::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.035, 0.0, -0.035));
    contactCandidates.push_back(candidate);
  }
  {
    ocp_solver::ContactCandidate candidate;
    candidate.frameName = "rh";
    candidate.parentJointName = "right_rubber_hand";
    candidate.localPose = pinocchio::SE3(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0.035, -0.01, 0.0));
    candidate.searchContactPoint = true;
    candidate.alignContactFrameWithMeshNormal = true;
    candidate.meshNormalSubmeshRadius = 0.12;
    const assimp_eigen::MeshData mesh = assimp_eigen::loadMesh(rhContactMeshFile);
    candidate.meshVerticesInLocalFrame = mesh.vertices;
    candidate.meshNormalsInLocalFrame = mesh.normals;
    contactCandidates.push_back(candidate);
  }

  ocp_solver::OCPInterface interface;
  const std::vector<std::string> fixedJointNames = {"left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"};
  interface.initialize("body_contact_wall_hand_mpc", urdfFile, fixedJointNames, false, contactCandidates);
  interface.mpcSettings().timeHorizon_ = 1.0;
  interface.sqpSettings().nThreads = 20;
  interface.sqpSettings().dt = 0.02;
  interface.sqpSettings().sqpIteration = 80;
  interface.sqpSettings().printSolverStatistics = true;

  ocs2::vector_t initialState = makeInitialState(interface.getStateConverter());
  const auto rhContactPointProjector = std::make_shared<point_mesh_projector::PointMeshProjector>(rhContactMeshFile);
  const Eigen::Vector3d rhDefaultLocalPoint =
      interface.getStateConverter().getDefaultContactPointLocalPosition(2);
  initialState.segment<3>(interface.getStateConverter().getContactPointLocalPositionStartIndex(2)) = rhDefaultLocalPoint;
  initialState.segment<3>(interface.getStateConverter().getContactPointLocalPositionStartIndex(2)) =
      rhContactPointProjector->project(interface.getStateConverter().getContactPointLocalPosition(initialState, 2));
  std::cout << "right hand default local point: " << rhDefaultLocalPoint.transpose() << "\n";
  std::cout << "right hand initial projected local point: "
            << interface.getStateConverter().getContactPointLocalPosition(initialState, 2).transpose() << "\n";
  ocs2::vector_t initialInput = ocs2::vector_t::Zero(interface.getStateConverter().getInputDim());

  ocs2::PinocchioInterface& pinocchioInterface = interface.getPinocchioInterface();
  pinocchio::forwardKinematics(pinocchioInterface.getModel(), pinocchioInterface.getData(), initialState.head(pinocchioInterface.getModel().nq));
  pinocchio::updateFramePlacements(pinocchioInterface.getModel(), pinocchioInterface.getData());

  addContactFix(interface, 0, "rf_fixed");
  addContactFix(interface, 1, "lf_fixed");
  addContactFix(interface, 2, "rh_wall_contact");
  {
    auto frameDynamics = std::make_unique<ocp_solver::PinocchioFrameDynamics>(interface.getStateConverter(), 2);
    ocp_constraint::SwingPositionConstraint::Config swingConfig;
    swingConfig.Ax = Eigen::MatrixXd::Identity(3, 3) * 10.0;
    interface.getOptimalControlProblem().stateSoftConstraintPtr->add(
        "rh_swing_to_wall",
        std::make_unique<ocs2::StateSoftConstraint>(
            std::make_unique<ocp_constraint::SwingPositionConstraint>(
                *interface.getReferenceManagerPtr(), *frameDynamics, swingConfig, 0.8, 0.02, 5.0),
            std::make_unique<ocs2::QuadraticPenalty>(1e7)));
  }
  interface.getOptimalControlProblem().equalityConstraintPtr->add(
      "rh_contact_point_search",
      std::make_unique<ocp_constraint_body_contact::ContactPointSearchConstraint>(
          *interface.getReferenceManagerPtr(), 2, interface.getStateConverter(), rhContactMeshFile));
  std::vector<AssumedSurfaceVisualization> assumedSurfaceVisualizations;
  {
    ocs2::PieceWisePolynomialBarrierPenalty::Config footSurfaceContactConfig(1e5, 0.01);
    ocs2::PieceWisePolynomialBarrierPenalty::Config handSurfaceContactConfig(1e6, 0.01);
    ocp_constraint_body_contact::AssumedSurfaceContactConstraint::Config assumedSurfaceConfig;
    assumedSurfaceConfig.ellipseSafetyMargin = 0.01;
    assumedSurfaceConfig.ellipseScale = 1.15;
    assumedSurfaceConfig.frictionCoef = 10.0;
    const std::vector<std::string> contactMeshes = {
        robotDir + "/meshes/right_ankle_roll_link.STL",
        robotDir + "/meshes/left_ankle_roll_link.STL",
        rhContactMeshFile,
    };
    assumedSurfaceVisualizations.reserve(contactMeshes.size());
    for (size_t i = 0; i < contactMeshes.size(); ++i) {
      ocp_constraint_body_contact::AssumedSurfaceContactConstraint::Config contactSurfaceConfig = assumedSurfaceConfig;
      contactSurfaceConfig.ellipseScale = i == 2 ? 1.7 : 1.4;
      auto constraint = std::make_unique<ocp_constraint_body_contact::AssumedSurfaceContactConstraint>(
          *interface.getReferenceManagerPtr(), i, interface.getStateConverter(), contactMeshes[i], contactSurfaceConfig);
      assumedSurfaceVisualizations.push_back({i, constraint.get()});
      interface.getOptimalControlProblem().softConstraintPtr->add(
          "assumedSurfaceContact_" + contactCandidates[i].frameName,
          std::make_unique<ocs2::StateInputSoftConstraint>(
              std::move(constraint),
              std::make_unique<ocs2::PieceWisePolynomialBarrierPenalty>(
                  i == 2 ? handSurfaceContactConfig : footSurfaceContactConfig)));
    }
    interface.getOptimalControlProblem().softConstraintPtr->add(
        "rh_cop_center",
        std::make_unique<ocs2::StateInputSoftConstraint>(
            std::make_unique<ocp_constraint_body_contact::CopCenterConstraint>(
                2, interface.getStateConverter(), *assumedSurfaceVisualizations[2].constraint),
            std::make_unique<ocs2::QuadraticPenalty>(5e4)));
    interface.getOptimalControlProblem().softConstraintPtr->add(
        "rh_cop_moment",
        std::make_unique<ocs2::StateInputSoftConstraint>(
            std::make_unique<ocp_constraint_body_contact::CopMomentConstraint>(
                2, interface.getStateConverter(), *assumedSurfaceVisualizations[2].constraint),
            std::make_unique<ocs2::QuadraticPenalty>(5e4)));
  }

  const size_t stateVariableDim = interface.getStateConverter().getStateVariableDim();
  const size_t inputDim = interface.getStateConverter().getInputDim();
  ocs2::matrix_t Q = ocs2::matrix_t::Identity(stateVariableDim, stateVariableDim) * 1e-2;
  Q.diagonal().head(6).array() = 1.0;
  Q.diagonal().segment(interface.getStateConverter().getContactPointLocalPositionVariableStartIndex(2), 3).array() = 50.0;
  ocs2::matrix_t R = ocs2::matrix_t::Identity(inputDim, inputDim) * 1e-3;
  for (size_t i = 0; i < contactCandidates.size(); ++i) {
    R.diagonal().segment(interface.getStateConverter().getContactForceStartIndices(i), 3).array() = 2e-3;
    R.diagonal().segment(interface.getStateConverter().getContactMomentStartIndices(i), 3).array() = 5e-2;
  }
  R.diagonal().segment(interface.getStateConverter().getContactForceStartIndices(2), 3).array() = 5e-2;
  R.diagonal().segment(interface.getStateConverter().getContactMomentStartIndices(2), 3).array() = 5.0;
  R.diagonal().segment(interface.getStateConverter().getContactPointLocalVelocityStartIndex(2), 3).array() = 1.0;
  interface.getOptimalControlProblem().costPtr->add(
      "quadraticStateInputCost", std::make_unique<ocp_solver::QuadraticStateInputCost>(interface.getPinocchioInterface(), Q, R));
  interface.getOptimalControlProblem().finalCostPtr->add(
      "finalStateCost", std::make_unique<ocp_solver::QuadraticStateCost>(interface.getPinocchioInterface(), Q * 100.0));
  interface.getOptimalControlProblem().stateSoftConstraintPtr->add(
      "jointLimit", std::make_unique<ocp_constraint::JointLimitsConstraint>(interface.getPinocchioInterface(), interface.getStateConverter()));
  interface.getOptimalControlProblem().costPtr->add(
      "jointTorqueCost", std::make_unique<ocp_constraint::JointTorqueCost>(
                             ocs2::matrix_t::Identity(interface.getStateConverter().getJointDim(),
                                                      interface.getStateConverter().getJointDim()) * 1e-4,
                             interface.getStateConverter()));

  const pinocchio::SE3 rfPose = ocp_solver::getContactCandidatePlacement(pinocchioInterface, interface.getStateConverter().getContactCandidate(initialState, 0));
  const pinocchio::SE3 lfPose = ocp_solver::getContactCandidatePlacement(pinocchioInterface, interface.getStateConverter().getContactCandidate(initialState, 1));
  const pinocchio::SE3 rhInitialPose = ocp_solver::getContactCandidatePlacement(pinocchioInterface, interface.getStateConverter().getContactCandidate(initialState, 2));
  Eigen::Matrix3d rhWallRotation;
  rhWallRotation.col(0) = Eigen::Vector3d::UnitX();
  rhWallRotation.col(1) = -Eigen::Vector3d::UnitZ();
  rhWallRotation.col(2) = Eigen::Vector3d::UnitY();
  const pinocchio::SE3 rhWallPose(rhWallRotation, rhInitialPose.translation() + Eigen::Vector3d(0.1, -0.1, 0.1));

  interface.getReferenceManagerPtr()->setContactSchedule(
      ocp_solver::ContactSchedule({0.8}, {{{0, rfPose}, {1, lfPose}},
                                   {{0, rfPose}, {1, lfPose}, {2, rhWallPose}}}));

  {
    auto frameDynamics = std::make_unique<ocp_solver::PinocchioFrameDynamics>(interface.getStateConverter(), 2);
    ocp_constraint::SwingPositionConstraint::Config config;
    config.Ax = Eigen::MatrixXd::Identity(3, 3);
    interface.getOptimalControlProblem().stateSoftConstraintPtr->add(
        "rh_swing_touchdown_position",
        std::make_unique<ocs2::StateSoftConstraint>(
            std::make_unique<ocp_constraint::SwingPositionConstraint>(
                *interface.getReferenceManagerPtr(), *frameDynamics, config, 0.04, 0.0, 3.0),
            std::make_unique<ocs2::QuadraticPenalty>(1e10)));
  }
  ocs2::vector_t stanceInput = initialInput;
  ocs2::vector_t wallContactInput = initialInput;
  updatePinocchioKinematics(pinocchioInterface, initialState);
  for (const AssumedSurfaceVisualization& visualization : assumedSurfaceVisualizations) {
    const pinocchio::SE3 contactPose =
        ocp_solver::getContactCandidatePlacement(pinocchioInterface,
                                                 interface.getStateConverter().getContactCandidate(initialState, visualization.contactIndex));
    const auto stanceSurfaceGeometry =
        visualization.constraint->getSurfaceGeometry(0.0, initialState, pinocchioInterface);
    const double stanceNormalForce = visualization.contactIndex == 2 ? 0.0 : 100.0;
    if (stanceNormalForce > 0.0) {
      interface.getStateConverter().setContactWrench(
          stanceInput, makeCopCenteredWrench(contactPose, stanceSurfaceGeometry, stanceNormalForce), visualization.contactIndex);
      interface.getStateConverter().setContactWrench(
          wallContactInput, makeCopCenteredWrench(contactPose, stanceSurfaceGeometry, 85.0), visualization.contactIndex);
    }
  }
  {
    const pinocchio::SE3 rhContactPose =
        ocp_solver::getContactCandidatePlacement(pinocchioInterface,
                                                 interface.getStateConverter().getContactCandidate(initialState, 2));
    const auto rhSurfaceGeometry =
        assumedSurfaceVisualizations[2].constraint->getSurfaceGeometry(1.0, initialState, pinocchioInterface);
    interface.getStateConverter().setContactWrench(
        wallContactInput, makeCopCenteredWrench(rhContactPose, rhSurfaceGeometry, 80.0), 2);
  }

  const ocs2::scalar_array_t timeTrajectory{0.0, 0.799, 0.8, 1.0};
  const ocs2::vector_array_t stateTrajectory{initialState, initialState, initialState, initialState};
  const ocs2::vector_array_t inputTrajectory{stanceInput, stanceInput, wallContactInput, wallContactInput};
  interface.getReferenceManagerPtr()->setTargetTrajectories({timeTrajectory, stateTrajectory, inputTrajectory});

  auto mpc = interface.createSqpMpc();
  {
    const size_t rhContactPointStateStart =
        interface.getStateConverter().getContactPointLocalPositionStartIndex(2);
    mpc->getSolverPtr()->addStateProjection(
        [rhContactPointProjector, rhContactPointStateStart](ocs2::vector_t& state) {
          state.segment<3>(rhContactPointStateStart) =
              rhContactPointProjector->project(state.segment<3>(rhContactPointStateStart));
        });
  }
  ocs2::SystemObservation obs;
  obs.time = 0.0;
  obs.state = initialState;
  obs.input = initialInput;

  const auto start = std::chrono::high_resolution_clock::now();

  mpc->run(obs.time, obs.state);

  const auto end = std::chrono::high_resolution_clock::now();
  std::cout << "elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

  auto primalSolutionPtr = std::make_unique<ocs2::PrimalSolution>();
  mpc->getSolverPtr()->getPrimalSolution(0.0, primalSolutionPtr.get());
  std::cout << "nodes: " << primalSolutionPtr->timeTrajectory_.size() << "\n";
  for (size_t i = 1; i < primalSolutionPtr->timeTrajectory_.size(); ++i) {
    const double previousTime = primalSolutionPtr->timeTrajectory_[i - 1];
    const double currentTime = primalSolutionPtr->timeTrajectory_[i];
    if (previousTime < 0.8 && currentTime >= 0.8) {
      for (size_t j = i - 1; j <= std::min(i + 1, primalSolutionPtr->stateTrajectory_.size() - 1); ++j) {
        const Eigen::Vector3d localPoint =
            interface.getStateConverter().getContactPointLocalPosition(primalSolutionPtr->stateTrajectory_[j], 2);
        pinocchio::forwardKinematics(pinocchioInterface.getModel(), pinocchioInterface.getData(),
                                     primalSolutionPtr->stateTrajectory_[j].head(pinocchioInterface.getModel().nq));
        const pinocchio::SE3 contactPose =
            ocp_solver::getContactCandidatePlacement(pinocchioInterface,
                                                     interface.getStateConverter().getContactCandidate(primalSolutionPtr->stateTrajectory_[j], 2));
        std::cout << "rh contact sample around touchdown: index=" << j
                  << " time=" << primalSolutionPtr->timeTrajectory_[j]
                  << " inContact=" << interface.getReferenceManagerPtr()->isInContact(primalSolutionPtr->timeTrajectory_[j], 2)
                  << " local=" << localPoint.transpose()
                  << " world=" << contactPose.translation().transpose()
                  << " targetError=" << (contactPose.translation() - rhWallPose.translation()).transpose() << "\n";
      }
      break;
    }
  }
  if (!primalSolutionPtr->stateTrajectory_.empty()) {
    const ocs2::vector_t& finalState = primalSolutionPtr->stateTrajectory_.back();
    pinocchio::forwardKinematics(pinocchioInterface.getModel(), pinocchioInterface.getData(), finalState.head(pinocchioInterface.getModel().nq));
    const pinocchio::SE3 rhFinalPose =
      ocp_solver::getContactCandidatePlacement(pinocchioInterface, interface.getStateConverter().getContactCandidate(finalState, 2));
    const pinocchio::SE3 rfFinalPose =
      ocp_solver::getContactCandidatePlacement(pinocchioInterface, interface.getStateConverter().getContactCandidate(finalState, 0));
    const pinocchio::SE3 lfFinalPose =
      ocp_solver::getContactCandidatePlacement(pinocchioInterface, interface.getStateConverter().getContactCandidate(finalState, 1));
    std::cout << "right foot position error: " << (rfFinalPose.translation() - rfPose.translation()).transpose() << "\n";
    std::cout << "left foot position error: " << (lfFinalPose.translation() - lfPose.translation()).transpose() << "\n";
    std::cout << "right hand final position: " << rhFinalPose.translation().transpose() << "\n";
    std::cout << "right hand wall target error: " << (rhFinalPose.translation() - rhWallPose.translation()).transpose() << "\n";
    const Eigen::Vector3d searchedLocalPoint =
        interface.getStateConverter().getContactPointLocalPosition(finalState, 2);
    std::cout << "searched local point: " << searchedLocalPoint.transpose() << "\n";
    std::cout << "searched local point projection error: "
              << (searchedLocalPoint - rhContactPointProjector->project(searchedLocalPoint)).norm() << "\n";
    if (!primalSolutionPtr->inputTrajectory_.empty()) {
      const ocs2::vector_t& finalInput =
          primalSolutionPtr->inputTrajectory_[std::min(primalSolutionPtr->inputTrajectory_.size() - 1,
                                                       primalSolutionPtr->stateTrajectory_.size() - 1)];
      for (const AssumedSurfaceVisualization& visualization : assumedSurfaceVisualizations) {
        const pinocchio::SE3 contactPose =
            ocp_solver::getContactCandidatePlacement(pinocchioInterface,
                                                     interface.getStateConverter().getContactCandidate(finalState, visualization.contactIndex));
        const auto surfaceGeometry =
            visualization.constraint->getSurfaceGeometry(primalSolutionPtr->timeTrajectory_.back(), finalState, pinocchioInterface);
        const Eigen::Vector3d copLocal =
            visualization.constraint->computePressureCenterInContactFrame(
                contactPose.rotation(),
                interface.getStateConverter().getContactWrench(finalInput, visualization.contactIndex),
                surfaceGeometry.normalInContactFrame);
        const Eigen::Vector3d copError = copLocal - surfaceGeometry.geometricCenterInContactPlane;
        const double ellipseMargin =
            1.0 - copError.dot(surfaceGeometry.ellipseMetric * copError);
        Eigen::Matrix<double, 6, 6> worldToContact = Eigen::Matrix<double, 6, 6>::Zero();
        worldToContact.block<3, 3>(0, 0) = contactPose.rotation().transpose();
        worldToContact.block<3, 3>(3, 3) = contactPose.rotation().transpose();
        const Eigen::Matrix<double, 6, 1> localWrench =
            worldToContact * interface.getStateConverter().getContactWrench(finalInput, visualization.contactIndex);
        const double normalForce =
            std::abs(surfaceGeometry.normalInContactFrame.normalized().dot(localWrench.head<3>()));
        std::cout << interface.getStateConverter().getContactCandidate(visualization.contactIndex).frameName
                  << " cop ellipse margin: " << ellipseMargin
                  << " normalForce: " << normalForce
                  << " cop: " << copLocal.transpose()
                  << " center: " << surfaceGeometry.geometricCenterInContactPlane.transpose()
                  << " localWrench: " << localWrench.transpose() << "\n";
      }
    }
  }

  std::map<std::string, std::vector<std::vector<double>>> results;
  const int N = primalSolutionPtr->timeTrajectory_.size();
  const int dof = 30 + 6;
  results["q"].resize(N, std::vector<double>(dof, 0.0));
  for (int i = 0; i < N; ++i) {
    size_t offset = 0;
    const int nq = static_cast<int>(pinocchioInterface.getModel().nq);
    for (int j = 0; j < nq; ++j) {
      results["q"][i][j + offset] = primalSolutionPtr->stateTrajectory_[i][j];
      if (j == 24) {
        results["q"][i][j + offset + 1] = 0.0;
        results["q"][i][j + offset + 2] = 0.0;
        results["q"][i][j + offset + 3] = 0.0;
        offset += 3;
      }
    }
  }
  trajectory_logger::write(packageShare + "/wall_hand_mpc.csv", results, primalSolutionPtr->timeTrajectory_);
  if (visualizeTrajectory) {
    const std::vector<ContactTrajectoryVisualization> visualizations = {
        {0, {0.1f, 0.35f, 1.0f, 1.0f}, {1.0f, 0.15f, 0.15f, 1.0f}, {1.0f, 0.85f, 0.05f, 1.0f}},
        {1, {0.1f, 0.8f, 0.35f, 1.0f}, {1.0f, 0.15f, 0.15f, 1.0f}, {1.0f, 0.85f, 0.05f, 1.0f}},
        {2, {0.45f, 0.2f, 1.0f, 1.0f}, {1.0f, 0.15f, 0.15f, 1.0f}, {1.0f, 0.85f, 0.05f, 1.0f}},
    };
    visualizeOptimizationTrajectory(mujocoModelFile, primalSolutionPtr->timeTrajectory_,
                                    primalSolutionPtr->stateTrajectory_, primalSolutionPtr->inputTrajectory_,
                                    pinocchioInterface,
                                    interface.getStateConverter(), *interface.getReferenceManagerPtr(),
                                    visualizations, assumedSurfaceVisualizations);
  }

  return 0;
}

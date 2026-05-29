#include <ament_index_cpp/get_package_share_directory.hpp>

#include <chrono>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <thread>
#include <vector>

#include <Eigen/Eigenvalues>
#include <mujoco/mujoco.h>
#include <mujoco_viewer/mujoco_viewer.h>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/fwd.hpp>

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocp_constraint/contact_fix_constraint.h>
#include <ocp_constraint/joint_limit_constraint.h>
#include <ocp_constraint/joint_torque_cost.h>
#include <ocp_constraint/penalties/piece_wise_polynominal_barrier_penalty.h>
#include <ocp_constraint/position_constraint.h>
#include <ocp_constraint_body_contact/assumed_surface_contact_constraint.h>
#include <ocp_solver/common/quadratic_state_cost.h>
#include <ocp_solver/common/quadratic_state_input_cost.h>
#include <ocp_solver/ocp_interface.h>
#include <ocp_solver/pinocchio/pinocchio_frame_dynamics.h>
#include <ocp_solver/solver/dynamics_helper_functions.h>
#include <trajectory_logger/trajectory_logger.h>

namespace {

struct AssumedSurfaceVisualization {
  size_t contactIndex = 0;
  const ocp_constraint_body_contact::AssumedSurfaceContactConstraint* constraint = nullptr;
};

ocs2::vector_t makeInitialState() {
  std::vector<double> initialState_v = {0.0, 0.0, 0.65, 0.0, 0.0, 0.0, 1.0,
                                        -0.6, 0.0, 0.0, 1.4, -0.8, 0.0,
                                        -0.6, 0.0, 0.0, 1.4, -0.8, 0.0,
                                        0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0};
  return Eigen::Map<ocs2::vector_t>(initialState_v.data(), initialState_v.size());
}

std::vector<double> toViewerQ(const ocs2::vector_t& state) {
  std::vector<double> q(36, 0.0);
  size_t offset = 0;
  for (int j = 0; j < 30; ++j) {
    q[j + offset] = state[j];
    if ((j == 24) || (j == 28)) {
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

void addFootConstraint(ocp_solver::OCPInterface& interface, size_t contactIndex) {
  ocp_constraint::PositionConstraint::Config config;
  config.Ax = ocs2::matrix_t::Zero(6, 6);
  config.Ax.leftCols(6).setIdentity();

  const std::string frameName = interface.getStateConverter().getContactCandidate(contactIndex).frameName;
  auto frameDynamics = std::make_unique<ocp_solver::PinocchioFrameDynamics>(interface.getStateConverter(), contactIndex);
  interface.getOptimalControlProblem().equalityConstraintPtr->add(
      frameName + "_pose",
      std::make_unique<ocp_constraint::ContactFixConstraint>(*interface.getReferenceManagerPtr(), *frameDynamics, 6, config));
}

void appendSphere(mujoco_viewer::Viewer& viewer,
                  const Eigen::Vector3d& position,
                  double radius,
                  const std::array<float, 4>& rgba) {
  mjvGeom* geom = viewer.appendGeom();
  if (!geom) return;

  const mjtNum size[3] = {radius, radius, radius};
  const mjtNum xpos[3] = {position.x(), position.y(), position.z()};
  mjv_initGeom(geom, mjGEOM_SPHERE, size, xpos, nullptr, rgba.data());
}

void appendLine(mujoco_viewer::Viewer& viewer,
                const Eigen::Vector3d& from,
                const Eigen::Vector3d& to,
                double width,
                const std::array<float, 4>& rgba) {
  mjvGeom* geom = viewer.appendGeom();
  if (!geom) return;

  const mjtNum fromArray[3] = {from.x(), from.y(), from.z()};
  const mjtNum toArray[3] = {to.x(), to.y(), to.z()};
  mjv_initGeom(geom, mjGEOM_LINE, nullptr, nullptr, nullptr, rgba.data());
  mjv_connector(geom, mjGEOM_LINE, width, fromArray, toArray);
}

void appendAssumedSurfaceEllipse(mujoco_viewer::Viewer& viewer,
                                 const pinocchio::SE3& contactPose,
                                 const ocp_constraint_body_contact::AssumedSurfaceContactConstraint& constraint) {
  const Eigen::Vector3d normal = constraint.getNormalInContactFrame().normalized();
  const Eigen::Vector3d tangent0 = normal.unitOrthogonal();
  const Eigen::Vector3d tangent1 = normal.cross(tangent0).normalized();
  Eigen::Matrix<double, 3, 2> tangentBasis;
  tangentBasis.col(0) = tangent0;
  tangentBasis.col(1) = tangent1;

  const Eigen::Matrix2d tangentMetric =
      tangentBasis.transpose() * constraint.getEllipseMetricInContactFrame() * tangentBasis;
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(tangentMetric);
  if (solver.info() != Eigen::Success) return;

  const Eigen::Vector2d eigenvalues = solver.eigenvalues().cwiseMax(1e-12);
  const Eigen::Matrix2d eigenvectors = solver.eigenvectors();
  const Eigen::Vector3d centerLocal = constraint.getGeometricCenterInContactPlane();

  constexpr int segments = 96;
  const std::array<float, 4> ellipseColor = {0.1f, 0.9f, 0.25f, 1.0f};
  Eigen::Vector3d previousWorld;
  for (int i = 0; i <= segments; ++i) {
    constexpr double twoPi = 6.28318530717958647692;
    const double theta = twoPi * static_cast<double>(i) / static_cast<double>(segments);
    Eigen::Vector2d ellipsePoint;
    ellipsePoint << std::cos(theta) / std::sqrt(eigenvalues[0]),
                    std::sin(theta) / std::sqrt(eigenvalues[1]);
    const Eigen::Vector3d pointLocal = centerLocal + tangentBasis * (eigenvectors * ellipsePoint);
    const Eigen::Vector3d pointWorld = contactPose.translation() + contactPose.rotation() * pointLocal;
    if (i > 0) appendLine(viewer, previousWorld, pointWorld, 3.0, ellipseColor);
    previousWorld = pointWorld;
  }
}

void appendAssumedSurfaceVisualization(mujoco_viewer::Viewer& viewer,
                                       ocs2::PinocchioInterface& pinocchioInterface,
                                       const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
                                       const ocs2::vector_t& input,
                                       const std::vector<AssumedSurfaceVisualization>& visualizations) {
  const std::array<float, 4> contactColor = {0.1f, 0.3f, 1.0f, 1.0f};
  const std::array<float, 4> centerColor = {1.0f, 0.9f, 0.1f, 1.0f};
  const std::array<float, 4> copColor = {1.0f, 0.05f, 0.05f, 1.0f};
  const std::array<float, 4> normalColor = {0.95f, 0.95f, 0.95f, 1.0f};

  for (const AssumedSurfaceVisualization& visualization : visualizations) {
    if (!visualization.constraint) continue;
    const pinocchio::SE3 contactPose =
        ocp_solver::getContactCandidatePlacement(pinocchioInterface,
                                                 stateConverter.getContactCandidate(visualization.contactIndex));
    const Eigen::Matrix<ocs2::scalar_t, 6, 1> wrench =
        stateConverter.getContactWrench(input, visualization.contactIndex);
    const Eigen::Vector3d centerWorld =
        contactPose.translation() + contactPose.rotation() * visualization.constraint->getGeometricCenterInContactPlane();
    const Eigen::Vector3d copLocal =
        visualization.constraint->computePressureCenterInContactFrame(contactPose.rotation(), wrench);
    const Eigen::Vector3d copWorld = contactPose.translation() + contactPose.rotation() * copLocal;
    const Eigen::Vector3d normalWorld =
        contactPose.rotation() * visualization.constraint->getNormalInContactFrame().normalized();

    appendAssumedSurfaceEllipse(viewer, contactPose, *visualization.constraint);
    appendSphere(viewer, contactPose.translation(), 0.012, contactColor);
    appendSphere(viewer, centerWorld, 0.01, centerColor);
    appendSphere(viewer, copWorld, 0.014, copColor);
    appendLine(viewer, contactPose.translation(), contactPose.translation() + 0.08 * normalWorld, 2.0, normalColor);
  }
}

void setViewerQ(mujoco_viewer::Viewer& viewer, const ocs2::vector_t& state) {
  const std::vector<double> viewerQ = toViewerQ(state);
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

void visualizeOptimizationTrajectory(const std::string& modelPath,
                                     const ocs2::vector_array_t& stateTrajectory,
                                     const ocs2::vector_array_t& inputTrajectory,
                                     ocs2::PinocchioInterface& pinocchioInterface,
                                     const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
                                     const std::vector<AssumedSurfaceVisualization>& visualizations) {
  if (stateTrajectory.empty() || inputTrajectory.empty()) {
    return;
  }

  mujoco_viewer::Viewer viewer;
  viewer.viewModel(modelPath);

  size_t trajectoryIndex = 0;
  auto lastFrameTime = std::chrono::steady_clock::now();
  constexpr auto frameDuration = std::chrono::milliseconds(120);

  while (viewer.isOpen()) {
    const auto now = std::chrono::steady_clock::now();
    if (now - lastFrameTime >= frameDuration) {
      trajectoryIndex = (trajectoryIndex + 1) % std::min(stateTrajectory.size(), inputTrajectory.size());
      lastFrameTime = now;
    }

    updatePinocchioKinematics(pinocchioInterface, stateTrajectory[trajectoryIndex]);
    setViewerQ(viewer, stateTrajectory[trajectoryIndex]);
    viewer.updateScene();
    appendAssumedSurfaceVisualization(viewer, pinocchioInterface, stateConverter, inputTrajectory[trajectoryIndex], visualizations);
    viewer.drawScene();
    viewer.pollEvents();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

}  // namespace

int main() {
  const std::string packageShare = ament_index_cpp::get_package_share_directory("ocp_constraint_body_contact_sample");
  const std::string robotDir = packageShare + "/unitree_ros/robots/g1_description";
  const std::string urdfFile = robotDir + "/g1_29dof.urdf";
  const std::string mujocoModelFile = robotDir + "/g1_29dof.xml";

  ocp_solver::OCPInterface interface;
  const std::vector<std::string> fixedJointNames = {"left_wrist_roll_joint",
                                                    "left_wrist_pitch_joint",
                                                    "left_wrist_yaw_joint",
                                                    "right_wrist_roll_joint",
                                                    "right_wrist_pitch_joint",
                                                    "right_wrist_yaw_joint"};
  std::vector<ocp_solver::ContactCandidate> contactCandidates;
  {
    ocp_solver::ContactCandidate candidate;
    candidate.frameName = "rf";
    candidate.parentJointName = "right_ankle_roll_joint";
    Eigen::Vector3d translationFromParent;
    translationFromParent << 0.035, 0.0, -0.035;
    candidate.localPose = pinocchio::SE3(Eigen::Matrix3d::Identity(), translationFromParent);
    contactCandidates.push_back(candidate);
  }
  {
    ocp_solver::ContactCandidate candidate;
    candidate.frameName = "lf";
    candidate.parentJointName = "left_ankle_roll_joint";
    Eigen::Vector3d translationFromParent;
    translationFromParent << 0.035, 0.0, -0.035;
    candidate.localPose = pinocchio::SE3(Eigen::Matrix3d::Identity(), translationFromParent);
    contactCandidates.push_back(candidate);
  }

  interface.initialize("body_contact_reach_pose", urdfFile, fixedJointNames, false, contactCandidates);
  interface.sqpSettings().sqpIteration = 100;
  interface.sqpSettings().printSolverStatistics = true;

  ocs2::vector_t initialState = makeInitialState();
  ocs2::vector_t initialInput = ocs2::vector_t::Zero(interface.getStateConverter().getInputDim());
  ocs2::PinocchioInterface& pinocchioInterface = interface.getPinocchioInterface();
  pinocchio::forwardKinematics(pinocchioInterface.getModel(), pinocchioInterface.getData(),
                               initialState.head(pinocchioInterface.getModel().nq));
  pinocchio::updateFramePlacements(pinocchioInterface.getModel(), pinocchioInterface.getData());

  const Eigen::Vector3d targetPosition(0.15, -0.4, 0.8);
  {
    ocp_constraint::PositionConstraint::Config config;
    config.Ax = ocs2::matrix_t::Zero(3, 6);
    config.Ax.leftCols(3).setIdentity();

    auto frameDynamics = std::make_unique<ocp_solver::PinocchioFrameDynamics>(interface.getPinocchioInterface(),
                                                                              interface.getStateConverter(),
                                                                              "right_wrist_pitch_link");
    interface.getOptimalControlProblem().equalityConstraintPtr->add(
        "right_wrist_position",
        std::make_unique<ocp_constraint::PositionConstraint>(*frameDynamics, 3, config,
                                                             pinocchio::SE3(Eigen::Matrix3d::Identity(), targetPosition)));
  }

  addFootConstraint(interface, 0);
  addFootConstraint(interface, 1);
  std::vector<AssumedSurfaceVisualization> assumedSurfaceVisualizations;
  assumedSurfaceVisualizations.reserve(contactCandidates.size());

  {
    std::vector<double> Q_v = {0.0, 0.0, 100.0, 1.0, 1.0, 1.0,
                               0.5, 2.0, 2.0, 1.0, 1.0, 1.0,
                               0.5, 2.0, 2.0, 1.0, 1.0, 1.0,
                               2.0, 2.0, 2.0,
                               0.4, 2.0, 0.4, 0.4,
                               0.4, 2.0, 0.4, 0.4,
                               3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
                               0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
                               0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
                               0.2, 0.2, 0.2,
                               0.02, 0.02, 0.02, 0.02,
                               0.02, 0.02, 0.02, 0.02};
    Eigen::VectorXd Qv = Eigen::Map<Eigen::VectorXd>(Q_v.data(), Q_v.size());
    ocs2::matrix_t Q = Qv.asDiagonal();

    std::vector<double> R_v = {0.003, 0.003, 0.001, 0.01, 0.01, 0.1,
                               0.003, 0.003, 0.001, 0.01, 0.01, 0.1,
                               0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
                               0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
                               0.005, 0.005, 0.005,
                               0.005, 0.005, 0.005, 0.005,
                               0.005, 0.005, 0.005, 0.005};
    Eigen::VectorXd Rv = Eigen::Map<Eigen::VectorXd>(R_v.data(), R_v.size());
    ocs2::matrix_t R = Rv.asDiagonal();
    interface.getOptimalControlProblem().costPtr->add(
        "quadraticStateInputCost",
        std::make_unique<ocp_solver::QuadraticStateInputCost>(interface.getPinocchioInterface(), Q, R));
    interface.getOptimalControlProblem().stateCostPtr->add(
        "stateCost",
        std::make_unique<ocp_solver::QuadraticStateCost>(interface.getPinocchioInterface(), Q * 1e2));
  }

  {
    interface.getOptimalControlProblem().stateSoftConstraintPtr->add(
        "jointLimit",
        std::unique_ptr<ocs2::StateCost>(
            new ocp_constraint::JointLimitsConstraint(interface.getPinocchioInterface(), interface.getStateConverter())));

    ocs2::PieceWisePolynomialBarrierPenalty::Config surfaceContactConfig(1e6, 1e-4);
    ocp_constraint_body_contact::AssumedSurfaceContactConstraint::Config assumedSurfaceConfig;
    assumedSurfaceConfig.ellipseSafetyMargin = 0.02;
    const std::vector<std::string> footMeshes = {robotDir + "/meshes/right_ankle_roll_link.STL",
                                                robotDir + "/meshes/left_ankle_roll_link.STL"};
    for (size_t i = 0; i < contactCandidates.size(); ++i) {
      auto constraint = std::make_unique<ocp_constraint_body_contact::AssumedSurfaceContactConstraint>(
          *interface.getReferenceManagerPtr(), i, interface.getStateConverter(), footMeshes[i], assumedSurfaceConfig);
      assumedSurfaceVisualizations.push_back({i, constraint.get()});
      interface.getOptimalControlProblem().softConstraintPtr->add(
          "assumedSurfaceContact_" + contactCandidates[i].frameName,
          std::make_unique<ocs2::StateInputSoftConstraint>(
              std::move(constraint),
              std::make_unique<ocs2::PieceWisePolynomialBarrierPenalty>(surfaceContactConfig)));
    }

    interface.getOptimalControlProblem().costPtr->add(
        "jointTorqueCost_analytical",
        std::unique_ptr<ocs2::StateInputCost>(
            new ocp_constraint::JointTorqueCost(
                ocs2::matrix_t::Identity(interface.getStateConverter().getJointDim(),
                                         interface.getStateConverter().getJointDim()) *
                    1e-3,
                interface.getStateConverter())));
  }

  const ocs2::scalar_array_t timeTrajectory{0.0};
  const ocs2::vector_array_t stateTrajectory{initialState};
  const ocs2::vector_array_t inputTrajectory{initialInput};
  interface.getReferenceManagerPtr()->setTargetTrajectories({timeTrajectory, stateTrajectory, inputTrajectory});

  const pinocchio::FrameIndex wristFrameId = pinocchioInterface.getModel().getFrameId("right_wrist_pitch_link");
  const pinocchio::SE3 rfPose = ocp_solver::getContactCandidatePlacement(pinocchioInterface, interface.getStateConverter().getContactCandidate(0));
  const pinocchio::SE3 lfPose = ocp_solver::getContactCandidatePlacement(pinocchioInterface, interface.getStateConverter().getContactCandidate(1));
  interface.getReferenceManagerPtr()->setContactSchedule(
      ocp_solver::ContactSchedule({}, {{{0, rfPose},
                                        {1, lfPose}}}));
  interface.getReferenceManagerPtr()->preSolverRun(0.0, 0.0, initialState);

  auto optimizer = interface.createPoseOptimizer();
  const auto start = std::chrono::high_resolution_clock::now();
  const ocp_solver::PoseOptimizerResult result = optimizer->run(0.0, initialState, initialInput);
  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "elapsed: " << elapsed.count() << " us\n";
  std::cout << "iterations: " << result.iterations << "\n";
  std::cout << "success: " << result.success << "\n";
  std::cout << "merit: " << result.performance.merit << "\n";
  std::cout << "constraint violation: " << ocs2::FilterLinesearch::totalConstraintViolation(result.performance) << "\n";

  pinocchio::forwardKinematics(pinocchioInterface.getModel(), pinocchioInterface.getData(),
                               result.state.head(pinocchioInterface.getModel().nq));
  pinocchio::updateFramePlacements(pinocchioInterface.getModel(), pinocchioInterface.getData());
  std::cout << "right_wrist_pitch_link position:\n"
            << pinocchioInterface.getData().oMf[wristFrameId].translation().transpose() << "\n";
  std::cout << "rf position error:\n"
            << (ocp_solver::getContactCandidatePlacement(pinocchioInterface, interface.getStateConverter().getContactCandidate(0)).translation() - rfPose.translation()).transpose() << "\n";
  std::cout << "lf position error:\n"
            << (ocp_solver::getContactCandidatePlacement(pinocchioInterface, interface.getStateConverter().getContactCandidate(1)).translation() - lfPose.translation()).transpose() << "\n";
  for (const AssumedSurfaceVisualization& visualization : assumedSurfaceVisualizations) {
    const pinocchio::SE3 contactPose =
        ocp_solver::getContactCandidatePlacement(pinocchioInterface,
                                                 interface.getStateConverter().getContactCandidate(visualization.contactIndex));
    const Eigen::Vector3d copLocal =
        visualization.constraint->computePressureCenterInContactFrame(
            contactPose.rotation(), interface.getStateConverter().getContactWrench(result.input, visualization.contactIndex));
    const Eigen::Vector3d copError =
        copLocal - visualization.constraint->getGeometricCenterInContactPlane();
    const double ellipseMargin =
        1.0 - copError.dot(visualization.constraint->getEllipseMetricInContactFrame() * copError);
    std::cout << interface.getStateConverter().getContactCandidate(visualization.contactIndex).frameName
              << " assumed surface margin: " << ellipseMargin << "\n";
  }

  std::map<std::string, std::vector<std::vector<double>>> results;
  std::vector<double> times(result.stateTrajectory.size());
  results["q"].resize(result.stateTrajectory.size());
  for (size_t i = 0; i < result.stateTrajectory.size(); ++i) {
    times[i] = 0.01 * static_cast<double>(i);
    results["q"][i] = toViewerQ(result.stateTrajectory[i]);
  }
  trajectory_logger::write(packageShare + "/reach_pose.csv", results, times);
  visualizeOptimizationTrajectory(mujocoModelFile, result.stateTrajectory, result.inputTrajectory, pinocchioInterface,
                                  interface.getStateConverter(), assumedSurfaceVisualizations);

  return 0;
}

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Eigenvalues>
#include <mujoco/mujoco.h>
#include <mujoco_viewer/mujoco_viewer.h>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/fwd.hpp>

#include <assimp_eigen/assimp_eigen.h>
#include <ocp_constraint/position_constraint.h>
#include <ocp_constraint_body_contact/assumed_surface_contact_constraint.h>
#include <ocp_constraint_body_contact/contact_point_search_constraint.h>
#include <ocp_solver/common/quadratic_state_input_cost.h>
#include <ocp_solver/ocp_interface.h>
#include <ocp_solver/pinocchio/pinocchio_frame_dynamics.h>
#include <ocp_solver/solver/dynamics_helper_functions.h>
#include <ocs2_core/penalties/penalties/QuadraticPenalty.h>
#include <ocs2_core/soft_constraint/StateInputSoftConstraint.h>
#include <ocp_constraint/penalties/piece_wise_polynominal_barrier_penalty.h>
#include <point_mesh_projector/point_mesh_projector.h>

namespace {

std::string readFile(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open: " + path);
  }
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void writeFile(const std::string& path, const std::string& text) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to write: " + path);
  }
  output << text;
}

void replaceAll(std::string& text, const std::string& from, const std::string& to) {
  size_t pos = 0;
  while ((pos = text.find(from, pos)) != std::string::npos) {
    text.replace(pos, from.size(), to);
    pos += to.size();
  }
}

std::string makeFixedRootMujocoXml(const std::string& robotDir) {
  std::string xml = readFile(robotDir + "/g1_29dof.xml");
  replaceAll(xml, "meshdir=\"meshes\"", "meshdir=\"" + robotDir + "/meshes\"");
  replaceAll(xml,
             "      <joint name=\"floating_base_joint\" type=\"free\" limited=\"false\" actuatorfrclimited=\"false\"/>\n",
             "");

  const std::string fixedRootXml = "/tmp/g1_29dof_fixed_root.xml";
  writeFile(fixedRootXml, xml);
  return fixedRootXml;
}

Eigen::Vector3d computeWeightedMeshNormal(const assimp_eigen::MeshData& mesh,
                                          const Eigen::Vector3d& point) {
  Eigen::Vector3d weightedNormal = Eigen::Vector3d::Zero();
  double weightSum = 0.0;
  const double lengthScale = 0.02;
  const double invTwoSigma2 = 0.5 / (lengthScale * lengthScale);
  for (size_t i = 0; i < mesh.vertices.size(); ++i) {
    const double squaredDistance = (mesh.vertices[i] - point).squaredNorm();
    const double weight = std::exp(-squaredDistance * invTwoSigma2);
    const Eigen::Vector3d normal = mesh.normals[std::min(i, mesh.normals.size() - 1)];
    if (normal.allFinite() && normal.squaredNorm() > 1e-12) {
      weightedNormal.noalias() += weight * normal.normalized();
      weightSum += weight;
    }
  }
  if (weightSum <= 0.0 || !weightedNormal.allFinite() || weightedNormal.squaredNorm() < 1e-12) {
    return Eigen::Vector3d::UnitZ();
  }
  return (weightedNormal / weightSum).normalized();
}

Eigen::Vector3d computeMeshCenter(const assimp_eigen::MeshData& mesh) {
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (const Eigen::Vector3d& vertex : mesh.vertices) {
    center += vertex;
  }
  return mesh.vertices.empty() ? center : center / static_cast<double>(mesh.vertices.size());
}

Eigen::Matrix3d makeContactRotationWithInwardZ(const assimp_eigen::MeshData& mesh,
                                               const Eigen::Vector3d& point) {
  Eigen::Vector3d inwardNormal = computeWeightedMeshNormal(mesh, point);
  const Eigen::Vector3d towardMeshCenter = computeMeshCenter(mesh) - point;
  if (towardMeshCenter.allFinite() && towardMeshCenter.squaredNorm() > 1e-12
      && inwardNormal.dot(towardMeshCenter) < 0.0) {
    inwardNormal = -inwardNormal;
  }

  Eigen::Vector3d xAxis = Eigen::Vector3d::UnitX() - inwardNormal * inwardNormal.x();
  if (!xAxis.allFinite() || xAxis.squaredNorm() < 1e-12) {
    xAxis = Eigen::Vector3d::UnitY() - inwardNormal * inwardNormal.y();
  }
  if (!xAxis.allFinite() || xAxis.squaredNorm() < 1e-12) {
    xAxis = inwardNormal.unitOrthogonal();
  }
  xAxis.normalize();
  Eigen::Vector3d yAxis = inwardNormal.cross(xAxis).normalized();
  xAxis = yAxis.cross(inwardNormal).normalized();

  Eigen::Matrix3d rotation;
  rotation.col(0) = xAxis;
  rotation.col(1) = yAxis;
  rotation.col(2) = inwardNormal;
  return rotation;
}

std::array<mjtNum, 3> toMujocoPosition(const Eigen::Vector3d& position) {
  return {position.x(), position.y(), position.z()};
}

void setViewerQ(mujoco_viewer::Viewer& viewer, const ocs2::vector_t& state) {
  if (!viewer.model() || !viewer.data()) {
    return;
  }
  const int nq = std::min<int>(viewer.model()->nq, state.rows());
  for (int i = 0; i < nq; ++i) {
    viewer.data()->qpos[i] = state[i];
  }
  mj_forward(viewer.model(), viewer.data());
}

void updatePinocchioKinematics(ocs2::PinocchioInterface& pinocchioInterface, const ocs2::vector_t& state) {
  pinocchio::forwardKinematics(pinocchioInterface.getModel(), pinocchioInterface.getData(),
                               state.head(pinocchioInterface.getModel().nq));
  pinocchio::updateFramePlacements(pinocchioInterface.getModel(), pinocchioInterface.getData());
}

void projectContactPointToMesh(const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
                               const point_mesh_projector::PointMeshProjector& projector,
                               ocs2::vector_t& state) {
  state.segment<3>(stateConverter.getContactPointLocalPositionStartIndex(0)) =
      projector.project(stateConverter.getContactPointLocalPosition(state, 0));
}

void showOnlyBody(mujoco_viewer::Viewer& viewer, const std::string& bodyName) {
  mjModel* model = viewer.model();
  if (!model) return;

  const int visibleBodyId = mj_name2id(model, mjOBJ_BODY, bodyName.c_str());
  if (visibleBodyId < 0) {
    std::cerr << "failed to find MuJoCo body: " << bodyName << std::endl;
    return;
  }

  for (int geomId = 0; geomId < model->ngeom; ++geomId) {
    const bool visible = model->geom_bodyid[geomId] == visibleBodyId;
    model->geom_rgba[4 * geomId + 3] = visible ? 0.85f : 0.0f;
  }
}

void appendWall(mujoco_viewer::Viewer& viewer, double x, const Eigen::Vector3d& visualOffset) {
  mjvGeom* geom = viewer.appendGeom();
  if (!geom) return;

  const mjtNum size[3] = {0.004, 0.45, 1.5};
  const mjtNum xpos[3] = {x + visualOffset.x(), visualOffset.y(), visualOffset.z()};
  const float rgba[4] = {0.1f, 0.45f, 1.0f, 0.18f};
  mjv_initGeom(geom, mjGEOM_BOX, size, xpos, nullptr, rgba);
}

void appendAssumedSurfaceEllipse(
    mujoco_viewer::Viewer& viewer,
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
  if (solver.info() != Eigen::Success) {
    return;
  }

  const Eigen::Vector2d eigenvalues = solver.eigenvalues().cwiseMax(1e-12);
  const Eigen::Matrix2d eigenvectors = solver.eigenvectors();
  const Eigen::Vector3d centerLocal = surfaceGeometry.geometricCenterInContactPlane;

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
    if (i > 0) {
      mujoco_viewer::appendLine(viewer, toMujocoPosition(previousWorld), toMujocoPosition(pointWorld), 3.0, ellipseColor);
    }
    previousWorld = pointWorld;
  }
}

void appendContactVisualization(
    mujoco_viewer::Viewer& viewer,
    ocs2::PinocchioInterface& pinocchioInterface,
    const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
    const ocp_constraint_body_contact::AssumedSurfaceContactConstraint& surfaceConstraint,
    const ocs2::vector_t& state,
    double wallX,
    const Eigen::Vector3d& visualOffset) {
  updatePinocchioKinematics(pinocchioInterface, state);
  pinocchio::SE3 contactPose =
      ocp_solver::getContactCandidatePlacement(pinocchioInterface, stateConverter.getContactCandidate(state, 0));
  contactPose.translation() += visualOffset;
  const auto surfaceGeometry = surfaceConstraint.getSurfaceGeometry(0.0, state, pinocchioInterface);
  const Eigen::Vector3d centerWorld =
      contactPose.translation() + contactPose.rotation() * surfaceGeometry.geometricCenterInContactPlane;
  const Eigen::Vector3d normalWorld =
      contactPose.rotation() * surfaceGeometry.normalInContactFrame.normalized();
  const Eigen::Vector3d targetWorld(wallX, contactPose.translation().y(), contactPose.translation().z());

  appendAssumedSurfaceEllipse(viewer, contactPose, surfaceGeometry);
  mujoco_viewer::appendSphere(viewer, toMujocoPosition(contactPose.translation()), 0.012, {0.55f, 0.1f, 1.0f, 1.0f});
  mujoco_viewer::appendSphere(viewer, toMujocoPosition(centerWorld), 0.01, {1.0f, 0.9f, 0.1f, 1.0f});
  mujoco_viewer::appendSphere(viewer, toMujocoPosition(targetWorld), 0.012, {1.0f, 0.05f, 0.05f, 1.0f});
  mujoco_viewer::appendLine(viewer, toMujocoPosition(contactPose.translation()),
                            toMujocoPosition(contactPose.translation() + 0.07 * normalWorld),
                            2.0, {0.95f, 0.95f, 0.95f, 1.0f});
}

void visualizeTrajectory(
    const std::string& modelPath,
    const ocs2::vector_array_t& stateTrajectory,
    ocs2::PinocchioInterface& pinocchioInterface,
    const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
    const ocp_constraint_body_contact::AssumedSurfaceContactConstraint& surfaceConstraint,
    double wallX) {
  if (stateTrajectory.empty()) {
    return;
  }

  mujoco_viewer::Viewer viewer;
  viewer.viewModel(modelPath);
  showOnlyBody(viewer, "left_hip_yaw_link");
  const Eigen::Vector3d visualOffset(0.0, 0.0, 0.793);

  size_t trajectoryIndex = 0;
  auto lastFrameTime = std::chrono::steady_clock::now();
  constexpr auto frameDuration = std::chrono::milliseconds(90);
  while (viewer.isOpen()) {
    const auto now = std::chrono::steady_clock::now();
    if (now - lastFrameTime >= frameDuration) {
      trajectoryIndex = (trajectoryIndex + 1) % stateTrajectory.size();
      lastFrameTime = now;
    }

    setViewerQ(viewer, stateTrajectory[trajectoryIndex]);
    viewer.updateScene();
    appendWall(viewer, wallX, visualOffset);
    appendContactVisualization(viewer, pinocchioInterface, stateConverter, surfaceConstraint,
                               stateTrajectory[trajectoryIndex], wallX, visualOffset);
    viewer.drawScene();
    viewer.pollEvents();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool noViewer = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--no-viewer") {
      noViewer = true;
    }
  }

  const std::string packageShare =
      ament_index_cpp::get_package_share_directory("ocp_constraint_body_contact_sample");
  const std::string robotDir = packageShare + "/unitree_ros/robots/g1_description";
  const std::string urdfFile = robotDir + "/g1_29dof.urdf";
  const std::string mujocoModelFile = makeFixedRootMujocoXml(robotDir);
  const std::string contactMeshFile = robotDir + "/meshes/left_hip_yaw_link.STL";
  const point_mesh_projector::PointMeshProjector contactPointProjector(contactMeshFile);
  constexpr double wallX = 0.2;

  std::vector<ocp_solver::ContactCandidate> contactCandidates;
  {
    ocp_solver::ContactCandidate candidate;
    candidate.frameName = "left_hip_yaw_contact";
    candidate.parentJointName = "left_hip_yaw_link";
    const assimp_eigen::MeshData mesh = assimp_eigen::loadMesh(contactMeshFile);
    const Eigen::Vector3d defaultLocalPoint = contactPointProjector.project(Eigen::Vector3d(-0.1, 0.05, -0.05));
    const Eigen::Matrix3d contactRotation = makeContactRotationWithInwardZ(mesh, defaultLocalPoint);
    candidate.localPose = pinocchio::SE3(contactRotation, defaultLocalPoint);
    candidate.searchContactPoint = true;
    candidate.alignContactFrameWithMeshNormal = true;
    candidate.meshNormalSubmeshRadius = 0.12;
    candidate.meshVerticesInLocalFrame = mesh.vertices;
    candidate.meshNormalsInLocalFrame = mesh.normals;
    std::cout << "initial inward contact z in local frame: "
              << contactRotation.col(2).transpose() << "\n";
    std::cout << "initial inward contact z dot center direction: "
              << contactRotation.col(2).dot((computeMeshCenter(mesh) - defaultLocalPoint).normalized()) << "\n";
    contactCandidates.push_back(candidate);
  }

  ocp_solver::OCPInterface interface;
  pinocchio::JointModelComposite fixedRoot;
  const std::vector<std::string> fixedJointNames = {
      "floating_base_joint",
      "left_hip_pitch_joint",
      "left_hip_roll_joint",
      "left_hip_yaw_joint",
      "left_knee_joint",
      "left_ankle_pitch_joint",
      "left_ankle_roll_joint"};
  interface.initialize("contact_search_visualizer", urdfFile, fixedJointNames, false,
                       contactCandidates, fixedRoot);
  interface.sqpSettings().sqpIteration = 160;
  interface.sqpSettings().printSolverStatistics = true;
  interface.sqpSettings().printSolverStatus = false;
  interface.sqpSettings().printLinesearch = false;

  ocs2::vector_t initialState = ocs2::vector_t::Zero(interface.getStateConverter().getStateDim());
  ocs2::vector_t initialInput = ocs2::vector_t::Zero(interface.getStateConverter().getInputDim());
  initialState.segment<3>(interface.getStateConverter().getContactPointLocalPositionStartIndex(0)) =
      interface.getStateConverter().getDefaultContactPointLocalPosition(0);

  ocs2::PinocchioInterface& pinocchioInterface = interface.getPinocchioInterface();
  updatePinocchioKinematics(pinocchioInterface, initialState);
  const pinocchio::SE3 initialContactPose =
      ocp_solver::getContactCandidatePlacement(pinocchioInterface,
                                               interface.getStateConverter().getContactCandidate(initialState, 0));
  const Eigen::Vector3d targetPosition(wallX, initialContactPose.translation().y(), initialContactPose.translation().z());
  std::cout << "initial contact position: " << initialContactPose.translation().transpose() << "\n";
  std::cout << "target wall position: " << targetPosition.transpose() << "\n";

  {
    ocp_constraint::PositionConstraint::Config config;
    config.Ax = ocs2::matrix_t::Zero(3, 6);
    config.Ax(0, 0) = 1.0;
    auto frameDynamics = std::make_unique<ocp_solver::PinocchioFrameDynamics>(interface.getStateConverter(), 0);
    interface.getOptimalControlProblem().softConstraintPtr->add(
        "left_hip_yaw_contact_wall_position",
        std::make_unique<ocs2::StateInputSoftConstraint>(
            std::make_unique<ocp_constraint::PositionConstraint>(
                *frameDynamics, 3, config, pinocchio::SE3(Eigen::Matrix3d::Identity(), targetPosition)),
            std::make_unique<ocs2::QuadraticPenalty>(1e5)));
  }

  interface.getOptimalControlProblem().equalityConstraintPtr->add(
      "left_hip_yaw_contact_search",
      std::make_unique<ocp_constraint_body_contact::ContactPointSearchConstraint>(
          *interface.getReferenceManagerPtr(), 0, interface.getStateConverter(), contactMeshFile));

  ocp_constraint_body_contact::AssumedSurfaceContactConstraint* surfaceConstraintRaw = nullptr;
  {
    ocs2::PieceWisePolynomialBarrierPenalty::Config surfacePenalty(1e4, 1e-4);
    ocp_constraint_body_contact::AssumedSurfaceContactConstraint::Config surfaceConfig;
    surfaceConfig.ellipseSafetyMargin = 0.01;
    surfaceConfig.ellipseScale = 1.2;
    surfaceConfig.frictionCoef = 5.0;
    auto surfaceConstraint = std::make_unique<ocp_constraint_body_contact::AssumedSurfaceContactConstraint>(
        *interface.getReferenceManagerPtr(), 0, interface.getStateConverter(), contactMeshFile, surfaceConfig);
    surfaceConstraintRaw = surfaceConstraint.get();
    interface.getOptimalControlProblem().softConstraintPtr->add(
        "left_hip_yaw_assumed_surface",
        std::make_unique<ocs2::StateInputSoftConstraint>(
            std::move(surfaceConstraint),
            std::make_unique<ocs2::PieceWisePolynomialBarrierPenalty>(surfacePenalty)));
  }

  {
    ocs2::matrix_t Q = ocs2::matrix_t::Identity(interface.getStateConverter().getStateVariableDim(),
                                                interface.getStateConverter().getStateVariableDim()) * 1e-2;
    Q.diagonal().segment(interface.getStateConverter().getContactPointLocalPositionVariableStartIndex(0), 3).array() = 1e4;
    ocs2::matrix_t R = ocs2::matrix_t::Identity(interface.getStateConverter().getInputDim(),
                                                interface.getStateConverter().getInputDim()) * 1e-4;
    R.diagonal().head<6>().array() = 1e-6;
    interface.getOptimalControlProblem().costPtr->add(
        "regularization",
        std::make_unique<ocp_solver::QuadraticStateInputCost>(interface.getPinocchioInterface(), Q, R));
  }

  const ocs2::scalar_array_t timeTrajectory{0.0};
  const ocs2::vector_array_t stateTrajectory{initialState};
  const ocs2::vector_array_t inputTrajectory{initialInput};
  interface.getReferenceManagerPtr()->setTargetTrajectories({timeTrajectory, stateTrajectory, inputTrajectory});
  interface.getReferenceManagerPtr()->setContactSchedule(
      ocp_solver::ContactSchedule({}, {{{0, initialContactPose}}}));
  interface.getReferenceManagerPtr()->preSolverRun(0.0, 0.0, initialState);

  auto optimizer = interface.createPoseOptimizer();
  optimizer->setMaxLinesearchStepSize(0.03);
  optimizer->addStateProjection(
      [&interface, &contactPointProjector](ocs2::vector_t& state) {
        projectContactPointToMesh(interface.getStateConverter(), contactPointProjector, state);
      });
  const auto start = std::chrono::high_resolution_clock::now();
  const ocp_solver::PoseOptimizerResult result = optimizer->run(0.0, initialState, initialInput);
  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  updatePinocchioKinematics(pinocchioInterface, result.state);
  const pinocchio::SE3 finalContactPose =
      ocp_solver::getContactCandidatePlacement(pinocchioInterface,
                                               interface.getStateConverter().getContactCandidate(result.state, 0));
  std::cout << "elapsed: " << elapsed.count() << " us\n";
  std::cout << "iterations: " << result.iterations << "\n";
  std::cout << "success: " << result.success << "\n";
  std::cout << "constraint violation: "
            << ocs2::FilterLinesearch::totalConstraintViolation(result.performance) << "\n";
  std::cout << "final contact position: " << finalContactPose.translation().transpose() << "\n";
  std::cout << "target error: " << (finalContactPose.translation() - targetPosition).transpose() << "\n";
  std::cout << "default local point: "
            << interface.getStateConverter().getDefaultContactPointLocalPosition(0).transpose() << "\n";
  std::cout << "searched local point: "
            << interface.getStateConverter().getContactPointLocalPosition(result.state, 0).transpose() << "\n";
  std::cout << "local projection error: "
            << (interface.getStateConverter().getContactPointLocalPosition(result.state, 0)
                - contactPointProjector.project(interface.getStateConverter().getContactPointLocalPosition(result.state, 0))).norm() << "\n";

  if (!noViewer && surfaceConstraintRaw) {
    visualizeTrajectory(mujocoModelFile, result.stateTrajectory, pinocchioInterface,
                        interface.getStateConverter(), *surfaceConstraintRaw, wallX);
  }

  return result.success ? 0 : 1;
}

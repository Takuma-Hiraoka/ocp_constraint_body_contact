#include "ocp_constraint_body_contact/assumed_surface_contact_constraint.h"

#include <assimp_eigen/assimp_eigen.h>
#include <ocp_solver/common/scope_profiler.h>
#include <ocp_solver/solver/dynamics_helper_functions.h>
#include <ocp_solver/solver/ocp_pre_computation.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace ocp_constraint_body_contact {
namespace {

  Eigen::Matrix3d tangentProjection(const Eigen::Vector3d& normal) {
    return Eigen::Matrix3d::Identity() - normal * normal.transpose();
  }

  struct VoxelKey {
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const VoxelKey& rhs) const {
      return x == rhs.x && y == rhs.y && z == rhs.z;
    }
    bool operator<(const VoxelKey& rhs) const {
      if (x != rhs.x) {
        return x < rhs.x;
      }
      if (y != rhs.y) {
        return y < rhs.y;
      }
      return z < rhs.z;
    }
  };

  struct VoxelAccumulator {
    Eigen::Vector3d vertexSum = Eigen::Vector3d::Zero();
    Eigen::Vector3d normalSum = Eigen::Vector3d::Zero();
    size_t count = 0;
  };

  VoxelKey makeVoxelKey(const Eigen::Vector3d& vertex, ocs2::scalar_t voxelSize) {
    return {
      static_cast<int>(std::floor(vertex.x() / voxelSize)),
      static_cast<int>(std::floor(vertex.y() / voxelSize)),
      static_cast<int>(std::floor(vertex.z() / voxelSize)),
    };
  }

  void voxelGridAverageMesh(const std::vector<Eigen::Vector3d>& inputVertices,
                            const std::vector<Eigen::Vector3d>& inputNormals,
                            ocs2::scalar_t voxelSize,
                            std::vector<Eigen::Vector3d>& outputVertices,
                            std::vector<Eigen::Vector3d>& outputNormals) {
    outputVertices.clear();
    outputNormals.clear();
    if (voxelSize <= 0.0 || inputVertices.empty()) {
      outputVertices = inputVertices;
      outputNormals = inputNormals;
      return;
    }

    std::map<VoxelKey, VoxelAccumulator> voxels;
    for (size_t i = 0; i < inputVertices.size(); ++i) {
      const Eigen::Vector3d& vertex = inputVertices[i];
      if (!vertex.allFinite()) {
        continue;
      }
      VoxelAccumulator& accumulator = voxels[makeVoxelKey(vertex, voxelSize)];
      accumulator.vertexSum += vertex;
      if (i < inputNormals.size() && inputNormals[i].allFinite()) {
        accumulator.normalSum += inputNormals[i];
      }
      ++accumulator.count;
    }

    outputVertices.reserve(voxels.size());
    outputNormals.reserve(voxels.size());
    for (const auto& voxel : voxels) {
      const VoxelAccumulator& accumulator = voxel.second;
      if (accumulator.count == 0) {
        continue;
      }
      outputVertices.push_back(accumulator.vertexSum / static_cast<double>(accumulator.count));
      Eigen::Vector3d normal = accumulator.normalSum;
      if (!normal.allFinite() || normal.squaredNorm() < 1e-12) {
        normal = Eigen::Vector3d::UnitZ();
      } else {
        normal.normalize();
      }
      outputNormals.push_back(normal);
    }
  }

  Eigen::Vector3d normalAlignedWithForce(const Eigen::Vector3d& normal,
                                         const Eigen::Vector3d& localForce) {
    return normal.dot(localForce) < 0.0 ? -normal : normal;
  }

  Eigen::Matrix3d inversePositiveDefiniteMetric(const Eigen::Matrix3d& matrix,
                                                ocs2::scalar_t scale) {
    const Eigen::Matrix3d symmetricMatrix = 0.5 * (matrix + matrix.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(symmetricMatrix);
    if (solver.info() != Eigen::Success) {
      return Eigen::Matrix3d::Identity() / (scale * scale);
    }
    const Eigen::Vector3d inverseEigenvalues =
      solver.eigenvalues().cwiseMax(1e-9).cwiseInverse() / (scale * scale);
    return solver.eigenvectors() * inverseEigenvalues.asDiagonal() * solver.eigenvectors().transpose();
  }

  Eigen::Matrix3d makeContactRotationFromNormal(const Eigen::Matrix3d& referenceRotation,
                                                Eigen::Vector3d normal) {
    if (!normal.allFinite() || normal.squaredNorm() < 1e-12) {
      return referenceRotation;
    }
    normal.normalize();
    if (normal.dot(referenceRotation.col(2)) < 0.0) {
      normal = -normal;
    }

    Eigen::Vector3d xAxis = referenceRotation.col(0) - normal * normal.dot(referenceRotation.col(0));
    if (!xAxis.allFinite() || xAxis.squaredNorm() < 1e-12) {
      xAxis = referenceRotation.col(1) - normal * normal.dot(referenceRotation.col(1));
    }
    if (!xAxis.allFinite() || xAxis.squaredNorm() < 1e-12) {
      xAxis = normal.unitOrthogonal();
    }
    xAxis.normalize();
    Eigen::Vector3d yAxis = normal.cross(xAxis).normalized();
    xAxis = yAxis.cross(normal).normalized();

    Eigen::Matrix3d rotation;
    rotation.col(0) = xAxis;
    rotation.col(1) = yAxis;
    rotation.col(2) = normal;
    return rotation;
  }

}  // namespace

  AssumedSurfaceContactConstraint::AssumedSurfaceContactConstraint(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                                                                   size_t contactIndex,
                                                                   const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
                                                                   const std::string& meshFile,
                                                                   Config config)
    : StateInputConstraint(ocs2::ConstraintOrder::Linear),
      stateConverterPtr_(&stateConverter),
      referenceManagerPtr_(&referenceManager),
      contactIndex_(contactIndex),
      config_(std::move(config)),
      defaultContactPoseInParent_(stateConverterPtr_->getContactCandidate(contactIndex_).localPose),
      defaultContactPoseInLocalFrame_(stateConverterPtr_->getContactCandidate(contactIndex_).localPoseInLocalFrame) {
    assimp_eigen::MeshData mesh = assimp_eigen::loadMesh(meshFile);
    for (Eigen::Vector3d& vertex : mesh.vertices) {
      vertex = config_.meshPoseInLocalFrame.act(vertex);
    }
    for (Eigen::Vector3d& normal : mesh.normals) {
      normal = config_.meshPoseInLocalFrame.rotation() * normal;
    }
    const Eigen::Vector3d fallbackNormalInLocalFrame =
      defaultContactPoseInLocalFrame_.rotation() * config_.surfaceNormalInContactFrame;
    std::vector<Eigen::Vector3d> meshNormals = mesh.normals;
    if (meshNormals.size() < mesh.vertices.size()) {
      meshNormals.resize(mesh.vertices.size(), fallbackNormalInLocalFrame);
    }
    for (Eigen::Vector3d& normal : meshNormals) {
      if (!normal.allFinite() || normal.squaredNorm() < 1e-12) {
        normal = fallbackNormalInLocalFrame;
      }
      if (!normal.allFinite() || normal.squaredNorm() < 1e-12) {
        normal = Eigen::Vector3d::UnitZ();
      }
      normal.normalize();
    }
    voxelGridAverageMesh(mesh.vertices, meshNormals, config_.meshVoxelGridSize, vertices_, normals_);
    if (vertices_.empty()) {
      throw std::runtime_error("AssumedSurfaceContactConstraint: mesh has no vertices: " + meshFile);
    }
    const auto geometryComputationStart = std::chrono::steady_clock::now();
    computeWeightedGeometry();
    initialWeightedGeometryComputationTimeMs_ = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - geometryComputationStart).count();
  }

  AssumedSurfaceContactConstraint::AssumedSurfaceContactConstraint(const AssumedSurfaceContactConstraint& rhs)
    : StateInputConstraint(rhs),
      stateConverterPtr_(rhs.stateConverterPtr_->clone()),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      contactIndex_(rhs.contactIndex_),
      config_(rhs.config_),
      defaultContactPoseInParent_(rhs.defaultContactPoseInParent_),
      defaultContactPoseInLocalFrame_(rhs.defaultContactPoseInLocalFrame_),
      vertices_(rhs.vertices_),
      normals_(rhs.normals_),
      normalInContactFrame_(rhs.normalInContactFrame_),
      surfaceNormalInContactFrame_(rhs.surfaceNormalInContactFrame_),
      geometricCenter_(rhs.geometricCenter_),
      geometricCenterInContactPlane_(rhs.geometricCenterInContactPlane_),
      covariance_(rhs.covariance_),
      ellipseMetric_(rhs.ellipseMetric_),
      initialWeightedGeometryComputationTimeMs_(rhs.initialWeightedGeometryComputationTimeMs_) {}

  void AssumedSurfaceContactConstraint::computeWeightedGeometry() {
    normalInContactFrame_ = config_.contactNormalInContactFrame;
    if (!normalInContactFrame_.allFinite() || normalInContactFrame_.squaredNorm() < 1e-12) {
      throw std::runtime_error("AssumedSurfaceContactConstraint: contactNormalInContactFrame must be non-zero.");
    }
    normalInContactFrame_.normalize();
    surfaceNormalInContactFrame_ = config_.surfaceNormalInContactFrame;
    if (!surfaceNormalInContactFrame_.allFinite() || surfaceNormalInContactFrame_.squaredNorm() < 1e-12) {
      throw std::runtime_error("AssumedSurfaceContactConstraint: surfaceNormalInContactFrame must be non-zero.");
    }
    surfaceNormalInContactFrame_.normalize();

    ocs2::scalar_t maxSignedDistance = -std::numeric_limits<ocs2::scalar_t>::infinity();
    const pinocchio::SE3 localFramePoseInContact = defaultContactPoseInLocalFrame_.inverse();
    for (const Eigen::Vector3d& vertex : vertices_) {
      const Eigen::Vector3d vertexInContactFrame = localFramePoseInContact.act(vertex);
      maxSignedDistance = std::max(maxSignedDistance, surfaceNormalInContactFrame_.dot(vertexInContactFrame));
    }

    ocs2::scalar_t weightSum = 0.0;
    geometricCenter_.setZero();
    const double proximityLengthScale = std::max<ocs2::scalar_t>(config_.proximityLengthScale, 1.0e-9);
    const double invTwoProximitySigma2 = 0.5 / (proximityLengthScale * proximityLengthScale);
    for (const Eigen::Vector3d& vertex : vertices_) {
      const Eigen::Vector3d vertexInContactFrame = localFramePoseInContact.act(vertex);
      const ocs2::scalar_t signedDistance = surfaceNormalInContactFrame_.dot(vertexInContactFrame);
      const ocs2::scalar_t weight =
        std::exp(config_.normalWeightScale * (signedDistance - maxSignedDistance))
        * std::exp(-vertexInContactFrame.squaredNorm() * invTwoProximitySigma2);
      weightSum += weight;
      geometricCenter_.noalias() += weight * vertexInContactFrame;
    }
    if (weightSum <= 0.0) {
      throw std::runtime_error("AssumedSurfaceContactConstraint: weighted mesh geometry is degenerate.");
    }
    geometricCenter_ /= weightSum;

    covariance_.setZero();
    for (const Eigen::Vector3d& vertex : vertices_) {
      const Eigen::Vector3d vertexInContactFrame = localFramePoseInContact.act(vertex);
      const ocs2::scalar_t signedDistance = surfaceNormalInContactFrame_.dot(vertexInContactFrame);
      const ocs2::scalar_t weight =
        std::exp(config_.normalWeightScale * (signedDistance - maxSignedDistance))
        * std::exp(-vertexInContactFrame.squaredNorm() * invTwoProximitySigma2);
      const Eigen::Vector3d delta = vertexInContactFrame - geometricCenter_;
      covariance_.noalias() += weight * delta * delta.transpose();
    }
    covariance_ /= weightSum;

    const Eigen::Matrix3d projection = tangentProjection(normalInContactFrame_);
    geometricCenterInContactPlane_ = projection * geometricCenter_;
    const Eigen::Matrix3d tangentCovariance =
      projection * covariance_ * projection
      + config_.covarianceRegularization * projection
      + normalInContactFrame_ * normalInContactFrame_.transpose();
    ellipseMetric_ = inversePositiveDefiniteMetric(tangentCovariance, config_.ellipseScale);
  }

  AssumedSurfaceContactConstraint::SurfaceGeometry AssumedSurfaceContactConstraint::computeWeightedGeometry(
      const Eigen::Vector3d& normalInContactFrame,
      const pinocchio::SE3& contactPoseInLocalFrame) const {
    return computeWeightedGeometry(normalInContactFrame, surfaceNormalInContactFrame_, contactPoseInLocalFrame);
  }

  AssumedSurfaceContactConstraint::SurfaceGeometry AssumedSurfaceContactConstraint::computeWeightedGeometry(
      const Eigen::Vector3d& normalInContactFrame,
      const Eigen::Vector3d& surfaceWeightNormalInContactFrame,
      const pinocchio::SE3& contactPoseInLocalFrame) const {
    SurfaceGeometry geometry;
    geometry.normalInContactFrame = normalInContactFrame;
    if (!geometry.normalInContactFrame.allFinite() || geometry.normalInContactFrame.squaredNorm() < 1e-12) {
      geometry.normalInContactFrame = normalInContactFrame_;
    }
    geometry.normalInContactFrame.normalize();
    Eigen::Vector3d surfaceGeometryWeightNormalInContactFrame = surfaceWeightNormalInContactFrame;
    if (!surfaceGeometryWeightNormalInContactFrame.allFinite()
        || surfaceGeometryWeightNormalInContactFrame.squaredNorm() < 1e-12) {
      surfaceGeometryWeightNormalInContactFrame = surfaceNormalInContactFrame_;
    }
    surfaceGeometryWeightNormalInContactFrame.normalize();

    ocs2::scalar_t maxSignedDistance = -std::numeric_limits<ocs2::scalar_t>::infinity();
    const pinocchio::SE3 localFramePoseInContact = contactPoseInLocalFrame.inverse();
    for (const Eigen::Vector3d& vertex : vertices_) {
      const Eigen::Vector3d vertexInContactFrame = localFramePoseInContact.act(vertex);
      maxSignedDistance = std::max(maxSignedDistance, surfaceGeometryWeightNormalInContactFrame.dot(vertexInContactFrame));
    }

    ocs2::scalar_t weightSum = 0.0;
    Eigen::Vector3d geometricCenter = Eigen::Vector3d::Zero();
    const double proximityLengthScale = std::max<ocs2::scalar_t>(config_.proximityLengthScale, 1.0e-9);
    const double invTwoProximitySigma2 = 0.5 / (proximityLengthScale * proximityLengthScale);
    for (const Eigen::Vector3d& vertex : vertices_) {
      const Eigen::Vector3d vertexInContactFrame = localFramePoseInContact.act(vertex);
      const ocs2::scalar_t signedDistance = surfaceGeometryWeightNormalInContactFrame.dot(vertexInContactFrame);
      const ocs2::scalar_t weight =
        std::exp(config_.normalWeightScale * (signedDistance - maxSignedDistance))
        * std::exp(-vertexInContactFrame.squaredNorm() * invTwoProximitySigma2);
      weightSum += weight;
      geometricCenter.noalias() += weight * vertexInContactFrame;
    }
    if (weightSum <= 0.0) {
      throw std::runtime_error("AssumedSurfaceContactConstraint: weighted mesh geometry is degenerate.");
    }
    geometricCenter /= weightSum;

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (const Eigen::Vector3d& vertex : vertices_) {
      const Eigen::Vector3d vertexInContactFrame = localFramePoseInContact.act(vertex);
      const ocs2::scalar_t signedDistance = surfaceGeometryWeightNormalInContactFrame.dot(vertexInContactFrame);
      const ocs2::scalar_t weight =
        std::exp(config_.normalWeightScale * (signedDistance - maxSignedDistance))
        * std::exp(-vertexInContactFrame.squaredNorm() * invTwoProximitySigma2);
      const Eigen::Vector3d delta = vertexInContactFrame - geometricCenter;
      covariance.noalias() += weight * delta * delta.transpose();
    }
    covariance /= weightSum;

    const Eigen::Matrix3d projection = tangentProjection(geometry.normalInContactFrame);
    geometry.geometricCenterInContactPlane = projection * geometricCenter;
    const Eigen::Matrix3d tangentCovariance =
      projection * covariance * projection
      + config_.covarianceRegularization * projection
      + geometry.normalInContactFrame * geometry.normalInContactFrame.transpose();
    geometry.ellipseMetric = inversePositiveDefiniteMetric(tangentCovariance, config_.ellipseScale);
    return geometry;
  }

  bool AssumedSurfaceContactConstraint::isActive(ocs2::scalar_t time) const {
    return referenceManagerPtr_->isInContact(time, stateConverterPtr_->getContactCandidateIds()[contactIndex_]);
  }

  pinocchio::SE3 AssumedSurfaceContactConstraint::getTargetPose(ocs2::scalar_t time) const {
    for (const auto& contact : referenceManagerPtr_->getContacts(time)) {
      if (contact.first == stateConverterPtr_->getContactCandidateIds()[contactIndex_]) {
        return contact.second.getTargetPose(time);
      }
    }
    return pinocchio::SE3::Identity();
  }

  Eigen::Vector3d AssumedSurfaceContactConstraint::computeMeshNormalInLocalFrame(
      const Eigen::Vector3d& contactPointInLocalFrame) const {
    if (vertices_.empty() || normals_.empty()) {
      return defaultContactPoseInLocalFrame_.rotation() * normalInContactFrame_;
    }

    Eigen::Vector3d weightedNormalInLocalFrame = Eigen::Vector3d::Zero();
    double weightSum = 0.0;
    const double lengthScale = 0.02;
    const double invTwoSigma2 = 0.5 / (lengthScale * lengthScale);
    for (size_t i = 0; i < vertices_.size(); ++i) {
      const double squaredDistance = (vertices_[i] - contactPointInLocalFrame).squaredNorm();
      const double weight = std::exp(-squaredDistance * invTwoSigma2);
      const Eigen::Vector3d normal = normals_[std::min(i, normals_.size() - 1)];
      if (normal.allFinite() && normal.squaredNorm() > 1e-12) {
        weightedNormalInLocalFrame.noalias() += weight * normal.normalized();
        weightSum += weight;
      }
    }

    Eigen::Vector3d normalInLocalFrame = Eigen::Vector3d::Zero();
    if (weightSum > 0.0) {
      normalInLocalFrame = weightedNormalInLocalFrame / weightSum;
    }
    if (!normalInLocalFrame.allFinite() || normalInLocalFrame.squaredNorm() < 1e-12) {
      normalInLocalFrame = defaultContactPoseInLocalFrame_.rotation() * normalInContactFrame_;
    }
    if (!normalInLocalFrame.allFinite() || normalInLocalFrame.squaredNorm() < 1e-12) {
      normalInLocalFrame = Eigen::Vector3d::UnitZ();
    }
    return normalInLocalFrame.normalized();
  }

  Eigen::Vector3d AssumedSurfaceContactConstraint::computeMeshNormalInContactFrame(
      const pinocchio::SE3& contactPoseInLocalFrame) const {
    const Eigen::Vector3d normalInLocalFrame =
      computeMeshNormalInLocalFrame(contactPoseInLocalFrame.translation());
    Eigen::Vector3d normalInContactFrame =
      contactPoseInLocalFrame.rotation().transpose() * normalInLocalFrame;
    if (!normalInContactFrame.allFinite() || normalInContactFrame.squaredNorm() < 1e-12) {
      return normalInContactFrame_;
    }
    normalInContactFrame.normalize();
    if (normalInContactFrame.dot(normalInContactFrame_) < 0.0) {
      normalInContactFrame = -normalInContactFrame;
    }
    return normalInContactFrame;
  }

  AssumedSurfaceContactConstraint::SurfaceGeometry AssumedSurfaceContactConstraint::getSurfaceGeometry(
      ocs2::scalar_t time,
      const ocs2::vector_t& state,
      ocs2::PinocchioInterface& pinocchioInterface) const {
    if (!stateConverterPtr_->getContactCandidate(contactIndex_).searchContactPoint) {
      SurfaceGeometry geometry;
      geometry.normalInContactFrame = normalInContactFrame_;
      geometry.geometricCenterInContactPlane = geometricCenterInContactPlane_;
      geometry.ellipseMetric = ellipseMetric_;
      return geometry;
    }

    const auto contactCandidate = stateConverterPtr_->getContactCandidate(state, contactIndex_);
    pinocchio::SE3 contactPoseInLocalFrame = contactCandidate.localPoseInLocalFrame;
    const Eigen::Vector3d meshNormalInLocalFrame =
      computeMeshNormalInLocalFrame(contactPoseInLocalFrame.translation());
    if (contactCandidate.alignContactFrameWithMeshNormal
        && contactCandidate.meshVerticesInLocalFrame
        && contactCandidate.meshNormalsInLocalFrame
        && !contactCandidate.meshVerticesInLocalFrame->empty()
        && !contactCandidate.meshNormalsInLocalFrame->empty()) {
      contactPoseInLocalFrame.rotation() =
        makeContactRotationFromNormal(contactPoseInLocalFrame.rotation(),
                                      meshNormalInLocalFrame);
    }
    Eigen::Vector3d surfaceWeightNormalInContactFrame =
      contactPoseInLocalFrame.rotation().transpose() * meshNormalInLocalFrame;
    if (!surfaceWeightNormalInContactFrame.allFinite()
        || surfaceWeightNormalInContactFrame.squaredNorm() < 1e-12) {
      surfaceWeightNormalInContactFrame = surfaceNormalInContactFrame_;
    }
    surfaceWeightNormalInContactFrame.normalize();
    Eigen::Vector3d normalInContactFrame = surfaceWeightNormalInContactFrame;
    if (normalInContactFrame.dot(normalInContactFrame_) < 0.0) {
      normalInContactFrame = -normalInContactFrame;
    }
    return computeWeightedGeometry(normalInContactFrame,
                                   surfaceWeightNormalInContactFrame,
                                   contactPoseInLocalFrame);
  }

  Eigen::Vector3d AssumedSurfaceContactConstraint::getGeometricCenterInContactPlane(const ocs2::vector_t& state) const {
    const auto contactCandidate = stateConverterPtr_->getContactCandidate(state, contactIndex_);
    return computeWeightedGeometry(normalInContactFrame_, contactCandidate.localPoseInLocalFrame).geometricCenterInContactPlane;
  }

  Eigen::Vector3d AssumedSurfaceContactConstraint::computePressureCenterInContactFrame(
      const Eigen::Matrix3d& contactRotation,
      const Eigen::Matrix<ocs2::scalar_t, 6, 1>& localWorldAlignedWrench) const {
    return computePressureCenterInContactFrame(contactRotation, localWorldAlignedWrench, normalInContactFrame_);
  }

  Eigen::Vector3d AssumedSurfaceContactConstraint::computePressureCenterInContactFrame(
      const Eigen::Matrix3d& contactRotation,
      const Eigen::Matrix<ocs2::scalar_t, 6, 1>& localWorldAlignedWrench,
      const Eigen::Vector3d& normalInContactFrame) const {
    Eigen::Matrix<ocs2::scalar_t, 6, 6> worldToContact = Eigen::Matrix<ocs2::scalar_t, 6, 6>::Zero();
    worldToContact.block<3, 3>(0, 0) = contactRotation.transpose();
    worldToContact.block<3, 3>(3, 3) = contactRotation.transpose();
    const Eigen::Matrix<ocs2::scalar_t, 6, 1> localWrench = worldToContact * localWorldAlignedWrench;
    const Eigen::Vector3d localForce = localWrench.head<3>();
    const Eigen::Vector3d localMoment = localWrench.tail<3>();
    const Eigen::Vector3d forceNormalInContactFrame =
      normalAlignedWithForce(normalInContactFrame.normalized(), localForce);

    const ocs2::scalar_t normalForce =
      forceNormalInContactFrame.dot(localForce) + config_.normalForceRegularization;
    return forceNormalInContactFrame.cross(localMoment) / normalForce;
  }

  ocs2::vector_t AssumedSurfaceContactConstraint::getValue(ocs2::scalar_t time,
                                                           const ocs2::vector_t& state,
                                                           const ocs2::vector_t& input,
                                                           const ocs2::PreComputation& preComp) const {
    OCP_SOLVER_PROFILE_SCOPE("AssumedSurfaceContactConstraint::getValue");
    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
    const Eigen::Matrix3d R_frame =
      ocp_solver::getContactCandidatePlacement(pinocchioInterface, stateConverterPtr_->getContactCandidate(state, contactIndex_)).rotation();
    const SurfaceGeometry surfaceGeometry = getSurfaceGeometry(time, state, pinocchioInterface);
    Eigen::Matrix<ocs2::scalar_t, 6, 6> worldToContact = Eigen::Matrix<ocs2::scalar_t, 6, 6>::Zero();
    worldToContact.block<3, 3>(0, 0) = R_frame.transpose();
    worldToContact.block<3, 3>(3, 3) = R_frame.transpose();
    const Eigen::Matrix<ocs2::scalar_t, 6, 1> localWrench =
      worldToContact * stateConverterPtr_->getContactWrench(input, contactIndex_);
    const Eigen::Vector3d forceNormalInContactFrame =
      normalAlignedWithForce(surfaceGeometry.normalInContactFrame.normalized(), localWrench.head<3>());
    const ocs2::scalar_t normalForce = forceNormalInContactFrame.dot(localWrench.head<3>());
    const Eigen::Vector3d tangent0 = forceNormalInContactFrame.unitOrthogonal();
    const Eigen::Vector3d tangent1 = forceNormalInContactFrame.cross(tangent0).normalized();
    const ocs2::scalar_t tangentForce0 = tangent0.dot(localWrench.head<3>());
    const ocs2::scalar_t tangentForce1 = tangent1.dot(localWrench.head<3>());
    const ocs2::scalar_t torsionalMoment = forceNormalInContactFrame.dot(localWrench.tail<3>());

    const Eigen::Vector3d pressureCenter =
      computePressureCenterInContactFrame(R_frame, stateConverterPtr_->getContactWrench(input, contactIndex_),
                                          surfaceGeometry.normalInContactFrame);
    const Eigen::Vector3d pressureCenterError =
      pressureCenter - surfaceGeometry.geometricCenterInContactPlane;

    ocs2::vector_t constraint(n_constraints);
    constraint[0] = normalForce - config_.minNormalForce;
    constraint[1] = config_.frictionCoef * normalForce + tangentForce0;
    constraint[2] = config_.frictionCoef * normalForce - tangentForce0;
    constraint[3] = config_.frictionCoef * normalForce + tangentForce1;
    constraint[4] = config_.frictionCoef * normalForce - tangentForce1;
    constraint[5] =
      1.0 - config_.ellipseSafetyMargin - pressureCenterError.dot(surfaceGeometry.ellipseMetric * pressureCenterError);
    constraint[6] = config_.rotFrictionCoef * normalForce + torsionalMoment;
    constraint[7] = config_.rotFrictionCoef * normalForce - torsionalMoment;
    return constraint;
  }

  ocs2::VectorFunctionLinearApproximation AssumedSurfaceContactConstraint::getLinearApproximation(ocs2::scalar_t time,
                                                                                                  const ocs2::vector_t& state,
                                                                                                  const ocs2::vector_t& input,
                                                                                                  const ocs2::PreComputation& preComp) const {
    OCP_SOLVER_PROFILE_SCOPE("AssumedSurfaceContactConstraint::getLinearApproximation");
    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
    const Eigen::Matrix3d R_frame =
      ocp_solver::getContactCandidatePlacement(pinocchioInterface, stateConverterPtr_->getContactCandidate(state, contactIndex_)).rotation();
    const SurfaceGeometry surfaceGeometry = getSurfaceGeometry(time, state, pinocchioInterface);

    // getContactWrench() is expressed in LOCAL_WORLD_ALIGNED axes. Convert it to the contact-local axes here.
    Eigen::Matrix<ocs2::scalar_t, 6, 6> worldToContact = Eigen::Matrix<ocs2::scalar_t, 6, 6>::Zero();
    worldToContact.block<3, 3>(0, 0) = R_frame.transpose();
    worldToContact.block<3, 3>(3, 3) = R_frame.transpose();
    const Eigen::Matrix<ocs2::scalar_t, 6, 1> worldWrench =
      stateConverterPtr_->getContactWrench(input, contactIndex_);
    const Eigen::Matrix<ocs2::scalar_t, 6, 1> localWrench = worldToContact * worldWrench;
    const Eigen::Vector3d localForce = localWrench.head<3>();
    const Eigen::Vector3d localMoment = localWrench.tail<3>();
    const Eigen::Vector3d forceNormalInContactFrame =
      normalAlignedWithForce(surfaceGeometry.normalInContactFrame.normalized(), localForce);
    const Eigen::Vector3d tangent0 = forceNormalInContactFrame.unitOrthogonal();
    const Eigen::Vector3d tangent1 = forceNormalInContactFrame.cross(tangent0).normalized();
    const ocs2::scalar_t normalForceWithoutRegularization = forceNormalInContactFrame.dot(localForce);
    const ocs2::scalar_t tangentForce0 = tangent0.dot(localForce);
    const ocs2::scalar_t tangentForce1 = tangent1.dot(localForce);
    const ocs2::scalar_t torsionalMoment = forceNormalInContactFrame.dot(localMoment);

    const ocs2::scalar_t normalForce =
      forceNormalInContactFrame.dot(localForce) + config_.normalForceRegularization;
    const Eigen::Vector3d pressureCenter =
      forceNormalInContactFrame.cross(localMoment) / normalForce;
    const Eigen::Vector3d pressureCenterError =
      pressureCenter - surfaceGeometry.geometricCenterInContactPlane;

    ocs2::VectorFunctionLinearApproximation approx;
    approx.f = ocs2::vector_t::Zero(n_constraints);
    approx.f[0] = normalForceWithoutRegularization - config_.minNormalForce;
    approx.f[1] = config_.frictionCoef * normalForceWithoutRegularization + tangentForce0;
    approx.f[2] = config_.frictionCoef * normalForceWithoutRegularization - tangentForce0;
    approx.f[3] = config_.frictionCoef * normalForceWithoutRegularization + tangentForce1;
    approx.f[4] = config_.frictionCoef * normalForceWithoutRegularization - tangentForce1;
    approx.f[5] =
      1.0 - config_.ellipseSafetyMargin - pressureCenterError.dot(surfaceGeometry.ellipseMetric * pressureCenterError);
    approx.f[6] = config_.rotFrictionCoef * normalForceWithoutRegularization + torsionalMoment;
    approx.f[7] = config_.rotFrictionCoef * normalForceWithoutRegularization - torsionalMoment;
    approx.dfdx = ocs2::matrix_t::Zero(n_constraints, stateConverterPtr_->getStateVariableDim());
    approx.dfdu = ocs2::matrix_t::Zero(n_constraints, stateConverterPtr_->getInputDim());

    const Eigen::RowVector3d dConstraintDPressureCenter =
      -2.0 * pressureCenterError.transpose() * surfaceGeometry.ellipseMetric;
    Eigen::Matrix<ocs2::scalar_t, 3, 6> dPressureCenterDLocalWrench =
      Eigen::Matrix<ocs2::scalar_t, 3, 6>::Zero();
    dPressureCenterDLocalWrench.leftCols<3>().noalias() =
      -(pressureCenter / normalForce) * forceNormalInContactFrame.transpose();
    dPressureCenterDLocalWrench.rightCols<3>().noalias() =
      ocs2::skewSymmetricMatrix(forceNormalInContactFrame) / normalForce;
    const Eigen::Matrix<ocs2::scalar_t, 1, 6> dConstraintDLocalWrench =
      dConstraintDPressureCenter * dPressureCenterDLocalWrench;

    approx.dfdu.block<1, 6>(0, stateConverterPtr_->getContactWrenchStartIndices(contactIndex_)).noalias() =
      forceNormalInContactFrame.transpose() * worldToContact.block<3, 6>(0, 0);
    approx.dfdu.block<1, 6>(1, stateConverterPtr_->getContactWrenchStartIndices(contactIndex_)).noalias() =
      (config_.frictionCoef * forceNormalInContactFrame + tangent0).transpose() * worldToContact.block<3, 6>(0, 0);
    approx.dfdu.block<1, 6>(2, stateConverterPtr_->getContactWrenchStartIndices(contactIndex_)).noalias() =
      (config_.frictionCoef * forceNormalInContactFrame - tangent0).transpose() * worldToContact.block<3, 6>(0, 0);
    approx.dfdu.block<1, 6>(3, stateConverterPtr_->getContactWrenchStartIndices(contactIndex_)).noalias() =
      (config_.frictionCoef * forceNormalInContactFrame + tangent1).transpose() * worldToContact.block<3, 6>(0, 0);
    approx.dfdu.block<1, 6>(4, stateConverterPtr_->getContactWrenchStartIndices(contactIndex_)).noalias() =
      (config_.frictionCoef * forceNormalInContactFrame - tangent1).transpose() * worldToContact.block<3, 6>(0, 0);
    approx.dfdu.block<1, 6>(5, stateConverterPtr_->getContactWrenchStartIndices(contactIndex_)).noalias() =
      dConstraintDLocalWrench * worldToContact;
    Eigen::Matrix<ocs2::scalar_t, 1, 6> dPositiveTorsionDLocalWrench =
      Eigen::Matrix<ocs2::scalar_t, 1, 6>::Zero();
    dPositiveTorsionDLocalWrench.leftCols<3>() = config_.rotFrictionCoef * forceNormalInContactFrame.transpose();
    dPositiveTorsionDLocalWrench.rightCols<3>() = forceNormalInContactFrame.transpose();
    Eigen::Matrix<ocs2::scalar_t, 1, 6> dNegativeTorsionDLocalWrench =
      Eigen::Matrix<ocs2::scalar_t, 1, 6>::Zero();
    dNegativeTorsionDLocalWrench.leftCols<3>() = config_.rotFrictionCoef * forceNormalInContactFrame.transpose();
    dNegativeTorsionDLocalWrench.rightCols<3>() = -forceNormalInContactFrame.transpose();
    approx.dfdu.block<1, 6>(6, stateConverterPtr_->getContactWrenchStartIndices(contactIndex_)).noalias() =
      dPositiveTorsionDLocalWrench * worldToContact;
    approx.dfdu.block<1, 6>(7, stateConverterPtr_->getContactWrenchStartIndices(contactIndex_)).noalias() =
      dNegativeTorsionDLocalWrench * worldToContact;
    if (stateConverterPtr_->getContactCandidate(contactIndex_).searchContactPoint) {
      const auto contactCandidate = stateConverterPtr_->getContactCandidate(state, contactIndex_);
      approx.dfdx.block<1, 3>(5, stateConverterPtr_->getContactPointLocalPositionVariableStartIndex(contactIndex_)).noalias() =
        dConstraintDPressureCenter * tangentProjection(surfaceGeometry.normalInContactFrame) * contactCandidate.localPoseInLocalFrame.rotation().transpose();
    }

    ocs2::matrix_t contactJacobian = ocs2::matrix_t::Zero(6, stateConverterPtr_->getTangentDim());
    ocp_solver::getContactCandidateJacobian(pinocchioInterface,
                                            stateConverterPtr_->getContactCandidate(state, contactIndex_),
                                            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                            contactJacobian);

    Eigen::Matrix<ocs2::scalar_t, 6, 6> dLocalWrenchDFrameRotation =
      Eigen::Matrix<ocs2::scalar_t, 6, 6>::Zero();
    dLocalWrenchDFrameRotation.block<3, 3>(0, 3) =
      R_frame.transpose() * ocs2::skewSymmetricMatrix(Eigen::Vector3d(worldWrench.head<3>()));
    dLocalWrenchDFrameRotation.block<3, 3>(3, 3) =
      R_frame.transpose() * ocs2::skewSymmetricMatrix(Eigen::Vector3d(worldWrench.tail<3>()));
    approx.dfdx.block(0, 0, 1, stateConverterPtr_->getTangentDim()).noalias() =
      forceNormalInContactFrame.transpose() * dLocalWrenchDFrameRotation.block<3, 6>(0, 0) * contactJacobian;
    approx.dfdx.block(1, 0, 1, stateConverterPtr_->getTangentDim()).noalias() =
      (config_.frictionCoef * forceNormalInContactFrame + tangent0).transpose()
      * dLocalWrenchDFrameRotation.block<3, 6>(0, 0) * contactJacobian;
    approx.dfdx.block(2, 0, 1, stateConverterPtr_->getTangentDim()).noalias() =
      (config_.frictionCoef * forceNormalInContactFrame - tangent0).transpose()
      * dLocalWrenchDFrameRotation.block<3, 6>(0, 0) * contactJacobian;
    approx.dfdx.block(3, 0, 1, stateConverterPtr_->getTangentDim()).noalias() =
      (config_.frictionCoef * forceNormalInContactFrame + tangent1).transpose()
      * dLocalWrenchDFrameRotation.block<3, 6>(0, 0) * contactJacobian;
    approx.dfdx.block(4, 0, 1, stateConverterPtr_->getTangentDim()).noalias() =
      (config_.frictionCoef * forceNormalInContactFrame - tangent1).transpose()
      * dLocalWrenchDFrameRotation.block<3, 6>(0, 0) * contactJacobian;
    approx.dfdx.block(5, 0, 1, stateConverterPtr_->getTangentDim()).noalias() =
      dConstraintDLocalWrench * dLocalWrenchDFrameRotation * contactJacobian;
    approx.dfdx.block(6, 0, 1, stateConverterPtr_->getTangentDim()).noalias() =
      dPositiveTorsionDLocalWrench * dLocalWrenchDFrameRotation * contactJacobian;
    approx.dfdx.block(7, 0, 1, stateConverterPtr_->getTangentDim()).noalias() =
      dNegativeTorsionDLocalWrench * dLocalWrenchDFrameRotation * contactJacobian;

    return approx;
  }

}

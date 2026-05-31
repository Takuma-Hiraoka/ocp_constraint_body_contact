#include "ocp_constraint_body_contact/assumed_surface_contact_constraint.h"

#include <assimp_eigen/assimp_eigen.h>
#include <ocp_solver/solver/dynamics_helper_functions.h>
#include <ocp_solver/solver/ocp_pre_computation.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ocp_constraint_body_contact {
namespace {

  Eigen::Matrix3d tangentProjection(const Eigen::Vector3d& normal) {
    return Eigen::Matrix3d::Identity() - normal * normal.transpose();
  }

  Eigen::Vector3d normalAlignedWithForce(const Eigen::Vector3d& normal,
                                         const Eigen::Vector3d& localForce) {
    return normal.dot(localForce) < 0.0 ? -normal : normal;
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
    const assimp_eigen::MeshData mesh = assimp_eigen::loadMesh(meshFile);
    vertices_.reserve(mesh.vertices.size());
    for (const Eigen::Vector3d& vertex : mesh.vertices) {
      vertices_.push_back(vertex);
    }
    if (vertices_.empty()) {
      throw std::runtime_error("AssumedSurfaceContactConstraint: mesh has no vertices: " + meshFile);
    }
    normals_ = mesh.normals;
    const Eigen::Vector3d fallbackNormalInLocalFrame =
      defaultContactPoseInLocalFrame_.rotation() * config_.surfaceNormalInContactFrame;
    if (normals_.size() < vertices_.size()) {
      normals_.resize(vertices_.size(), fallbackNormalInLocalFrame);
    }
    for (Eigen::Vector3d& normal : normals_) {
      if (!normal.allFinite() || normal.squaredNorm() < 1e-12) {
        normal = fallbackNormalInLocalFrame;
      }
      if (!normal.allFinite() || normal.squaredNorm() < 1e-12) {
        normal = Eigen::Vector3d::UnitZ();
      }
      normal.normalize();
    }
    computeWeightedGeometry();
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
      ellipseMetric_(rhs.ellipseMetric_) {}

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
    for (const Eigen::Vector3d& vertex : vertices_) {
      const Eigen::Vector3d vertexInContactFrame = localFramePoseInContact.act(vertex);
      const ocs2::scalar_t signedDistance = surfaceNormalInContactFrame_.dot(vertexInContactFrame);
      const ocs2::scalar_t weight = std::exp(config_.normalWeightScale * (signedDistance - maxSignedDistance));
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
      const ocs2::scalar_t weight = std::exp(config_.normalWeightScale * (signedDistance - maxSignedDistance));
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
    ellipseMetric_ = tangentCovariance.inverse() / (config_.ellipseScale * config_.ellipseScale);
  }

  AssumedSurfaceContactConstraint::SurfaceGeometry AssumedSurfaceContactConstraint::computeWeightedGeometry(
      const Eigen::Vector3d& normalInContactFrame,
      const pinocchio::SE3& contactPoseInLocalFrame) const {
    SurfaceGeometry geometry;
    geometry.normalInContactFrame = normalInContactFrame;
    if (!geometry.normalInContactFrame.allFinite() || geometry.normalInContactFrame.squaredNorm() < 1e-12) {
      geometry.normalInContactFrame = normalInContactFrame_;
    }
    geometry.normalInContactFrame.normalize();

    ocs2::scalar_t maxSignedDistance = -std::numeric_limits<ocs2::scalar_t>::infinity();
    const pinocchio::SE3 localFramePoseInContact = contactPoseInLocalFrame.inverse();
    for (const Eigen::Vector3d& vertex : vertices_) {
      const Eigen::Vector3d vertexInContactFrame = localFramePoseInContact.act(vertex);
      maxSignedDistance = std::max(maxSignedDistance, geometry.normalInContactFrame.dot(vertexInContactFrame));
    }

    ocs2::scalar_t weightSum = 0.0;
    Eigen::Vector3d geometricCenter = Eigen::Vector3d::Zero();
    for (const Eigen::Vector3d& vertex : vertices_) {
      const Eigen::Vector3d vertexInContactFrame = localFramePoseInContact.act(vertex);
      const ocs2::scalar_t signedDistance = geometry.normalInContactFrame.dot(vertexInContactFrame);
      const ocs2::scalar_t weight = std::exp(config_.normalWeightScale * (signedDistance - maxSignedDistance));
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
      const ocs2::scalar_t signedDistance = geometry.normalInContactFrame.dot(vertexInContactFrame);
      const ocs2::scalar_t weight = std::exp(config_.normalWeightScale * (signedDistance - maxSignedDistance));
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
    geometry.ellipseMetric = tangentCovariance.inverse() / (config_.ellipseScale * config_.ellipseScale);
    return geometry;
  }

  bool AssumedSurfaceContactConstraint::isActive(ocs2::scalar_t time) const {
    return referenceManagerPtr_->isInContact(time, stateConverterPtr_->getContactCandidateIds()[contactIndex_]);
  }

  pinocchio::SE3 AssumedSurfaceContactConstraint::getTargetPose(ocs2::scalar_t time) const {
    for (const auto& contact : referenceManagerPtr_->getContacts(time)) {
      if (contact.first == stateConverterPtr_->getContactCandidateIds()[contactIndex_]) {
        return contact.second;
      }
    }
    return pinocchio::SE3::Identity();
  }

  Eigen::Vector3d AssumedSurfaceContactConstraint::computeMeshNormalInContactFrame(
      const pinocchio::SE3& contactPoseInLocalFrame) const {
    if (vertices_.empty() || normals_.empty()) {
      return normalInContactFrame_;
    }

    const Eigen::Vector3d contactPointInLocalFrame = contactPoseInLocalFrame.translation();
    size_t nearestIndex = 0;
    double nearestDistance = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < vertices_.size(); ++i) {
      const double distance = (vertices_[i] - contactPointInLocalFrame).squaredNorm();
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestIndex = i;
      }
    }

    Eigen::Vector3d normalInContactFrame =
      contactPoseInLocalFrame.rotation().transpose() * normals_[nearestIndex];
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
    const Eigen::Vector3d normalInContactFrame =
      computeMeshNormalInContactFrame(contactCandidate.localPoseInLocalFrame);
    return computeWeightedGeometry(normalInContactFrame, contactCandidate.localPoseInLocalFrame);
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
    return constraint;
  }

  ocs2::VectorFunctionLinearApproximation AssumedSurfaceContactConstraint::getLinearApproximation(ocs2::scalar_t time,
                                                                                                  const ocs2::vector_t& state,
                                                                                                  const ocs2::vector_t& input,
                                                                                                  const ocs2::PreComputation& preComp) const {
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

    return approx;
  }

}

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
      config_(std::move(config)) {
    const assimp_eigen::MeshData mesh = assimp_eigen::loadMesh(meshFile);
    const pinocchio::SE3 contactPoseInParent =
      stateConverterPtr_->getContactCandidate(contactIndex_).localPose;
    const pinocchio::SE3 parentPoseInContact = contactPoseInParent.inverse();
    vertices_.reserve(mesh.vertices.size());
    for (const Eigen::Vector3d& vertex : mesh.vertices) {
      vertices_.push_back(parentPoseInContact.act(vertex));
    }
    if (vertices_.empty()) {
      throw std::runtime_error("AssumedSurfaceContactConstraint: mesh has no vertices: " + meshFile);
    }
    computeWeightedGeometry();
  }

  AssumedSurfaceContactConstraint::AssumedSurfaceContactConstraint(const AssumedSurfaceContactConstraint& rhs)
    : StateInputConstraint(rhs),
      stateConverterPtr_(rhs.stateConverterPtr_->clone()),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      contactIndex_(rhs.contactIndex_),
      config_(rhs.config_),
      vertices_(rhs.vertices_),
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
    for (const Eigen::Vector3d& vertex : vertices_) {
      maxSignedDistance = std::max(maxSignedDistance, surfaceNormalInContactFrame_.dot(vertex));
    }

    ocs2::scalar_t weightSum = 0.0;
    geometricCenter_.setZero();
    for (const Eigen::Vector3d& vertex : vertices_) {
      const ocs2::scalar_t signedDistance = surfaceNormalInContactFrame_.dot(vertex);
      const ocs2::scalar_t weight = std::exp(config_.normalWeightScale * (signedDistance - maxSignedDistance));
      weightSum += weight;
      geometricCenter_.noalias() += weight * vertex;
    }
    if (weightSum <= 0.0) {
      throw std::runtime_error("AssumedSurfaceContactConstraint: weighted mesh geometry is degenerate.");
    }
    geometricCenter_ /= weightSum;

    covariance_.setZero();
    for (const Eigen::Vector3d& vertex : vertices_) {
      const ocs2::scalar_t signedDistance = surfaceNormalInContactFrame_.dot(vertex);
      const ocs2::scalar_t weight = std::exp(config_.normalWeightScale * (signedDistance - maxSignedDistance));
      const Eigen::Vector3d delta = vertex - geometricCenter_;
      covariance_.noalias() += weight * delta * delta.transpose();
    }
    covariance_ /= weightSum;

    const Eigen::Matrix3d projection = tangentProjection(normalInContactFrame_);
    geometricCenterInContactPlane_ = projection * geometricCenter_;
    const Eigen::Matrix3d tangentCovariance =
      projection * covariance_ * projection
      + config_.covarianceRegularization * projection
      + normalInContactFrame_ * normalInContactFrame_.transpose();
    ellipseMetric_ = tangentCovariance.inverse();
  }

  bool AssumedSurfaceContactConstraint::isActive(ocs2::scalar_t time) const {
    return referenceManagerPtr_->isInContact(time, stateConverterPtr_->getContactCandidateIds()[contactIndex_]);
  }

  Eigen::Vector3d AssumedSurfaceContactConstraint::computePressureCenterInContactFrame(
      const Eigen::Matrix3d& contactRotation,
      const Eigen::Matrix<ocs2::scalar_t, 6, 1>& localWorldAlignedWrench) const {
    Eigen::Matrix<ocs2::scalar_t, 6, 6> worldToContact = Eigen::Matrix<ocs2::scalar_t, 6, 6>::Zero();
    worldToContact.block<3, 3>(0, 0) = contactRotation.transpose();
    worldToContact.block<3, 3>(3, 3) = contactRotation.transpose();
    const Eigen::Matrix<ocs2::scalar_t, 6, 1> localWrench = worldToContact * localWorldAlignedWrench;
    const Eigen::Vector3d localForce = localWrench.head<3>();
    const Eigen::Vector3d localMoment = localWrench.tail<3>();

    const ocs2::scalar_t normalForce =
      normalInContactFrame_.dot(localForce) + config_.normalForceRegularization;
    return normalInContactFrame_.cross(localMoment) / normalForce;
  }

  ocs2::vector_t AssumedSurfaceContactConstraint::getValue(ocs2::scalar_t time,
                                                           const ocs2::vector_t& state,
                                                           const ocs2::vector_t& input,
                                                           const ocs2::PreComputation& preComp) const {
    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
    const Eigen::Matrix3d R_frame =
      ocp_solver::getContactCandidatePlacement(pinocchioInterface, stateConverterPtr_->getContactCandidate(contactIndex_)).rotation();

    const Eigen::Vector3d pressureCenter =
      computePressureCenterInContactFrame(R_frame, stateConverterPtr_->getContactWrench(input, contactIndex_));
    const Eigen::Vector3d pressureCenterError =
      pressureCenter - geometricCenterInContactPlane_;

    ocs2::vector_t constraint(n_constraints);
    constraint[0] = 1.0 - config_.ellipseSafetyMargin - pressureCenterError.dot(ellipseMetric_ * pressureCenterError);
    return constraint;
  }

  ocs2::VectorFunctionLinearApproximation AssumedSurfaceContactConstraint::getLinearApproximation(ocs2::scalar_t time,
                                                                                                  const ocs2::vector_t& state,
                                                                                                  const ocs2::vector_t& input,
                                                                                                  const ocs2::PreComputation& preComp) const {
    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
    const Eigen::Matrix3d R_frame =
      ocp_solver::getContactCandidatePlacement(pinocchioInterface, stateConverterPtr_->getContactCandidate(contactIndex_)).rotation();

    // getContactWrench() is expressed in LOCAL_WORLD_ALIGNED axes. Convert it to the contact-local axes here.
    Eigen::Matrix<ocs2::scalar_t, 6, 6> worldToContact = Eigen::Matrix<ocs2::scalar_t, 6, 6>::Zero();
    worldToContact.block<3, 3>(0, 0) = R_frame.transpose();
    worldToContact.block<3, 3>(3, 3) = R_frame.transpose();
    const Eigen::Matrix<ocs2::scalar_t, 6, 1> worldWrench =
      stateConverterPtr_->getContactWrench(input, contactIndex_);
    const Eigen::Matrix<ocs2::scalar_t, 6, 1> localWrench = worldToContact * worldWrench;
    const Eigen::Vector3d localForce = localWrench.head<3>();
    const Eigen::Vector3d localMoment = localWrench.tail<3>();

    const ocs2::scalar_t normalForce =
      normalInContactFrame_.dot(localForce) + config_.normalForceRegularization;
    const Eigen::Vector3d pressureCenter =
      normalInContactFrame_.cross(localMoment) / normalForce;
    const Eigen::Vector3d pressureCenterError =
      pressureCenter - geometricCenterInContactPlane_;

    ocs2::VectorFunctionLinearApproximation approx;
    approx.f = ocs2::vector_t::Zero(n_constraints);
    approx.f[0] = 1.0 - config_.ellipseSafetyMargin - pressureCenterError.dot(ellipseMetric_ * pressureCenterError);
    approx.dfdx = ocs2::matrix_t::Zero(n_constraints, stateConverterPtr_->getStateVariableDim());
    approx.dfdu = ocs2::matrix_t::Zero(n_constraints, stateConverterPtr_->getInputDim());

    const Eigen::RowVector3d dConstraintDPressureCenter =
      -2.0 * pressureCenterError.transpose() * ellipseMetric_;
    Eigen::Matrix<ocs2::scalar_t, 3, 6> dPressureCenterDLocalWrench =
      Eigen::Matrix<ocs2::scalar_t, 3, 6>::Zero();
    dPressureCenterDLocalWrench.leftCols<3>().noalias() =
      -(pressureCenter / normalForce) * normalInContactFrame_.transpose();
    dPressureCenterDLocalWrench.rightCols<3>().noalias() =
      ocs2::skewSymmetricMatrix(normalInContactFrame_) / normalForce;
    const Eigen::Matrix<ocs2::scalar_t, 1, 6> dConstraintDLocalWrench =
      dConstraintDPressureCenter * dPressureCenterDLocalWrench;

    approx.dfdu.middleCols<6>(stateConverterPtr_->getContactWrenchStartIndices(contactIndex_)).noalias() =
      dConstraintDLocalWrench * worldToContact;

    ocs2::matrix_t contactJacobian = ocs2::matrix_t::Zero(6, stateConverterPtr_->getTangentDim());
    ocp_solver::getContactCandidateJacobian(pinocchioInterface,
                                            stateConverterPtr_->getContactCandidate(contactIndex_),
                                            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                            contactJacobian);

    Eigen::Matrix<ocs2::scalar_t, 6, 6> dLocalWrenchDFrameRotation =
      Eigen::Matrix<ocs2::scalar_t, 6, 6>::Zero();
    dLocalWrenchDFrameRotation.block<3, 3>(0, 3) =
      R_frame.transpose() * ocs2::skewSymmetricMatrix(Eigen::Vector3d(worldWrench.head<3>()));
    dLocalWrenchDFrameRotation.block<3, 3>(3, 3) =
      R_frame.transpose() * ocs2::skewSymmetricMatrix(Eigen::Vector3d(worldWrench.tail<3>()));
    approx.dfdx.leftCols(stateConverterPtr_->getTangentDim()).noalias() =
      dConstraintDLocalWrench * dLocalWrenchDFrameRotation * contactJacobian;

    return approx;
  }

}

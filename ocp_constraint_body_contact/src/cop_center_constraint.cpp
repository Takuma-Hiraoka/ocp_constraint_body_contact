#include "ocp_constraint_body_contact/cop_center_constraint.h"

#include <ocp_solver/common/scope_profiler.h>
#include <ocp_solver/solver/dynamics_helper_functions.h>
#include <ocp_solver/solver/ocp_pre_computation.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>

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

CopCenterConstraint::CopCenterConstraint(
    size_t contactIndex,
    const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
    const AssumedSurfaceContactConstraint& surfaceConstraint)
    : StateInputConstraint(ocs2::ConstraintOrder::Linear),
      contactIndex_(contactIndex),
      stateConverterPtr_(stateConverter.clone()),
      surfaceConstraintPtr_(&surfaceConstraint) {}

CopCenterConstraint::CopCenterConstraint(const CopCenterConstraint& rhs)
    : StateInputConstraint(rhs),
      contactIndex_(rhs.contactIndex_),
      stateConverterPtr_(rhs.stateConverterPtr_->clone()),
      surfaceConstraintPtr_(rhs.surfaceConstraintPtr_) {}

bool CopCenterConstraint::isActive(ocs2::scalar_t time) const {
  return surfaceConstraintPtr_->isActive(time);
}

size_t CopCenterConstraint::getNumConstraints(ocs2::scalar_t /*time*/) const {
  return 3;
}

ocs2::vector_t CopCenterConstraint::getValue(ocs2::scalar_t time,
                                             const ocs2::vector_t& state,
                                             const ocs2::vector_t& input,
                                             const ocs2::PreComputation& preComp) const {
  OCP_SOLVER_PROFILE_SCOPE("CopCenterConstraint::getValue");
  const ocp_solver::OCPPreComputation& ocpPreComp =
      static_cast<const ocp_solver::OCPPreComputation&>(preComp);
  ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
  const pinocchio::SE3 contactPose =
      ocp_solver::getContactCandidatePlacement(pinocchioInterface,
                                               stateConverterPtr_->getContactCandidate(state, contactIndex_));
  const auto surfaceGeometry =
      surfaceConstraintPtr_->getSurfaceGeometry(time, state, pinocchioInterface);
  const Eigen::Vector3d copLocal =
      surfaceConstraintPtr_->computePressureCenterInContactFrame(
          contactPose.rotation(), stateConverterPtr_->getContactWrench(input, contactIndex_),
          surfaceGeometry.normalInContactFrame);
  return tangentProjection(surfaceGeometry.normalInContactFrame.normalized())
         * (copLocal - surfaceGeometry.geometricCenterInContactPlane);
}

ocs2::VectorFunctionLinearApproximation CopCenterConstraint::getLinearApproximation(
    ocs2::scalar_t time,
    const ocs2::vector_t& state,
    const ocs2::vector_t& input,
    const ocs2::PreComputation& preComp) const {
  OCP_SOLVER_PROFILE_SCOPE("CopCenterConstraint::getLinearApproximation");
  const ocp_solver::OCPPreComputation& ocpPreComp =
      static_cast<const ocp_solver::OCPPreComputation&>(preComp);
  ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
  const auto contactCandidate = stateConverterPtr_->getContactCandidate(state, contactIndex_);
  const pinocchio::SE3 contactPose =
      ocp_solver::getContactCandidatePlacement(pinocchioInterface, contactCandidate);
  const auto surfaceGeometry =
      surfaceConstraintPtr_->getSurfaceGeometry(time, state, pinocchioInterface);
  const Eigen::Vector3d surfaceNormal = surfaceGeometry.normalInContactFrame.normalized();
  const Eigen::Matrix3d projection = tangentProjection(surfaceNormal);

  Eigen::Matrix<ocs2::scalar_t, 6, 6> worldToContact = Eigen::Matrix<ocs2::scalar_t, 6, 6>::Zero();
  worldToContact.block<3, 3>(0, 0) = contactPose.rotation().transpose();
  worldToContact.block<3, 3>(3, 3) = contactPose.rotation().transpose();
  const Eigen::Matrix<ocs2::scalar_t, 6, 1> worldWrench =
      stateConverterPtr_->getContactWrench(input, contactIndex_);
  const Eigen::Matrix<ocs2::scalar_t, 6, 1> localWrench = worldToContact * worldWrench;
  const Eigen::Vector3d localForce = localWrench.head<3>();
  const Eigen::Vector3d localMoment = localWrench.tail<3>();
  const Eigen::Vector3d forceNormal = normalAlignedWithForce(surfaceNormal, localForce);
  const ocs2::scalar_t normalForce = forceNormal.dot(localForce) + 1.0;
  const Eigen::Vector3d copLocal = forceNormal.cross(localMoment) / normalForce;

  Eigen::Matrix<ocs2::scalar_t, 3, 6> dCopDLocalWrench =
      Eigen::Matrix<ocs2::scalar_t, 3, 6>::Zero();
  dCopDLocalWrench.leftCols<3>().noalias() =
      -(copLocal / normalForce) * forceNormal.transpose();
  dCopDLocalWrench.rightCols<3>().noalias() =
      ocs2::skewSymmetricMatrix(forceNormal) / normalForce;

  ocs2::VectorFunctionLinearApproximation approx;
  approx.f = getValue(time, state, input, preComp);
  approx.dfdx = ocs2::matrix_t::Zero(3, stateConverterPtr_->getStateVariableDim());
  approx.dfdu = ocs2::matrix_t::Zero(3, stateConverterPtr_->getInputDim());
  approx.dfdu.block<3, 6>(0, stateConverterPtr_->getContactWrenchStartIndices(contactIndex_)).noalias() =
      projection * dCopDLocalWrench * worldToContact;

  ocs2::matrix_t contactJacobian = ocs2::matrix_t::Zero(6, stateConverterPtr_->getTangentDim());
  ocp_solver::getContactCandidateJacobian(pinocchioInterface, contactCandidate,
                                          pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED,
                                          contactJacobian);
  Eigen::Matrix<ocs2::scalar_t, 6, 6> dLocalWrenchDFrameRotation =
      Eigen::Matrix<ocs2::scalar_t, 6, 6>::Zero();
  dLocalWrenchDFrameRotation.block<3, 3>(0, 3) =
      contactPose.rotation().transpose() * ocs2::skewSymmetricMatrix(Eigen::Vector3d(worldWrench.head<3>()));
  dLocalWrenchDFrameRotation.block<3, 3>(3, 3) =
      contactPose.rotation().transpose() * ocs2::skewSymmetricMatrix(Eigen::Vector3d(worldWrench.tail<3>()));
  approx.dfdx.leftCols(stateConverterPtr_->getTangentDim()).noalias() =
      projection * dCopDLocalWrench * dLocalWrenchDFrameRotation * contactJacobian;

  if (contactCandidate.searchContactPoint) {
    approx.dfdx.block<3, 3>(
        0, stateConverterPtr_->getContactPointLocalPositionVariableStartIndex(contactIndex_)).noalias() =
        projection * contactCandidate.localPose.rotation().transpose();
  }
  return approx;
}

}  // namespace ocp_constraint_body_contact

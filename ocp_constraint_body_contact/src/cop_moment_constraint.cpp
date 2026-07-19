#include "ocp_constraint_body_contact/cop_moment_constraint.h"

#include <ocp_solver/common/scope_profiler.h>
#include <ocp_solver/solver/dynamics_helper_functions.h>
#include <ocp_solver/solver/ocp_pre_computation.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>

namespace ocp_constraint_body_contact {

CopMomentConstraint::CopMomentConstraint(
    size_t contactIndex,
    const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
    const AssumedSurfaceContactConstraint& surfaceConstraint)
    : StateInputConstraint(ocs2::ConstraintOrder::Linear),
      contactIndex_(contactIndex),
      stateConverterPtr_(stateConverter.clone()),
      surfaceConstraintPtr_(&surfaceConstraint) {}

CopMomentConstraint::CopMomentConstraint(const CopMomentConstraint& rhs)
    : StateInputConstraint(rhs),
      contactIndex_(rhs.contactIndex_),
      stateConverterPtr_(rhs.stateConverterPtr_->clone()),
      surfaceConstraintPtr_(rhs.surfaceConstraintPtr_) {}

bool CopMomentConstraint::isActive(ocs2::scalar_t time) const {
  return surfaceConstraintPtr_->isActive(time);
}

size_t CopMomentConstraint::getNumConstraints(ocs2::scalar_t /*time*/) const {
  return 3;
}

ocs2::vector_t CopMomentConstraint::getValue(ocs2::scalar_t time,
                                             const ocs2::vector_t& state,
                                             const ocs2::vector_t& input,
                                             const ocs2::PreComputation& preComp) const {
  OCP_SOLVER_PROFILE_SCOPE("CopMomentConstraint::getValue");
  const ocp_solver::OCPPreComputation& ocpPreComp =
      static_cast<const ocp_solver::OCPPreComputation&>(preComp);
  ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
  const pinocchio::SE3 contactPose =
      ocp_solver::getContactCandidatePlacement(pinocchioInterface,
                                               stateConverterPtr_->getContactCandidate(state, contactIndex_));
  const auto surfaceGeometry =
      surfaceConstraintPtr_->getSurfaceGeometry(time, state, pinocchioInterface);
  Eigen::Matrix<double, 6, 6> worldToContact = Eigen::Matrix<double, 6, 6>::Zero();
  worldToContact.block<3, 3>(0, 0) = contactPose.rotation().transpose();
  worldToContact.block<3, 3>(3, 3) = contactPose.rotation().transpose();
  const Eigen::Matrix<double, 6, 1> localWrench =
      worldToContact * stateConverterPtr_->getContactWrench(input, contactIndex_);
  const Eigen::Vector3d normal = surfaceGeometry.normalInContactFrame.normalized();
  const double normalForce = normal.dot(localWrench.head<3>());
  return localWrench.tail<3>() + normalForce * normal.cross(surfaceGeometry.geometricCenterInContactPlane);
}

ocs2::VectorFunctionLinearApproximation CopMomentConstraint::getLinearApproximation(
    ocs2::scalar_t time,
    const ocs2::vector_t& state,
    const ocs2::vector_t& input,
    const ocs2::PreComputation& preComp) const {
  OCP_SOLVER_PROFILE_SCOPE("CopMomentConstraint::getLinearApproximation");
  const ocp_solver::OCPPreComputation& ocpPreComp =
      static_cast<const ocp_solver::OCPPreComputation&>(preComp);
  ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
  const auto contactCandidate = stateConverterPtr_->getContactCandidate(state, contactIndex_);
  const pinocchio::SE3 contactPose =
      ocp_solver::getContactCandidatePlacement(pinocchioInterface, contactCandidate);
  const auto surfaceGeometry =
      surfaceConstraintPtr_->getSurfaceGeometry(time, state, pinocchioInterface);
  const Eigen::Vector3d normal = surfaceGeometry.normalInContactFrame.normalized();
  const Eigen::Vector3d centerLever = normal.cross(surfaceGeometry.geometricCenterInContactPlane);

  Eigen::Matrix<double, 3, 6> dResidualDLocalWrench = Eigen::Matrix<double, 3, 6>::Zero();
  dResidualDLocalWrench.leftCols<3>() = centerLever * normal.transpose();
  dResidualDLocalWrench.rightCols<3>().setIdentity();

  Eigen::Matrix<double, 6, 6> worldToContact = Eigen::Matrix<double, 6, 6>::Zero();
  worldToContact.block<3, 3>(0, 0) = contactPose.rotation().transpose();
  worldToContact.block<3, 3>(3, 3) = contactPose.rotation().transpose();
  const Eigen::Matrix<double, 6, 1> worldWrench =
      stateConverterPtr_->getContactWrench(input, contactIndex_);

  ocs2::VectorFunctionLinearApproximation approx;
  approx.f = getValue(time, state, input, preComp);
  approx.dfdx = ocs2::matrix_t::Zero(3, stateConverterPtr_->getStateVariableDim());
  approx.dfdu = ocs2::matrix_t::Zero(3, stateConverterPtr_->getInputDim());
  approx.dfdu.block<3, 6>(0, stateConverterPtr_->getContactWrenchStartIndices(contactIndex_)).noalias() =
      dResidualDLocalWrench * worldToContact;

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
      dResidualDLocalWrench * dLocalWrenchDFrameRotation * contactJacobian;
  return approx;
}

}  // namespace ocp_constraint_body_contact

#include "ocp_constraint_body_contact/assumed_surface_contact_constraint.h"
#include <ocp_solver/solver/ocp_pre_computation.h>
#include <ocp_solver/solver/dynamics_helper_functions.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/multibody/data.hpp>

namespace ocp_constraint_body_contact {

  AssumedSurfaceContactConstraint::AssumedSurfaceContactConstraint(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                                                                   size_t contactIndex,
                                                                   const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
                                                                   Config config)
    : StateInputConstraint(ocs2::ConstraintOrder::Linear),
      referenceManagerPtr_(&referenceManager),
      contactIndex_(contactIndex),
      stateConverterPtr_(&stateConverter) {
    coef_ = ocs2::matrix_t::Zero(n_constraints, 6);
    coef_( 0,2) =  1.0;
    coef_( 1,0) =  1.0; coef_( 1,2) = config.frictionCoef;
    coef_( 2,0) = -1.0; coef_( 2,2) = config.frictionCoef;
    coef_( 3,1) =  1.0; coef_( 3,2) = config.frictionCoef;
    coef_( 4,1) = -1.0; coef_( 4,2) = config.frictionCoef;
    coef_( 5,3) =  1.0; coef_( 5,2) = config.x;
    coef_( 6,3) = -1.0; coef_( 6,2) = config.x;
    coef_( 7,4) =  1.0; coef_( 7,2) = config.y;
    coef_( 8,4) = -1.0; coef_( 8,2) = config.y;
    coef_( 9,5) =  1.0; coef_( 9,2) = config.rotFrictionCoef;
    coef_(10,5) = -1.0; coef_(10,2) = config.rotFrictionCoef;
  }

  AssumedSurfaceContactConstraint::AssumedSurfaceContactConstraint(const AssumedSurfaceContactConstraint& rhs)
    : StateInputConstraint(rhs),
      referenceManagerPtr_(rhs.referenceManagerPtr_),
      contactIndex_(rhs.contactIndex_),
      stateConverterPtr_(rhs.stateConverterPtr_->clone()),
      coef_(rhs.coef_) {}

  bool AssumedSurfaceContactConstraint::isActive(ocs2::scalar_t time) const {
    return referenceManagerPtr_->isInContact(time, stateConverterPtr_->getContactCandidateIds()[contactIndex_]);
  }

  ocs2::vector_t AssumedSurfaceContactConstraint::getValue(ocs2::scalar_t time,
                                                           const ocs2::vector_t& state,
                                                           const ocs2::vector_t& input,
                                                           const ocs2::PreComputation& preComp) const {
    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
    Eigen::Matrix3d R_frame =
      ocp_solver::getContactCandidatePlacement(pinocchioInterface, stateConverterPtr_->getContactCandidate(contactIndex_)).rotation();
    Eigen::Matrix<ocs2::scalar_t, 6, 6> R = Eigen::Matrix<ocs2::scalar_t, 6, 6>::Zero();
    R.block(0,0,3,3) = R_frame.transpose();
    R.block(3,3,3,3) = R_frame.transpose();
    ocs2::vector_t constraint = coef_ * R * stateConverterPtr_->getContactWrench(input, contactIndex_);
    return constraint;
  }

  ocs2::VectorFunctionLinearApproximation AssumedSurfaceContactConstraint::getLinearApproximation(ocs2::scalar_t time,
                                                                                                  const ocs2::vector_t& state,
                                                                                                  const ocs2::vector_t& input,
                                                                                                  const ocs2::PreComputation& preComp) const {
    const ocp_solver::OCPPreComputation& ocpPreComp = static_cast<const ocp_solver::OCPPreComputation&>(preComp);
    ocs2::PinocchioInterface& pinocchioInterface = ocpPreComp.getPinocchioInterface();
    Eigen::Matrix3d R_frame =
      ocp_solver::getContactCandidatePlacement(pinocchioInterface, stateConverterPtr_->getContactCandidate(contactIndex_)).rotation();
    Eigen::Matrix<ocs2::scalar_t, 6, 6> R = Eigen::Matrix<ocs2::scalar_t, 6, 6>::Zero();
    R.block(0,0,3,3) = R_frame.transpose();
    R.block(3,3,3,3) = R_frame.transpose();
    const Eigen::Matrix<ocs2::scalar_t, 6, 1> wrench = stateConverterPtr_->getContactWrench(input, contactIndex_);

    ocs2::VectorFunctionLinearApproximation approx;
    approx.f = coef_ * R * wrench;
    approx.dfdx = ocs2::matrix_t::Zero(n_constraints, stateConverterPtr_->getStateVariableDim());
    approx.dfdu = ocs2::matrix_t::Zero(n_constraints, stateConverterPtr_->getInputDim());
    approx.dfdu.middleCols<6>(6 * contactIndex_) = coef_ * R;

    ocs2::matrix_t frameJacobian = ocs2::matrix_t::Zero(6, stateConverterPtr_->getTangentDim());
    ocp_solver::getContactCandidateJacobian(pinocchioInterface, stateConverterPtr_->getContactCandidate(contactIndex_),
                                            pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, frameJacobian);

    Eigen::Matrix<ocs2::scalar_t, 6, 6> localWrenchDerivativeWrtFrameRotation =
      Eigen::Matrix<ocs2::scalar_t, 6, 6>::Zero();
    localWrenchDerivativeWrtFrameRotation.block<3, 3>(0, 3) =
      R_frame.transpose() * ocs2::skewSymmetricMatrix(Eigen::Vector3d(wrench.head<3>()));
    localWrenchDerivativeWrtFrameRotation.block<3, 3>(3, 3) =
      R_frame.transpose() * ocs2::skewSymmetricMatrix(Eigen::Vector3d(wrench.tail<3>()));
    approx.dfdx.leftCols(stateConverterPtr_->getTangentDim()).noalias() =
      coef_ * localWrenchDerivativeWrtFrameRotation * frameJacobian;
    return approx;
  }

}

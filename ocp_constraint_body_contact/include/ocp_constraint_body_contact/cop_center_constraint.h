#pragma once

#include <memory>

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocp_solver/solver/state_converter.h>

#include "ocp_constraint_body_contact/assumed_surface_contact_constraint.h"

namespace ocp_constraint_body_contact {

class CopCenterConstraint final : public ocs2::StateInputConstraint {
 public:
  CopCenterConstraint(size_t contactIndex,
                      const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
                      const AssumedSurfaceContactConstraint& surfaceConstraint);

  CopCenterConstraint* clone() const override { return new CopCenterConstraint(*this); }

  bool isActive(ocs2::scalar_t time) const override;
  size_t getNumConstraints(ocs2::scalar_t time) const override;
  ocs2::vector_t getValue(ocs2::scalar_t time,
                          const ocs2::vector_t& state,
                          const ocs2::vector_t& input,
                          const ocs2::PreComputation& preComp) const override;
  ocs2::VectorFunctionLinearApproximation getLinearApproximation(
      ocs2::scalar_t time,
      const ocs2::vector_t& state,
      const ocs2::vector_t& input,
      const ocs2::PreComputation& preComp) const override;

 private:
  CopCenterConstraint(const CopCenterConstraint& rhs);

  size_t contactIndex_ = 0;
  std::unique_ptr<ocp_solver::StateConverter<ocs2::scalar_t>> stateConverterPtr_;
  const AssumedSurfaceContactConstraint* surfaceConstraintPtr_ = nullptr;
};

}  // namespace ocp_constraint_body_contact

#pragma once

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocp_solver/solver/state_converter.h>
#include <ocp_solver/solver/switched_model_reference_manager.h>

namespace ocp_constraint_body_contact {

  class AssumedSurfaceContactConstraint final : public ocs2::StateInputConstraint {
  public:
    struct Config {
      Config(){};
      ocs2::scalar_t frictionCoef = 0.2;
      ocs2::scalar_t x = 0.05;
      ocs2::scalar_t y = 0.05;
      ocs2::scalar_t rotFrictionCoef = 0.005;
    };
    AssumedSurfaceContactConstraint(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                                    size_t contactIndex,
                                    const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
                                    Config config=Config());

    ~AssumedSurfaceContactConstraint() override = default;
    AssumedSurfaceContactConstraint* clone() const override { return new AssumedSurfaceContactConstraint(*this); }

    bool isActive(ocs2::scalar_t time) const override;
    size_t getNumConstraints(ocs2::scalar_t time) const override { return n_constraints; }
    ocs2::vector_t getValue(ocs2::scalar_t time,
                            const ocs2::vector_t& state,
                            const ocs2::vector_t& input,
                            const ocs2::PreComputation& preComp) const override;
    ocs2::VectorFunctionLinearApproximation getLinearApproximation(ocs2::scalar_t time,
                                                                   const ocs2::vector_t& state,
                                                                   const ocs2::vector_t& input,
                                                                   const ocs2::PreComputation& preComp) const override;

  private:
    AssumedSurfaceContactConstraint(const AssumedSurfaceContactConstraint& rhs);
    const ocp_solver::StateConverter<ocs2::scalar_t>* stateConverterPtr_;
    const ocp_solver::SwitchedModelReferenceManager* referenceManagerPtr_;
    const size_t contactIndex_;
    static const int n_constraints = 11;
    ocs2::matrix_t coef_;
  };

}

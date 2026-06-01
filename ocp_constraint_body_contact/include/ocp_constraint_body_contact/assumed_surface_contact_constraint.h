#pragma once

#include <ocs2_core/constraint/StateInputConstraint.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocp_solver/solver/state_converter.h>
#include <ocp_solver/solver/switched_model_reference_manager.h>
#include <pinocchio/spatial/se3.hpp>

#include <string>
#include <vector>

namespace ocp_constraint_body_contact {

  class AssumedSurfaceContactConstraint final : public ocs2::StateInputConstraint {
  public:
    struct SurfaceGeometry {
      Eigen::Vector3d normalInContactFrame = Eigen::Vector3d::UnitZ();
      Eigen::Vector3d geometricCenterInContactPlane = Eigen::Vector3d::Zero();
      Eigen::Matrix3d ellipseMetric = Eigen::Matrix3d::Identity();
    };

    struct Config {
      Config(){};
      Eigen::Vector3d contactNormalInContactFrame = Eigen::Vector3d::UnitZ();
      Eigen::Vector3d surfaceNormalInContactFrame = -Eigen::Vector3d::UnitZ();
      ocs2::scalar_t normalWeightScale = 50.0;
      ocs2::scalar_t covarianceRegularization = 1e-6;
      ocs2::scalar_t normalForceRegularization = 1.0;
      ocs2::scalar_t minNormalForce = 1.0;
      ocs2::scalar_t frictionCoef = 0.5;
      ocs2::scalar_t ellipseScale = 1.0;
      ocs2::scalar_t ellipseSafetyMargin = 0.0;
    };
    AssumedSurfaceContactConstraint(const ocp_solver::SwitchedModelReferenceManager& referenceManager,
                                    size_t contactIndex,
                                    const ocp_solver::StateConverter<ocs2::scalar_t>& stateConverter,
                                    const std::string& meshFile,
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
    const Eigen::Vector3d& getNormalInContactFrame() const { return normalInContactFrame_; }
    const Eigen::Vector3d& getGeometricCenterInContactPlane() const { return geometricCenterInContactPlane_; }
    Eigen::Vector3d getGeometricCenterInContactPlane(const ocs2::vector_t& state) const;
    const Eigen::Matrix3d& getEllipseMetricInContactFrame() const { return ellipseMetric_; }
    SurfaceGeometry getSurfaceGeometry(ocs2::scalar_t time,
                                       const ocs2::vector_t& state,
                                       ocs2::PinocchioInterface& pinocchioInterface) const;
    Eigen::Vector3d computePressureCenterInContactFrame(
        const Eigen::Matrix3d& contactRotation,
        const Eigen::Matrix<ocs2::scalar_t, 6, 1>& localWorldAlignedWrench) const;
    Eigen::Vector3d computePressureCenterInContactFrame(
        const Eigen::Matrix3d& contactRotation,
        const Eigen::Matrix<ocs2::scalar_t, 6, 1>& localWorldAlignedWrench,
        const Eigen::Vector3d& normalInContactFrame) const;

  private:
    AssumedSurfaceContactConstraint(const AssumedSurfaceContactConstraint& rhs);
    void computeWeightedGeometry();
    SurfaceGeometry computeWeightedGeometry(const Eigen::Vector3d& normalInContactFrame,
                                            const pinocchio::SE3& contactPoseInLocalFrame) const;
    SurfaceGeometry computeWeightedGeometry(const Eigen::Vector3d& normalInContactFrame,
                                            const Eigen::Vector3d& surfaceWeightNormalInContactFrame,
                                            const pinocchio::SE3& contactPoseInLocalFrame) const;
    pinocchio::SE3 getTargetPose(ocs2::scalar_t time) const;
    Eigen::Vector3d computeMeshNormalInLocalFrame(const Eigen::Vector3d& contactPointInLocalFrame) const;
    Eigen::Vector3d computeMeshNormalInContactFrame(const pinocchio::SE3& contactPoseInLocalFrame) const;

    const ocp_solver::StateConverter<ocs2::scalar_t>* stateConverterPtr_;
    const ocp_solver::SwitchedModelReferenceManager* referenceManagerPtr_;
    const size_t contactIndex_;
    static const int n_constraints = 6;
    Config config_;
    pinocchio::SE3 defaultContactPoseInParent_ = pinocchio::SE3::Identity();
    pinocchio::SE3 defaultContactPoseInLocalFrame_ = pinocchio::SE3::Identity();
    std::vector<Eigen::Vector3d> vertices_;
    std::vector<Eigen::Vector3d> normals_;
    Eigen::Vector3d normalInContactFrame_ = Eigen::Vector3d::UnitZ();
    Eigen::Vector3d surfaceNormalInContactFrame_ = -Eigen::Vector3d::UnitZ();
    Eigen::Vector3d geometricCenter_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d geometricCenterInContactPlane_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covariance_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d ellipseMetric_ = Eigen::Matrix3d::Identity();
  };

}

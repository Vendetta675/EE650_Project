#include <string>
#include <vector>
#include <mutex>
#include <random>
#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/components/Joint.hh>
#include <gz/sim/components/JointForceCmd.hh>
#include <gz/sim/components/JointVelocityCmd.hh>
#include <gz/plugin/Register.hh>
#include <gz/transport/Node.hh>
#include <gz/msgs/boolean.pb.h>
#include <gz/common/Console.hh>

namespace motor_failure
{
  class MotorFailureSystem :
    public gz::sim::System,
    public gz::sim::ISystemConfigure,
    public gz::sim::ISystemPreUpdate
  {
    private: std::vector<gz::sim::Entity> joint_entities_;
    private: std::vector<std::string> joint_names_;
    private: gz::sim::Entity active_failed_joint_{gz::sim::kNullEntity};
    
    private: bool motor_failed_{false};
    
    private: gz::transport::Node gz_node_;
    private: std::mutex mutex_;

    public: void Configure(const gz::sim::Entity &_entity,
                           const std::shared_ptr<const sdf::Element> &_sdf,
                           gz::sim::EntityComponentManager &_ecm,
                           gz::sim::EventManager &/*_eventMgr*/) override
    {
      gz::sim::Model model(_entity);
      if (!model.Valid(_ecm)) {
        gzerr << "MotorFailureSystem must be attached to a model.\n";
        return;
      }

      if (!_sdf->HasElement("joint_name")) {
        gzerr << "MotorFailureSystem missing <joint_name>.\n";
        return;
      }

      // Read ALL <joint_name> elements provided in the SDF
      sdf::ElementPtr mutable_sdf = _sdf->Clone();
      auto joint_elem = mutable_sdf->GetElement("joint_name");
      while (joint_elem) {
        std::string j_name = joint_elem->Get<std::string>();
        this->joint_names_.push_back(j_name);
        
        gz::sim::Entity j_entity = gz::sim::kNullEntity;
        
        // Parse "::" to handle nested models
        size_t delim = j_name.find("::");
        if (delim != std::string::npos) {
          std::string nested_model_name = j_name.substr(0, delim);
          std::string inner_joint_name = j_name.substr(delim + 2);
          
          gz::sim::Entity nested_model_ent = model.ModelByName(_ecm, nested_model_name);
          if (nested_model_ent != gz::sim::kNullEntity) {
            gz::sim::Model nested_model(nested_model_ent);
            j_entity = nested_model.JointByName(_ecm, inner_joint_name);
          }
        } else {
          j_entity = model.JointByName(_ecm, j_name);
        }

        if (j_entity != gz::sim::kNullEntity) {
          this->joint_entities_.push_back(j_entity);
          gzmsg << "Registered joint for potential failure: " << j_name << "\n";
        } else {
          gzerr << "Could not find joint: " << j_name << " in ECS.\n";
        }

        joint_elem = joint_elem->GetNextElement("joint_name");
      }

      // Create a single global topic for the entire model
      std::string topic = "/model/" + model.Name(_ecm) + "/trigger_random_failure";
      this->gz_node_.Subscribe(topic, &MotorFailureSystem::OnTrigger, this);
      
      gzmsg << "Random MotorFailureSystem loaded. Listening on: " << topic << "\n";
    }

    public: void OnTrigger(const gz::msgs::Boolean &_msg)
    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      this->motor_failed_ = _msg.data();
      
      if (this->motor_failed_ && !this->joint_entities_.empty()) {
        // Pick a random index based on the number of registered joints
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, this->joint_entities_.size() - 1);
        
        int random_idx = distrib(gen);
        this->active_failed_joint_ = this->joint_entities_[random_idx];
        
        gzwarn << "RANDOM FAILURE TRIGGERED! Motor killed: " << this->joint_names_[random_idx] << "\n";
      } else {
        this->active_failed_joint_ = gz::sim::kNullEntity;
        gzmsg << "Motors RECOVERED.\n";
      }
    }

    public: void PreUpdate(const gz::sim::UpdateInfo &_info,
                           gz::sim::EntityComponentManager &_ecm) override
    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      // Only process if failure is active and a valid joint is selected
      if (!this->motor_failed_ || _info.paused || this->active_failed_joint_ == gz::sim::kNullEntity) return;

      auto force_cmd = _ecm.Component<gz::sim::components::JointForceCmd>(this->active_failed_joint_);
      if (force_cmd && !force_cmd->Data().empty()) {
        force_cmd->Data()[0] = 0.0; 
      }

      auto vel_cmd = _ecm.Component<gz::sim::components::JointVelocityCmd>(this->active_failed_joint_);
      if (vel_cmd && !vel_cmd->Data().empty()) {
        vel_cmd->Data()[0] = 0.0;
      }
    }
  };
}

GZ_ADD_PLUGIN(motor_failure::MotorFailureSystem,
              gz::sim::System,
              motor_failure::MotorFailureSystem::ISystemConfigure,
              motor_failure::MotorFailureSystem::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(motor_failure::MotorFailureSystem, "motor_failure_system")

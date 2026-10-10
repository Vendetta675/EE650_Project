// Progressive motor failure for Gazebo Sim (Harmonic) + ardupilot_gazebo.
//
// Instead of cutting the motor command to zero, a triggered failure is a Loss of
// Effectiveness (LOE): the failing joint's command is multiplied by a factor k(t) that
// ramps from 1.0 down to <loe_final> over <ramp_duration> seconds of sim time and then
// holds. On top of that, a sinusoidal "stutter" (desynchronised ESC / stripped gear) is
// injected into the same command. Its amplitude grows with the degradation, so the
// micro-vibrations build up before the motor is fully degraded. An optional hard cut
// (<cut_after>, seconds after the ramp ends) reproduces the old catastrophic failure.
//
// With ardupilot_gazebo's <useForce>1</useForce> + <type>VELOCITY</type> rotor control,
// ArduPilotPlugin writes the velocity-PID output to JointForceCmd (motor torque); LiftDrag
// turns the resulting rotor speed into thrust. This plugin must run after ArduPilotPlugin
// (declare it after it in the model SDF) so it modifies the command of the current step.
//
// Ground truth for training labels: every start / cut / recover event is appended to
// <event_log> (CSV, wall-clock + sim time, motor index, parameters), and the live state
// [active, motor_index, k, elapsed_s] is published on /model/<model>/motor_fault_state.
//
// SDF parameters (all optional except <joint_name>):
//   <joint_name>          one per candidate rotor joint (index = order of declaration)
//   <failure_motor>       fixed motor index, -1 = random (default -1)
//   <loe_final>           final effectiveness factor (default 0.6)
//   <ramp_duration>       seconds from 1.0 to loe_final (default 5.0)
//   <stutter_amplitude>   peak stutter, fraction of the command at full degradation (0.15)
//   <stutter_frequency>   Hz (default 35; keep well below half the physics rate)
//   <stutter_phase_jitter> std of the per-step phase noise, rad (default 0.05)
//   <cut_after>           seconds after the ramp until the command is cut to 0, -1 = never
//   <event_log>           CSV path (default /root/ros2_ws/fault_events.csv)
// Trigger: gz topic /model/<model>/trigger_random_failure (gz.msgs.Boolean): true starts a
// failure (ignored while one is active), false recovers.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include <gz/common/Console.hh>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/double_v.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/components/Joint.hh>
#include <gz/sim/components/JointForceCmd.hh>
#include <gz/sim/components/JointVelocityCmd.hh>
#include <gz/transport/Node.hh>

namespace motor_failure
{
  class MotorFailureSystem :
    public gz::sim::System,
    public gz::sim::ISystemConfigure,
    public gz::sim::ISystemPreUpdate
  {
    private: std::vector<gz::sim::Entity> joint_entities_;
    private: std::vector<std::string> joint_names_;
    private: std::string model_name_;

    // parameters
    private: int fixed_motor_{-1};
    private: double loe_final_{0.6};
    private: double ramp_s_{5.0};
    private: double stutter_amp_{0.15};
    private: double stutter_hz_{35.0};
    private: double phase_jitter_{0.05};
    private: double cut_after_s_{-1.0};
    private: std::string event_log_{"/root/ros2_ws/fault_events.csv"};

    // state (guarded by mutex_)
    private: bool requested_{false};      // last trigger value
    private: bool active_{false};         // failure in progress
    private: bool start_pending_{false};  // trigger arrived, start time not yet stamped
    private: bool cut_logged_{false};
    private: int motor_idx_{-1};
    private: double start_sim_s_{0.0};
    private: double phase_{0.0};
    private: double last_pub_sim_s_{-1.0};

    private: std::mt19937 rng_{std::random_device{}()};
    private: std::normal_distribution<double> phase_noise_{0.0, 1.0};
    private: gz::transport::Node gz_node_;
    private: gz::transport::Node::Publisher state_pub_;
    private: std::mutex mutex_;

    template <typename T>
    static T Param(const std::shared_ptr<const sdf::Element> &_sdf, const std::string &_name, T _default)
    {
      return _sdf->HasElement(_name) ? _sdf->Get<T>(_name) : _default;
    }

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
      this->model_name_ = model.Name(_ecm);

      // Read ALL <joint_name> elements provided in the SDF
      sdf::ElementPtr mutable_sdf = _sdf->Clone();
      auto joint_elem = mutable_sdf->GetElement("joint_name");
      while (joint_elem) {
        std::string j_name = joint_elem->Get<std::string>();
        gz::sim::Entity j_entity = gz::sim::kNullEntity;

        // Parse "::" to handle nested models
        size_t delim = j_name.find("::");
        if (delim != std::string::npos) {
          gz::sim::Entity nested_model_ent = model.ModelByName(_ecm, j_name.substr(0, delim));
          if (nested_model_ent != gz::sim::kNullEntity) {
            j_entity = gz::sim::Model(nested_model_ent).JointByName(_ecm, j_name.substr(delim + 2));
          }
        } else {
          j_entity = model.JointByName(_ecm, j_name);
        }

        if (j_entity != gz::sim::kNullEntity) {
          this->joint_entities_.push_back(j_entity);
          this->joint_names_.push_back(j_name);
          gzmsg << "Registered joint for potential failure: " << j_name << "\n";
        } else {
          gzerr << "Could not find joint: " << j_name << " in ECS.\n";
        }
        joint_elem = joint_elem->GetNextElement("joint_name");
      }

      this->fixed_motor_ = Param<int>(_sdf, "failure_motor", -1);
      this->loe_final_ = Param<double>(_sdf, "loe_final", 0.6);
      this->ramp_s_ = std::max(1e-3, Param<double>(_sdf, "ramp_duration", 5.0));
      this->stutter_amp_ = Param<double>(_sdf, "stutter_amplitude", 0.15);
      this->stutter_hz_ = Param<double>(_sdf, "stutter_frequency", 35.0);
      this->phase_jitter_ = Param<double>(_sdf, "stutter_phase_jitter", 0.05);
      this->cut_after_s_ = Param<double>(_sdf, "cut_after", -1.0);
      this->event_log_ = Param<std::string>(_sdf, "event_log", this->event_log_);

      const std::string base = "/model/" + this->model_name_;
      this->gz_node_.Subscribe(base + "/trigger_random_failure", &MotorFailureSystem::OnTrigger, this);
      this->state_pub_ = this->gz_node_.Advertise<gz::msgs::Double_V>(base + "/motor_fault_state");

      gzmsg << "Progressive MotorFailureSystem loaded. Trigger: " << base << "/trigger_random_failure"
            << " | LOE 1.0 -> " << this->loe_final_ << " over " << this->ramp_s_ << " s"
            << " | stutter " << this->stutter_amp_ << " @ " << this->stutter_hz_ << " Hz"
            << " | cut_after " << this->cut_after_s_ << " | event log " << this->event_log_ << "\n";
    }

    public: void OnTrigger(const gz::msgs::Boolean &_msg)
    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      this->requested_ = _msg.data();
      if (this->requested_ && !this->active_ && !this->joint_entities_.empty()) {
        this->start_pending_ = true;  // stamped with sim time in PreUpdate
      }
    }

    private: void LogEvent(const std::string &_event, double _sim_s, double _k)
    {
      const bool is_new = !std::ifstream(this->event_log_).good();
      std::ofstream out(this->event_log_, std::ios::app);
      if (!out) {
        gzerr << "Cannot write fault event log " << this->event_log_ << "\n";
        return;
      }
      if (is_new) {
        out << "wall_time,sim_time,event,motor_index,joint_name,effectiveness,loe_final,ramp_duration,"
               "stutter_amplitude,stutter_frequency,cut_after\n";
      }
      const double wall = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
      out << std::fixed << std::setprecision(6) << wall << "," << _sim_s << "," << _event << ","
          << this->motor_idx_ << ","
          << (this->motor_idx_ >= 0 ? this->joint_names_[this->motor_idx_] : "") << "," << _k << ","
          << this->loe_final_ << "," << this->ramp_s_ << "," << this->stutter_amp_ << ","
          << this->stutter_hz_ << "," << this->cut_after_s_ << "\n";
    }

    private: double Effectiveness(double _elapsed) const
    {
      const double frac = std::min(1.0, std::max(0.0, _elapsed / this->ramp_s_));
      return 1.0 - (1.0 - this->loe_final_) * frac;
    }

    public: void PreUpdate(const gz::sim::UpdateInfo &_info,
                           gz::sim::EntityComponentManager &_ecm) override
    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      if (_info.paused) return;
      const double sim_s = std::chrono::duration<double>(_info.simTime).count();
      const double dt = std::chrono::duration<double>(_info.dt).count();

      if (this->start_pending_) {
        this->start_pending_ = false;
        if (this->fixed_motor_ >= 0 && this->fixed_motor_ < static_cast<int>(this->joint_entities_.size())) {
          this->motor_idx_ = this->fixed_motor_;
        } else {
          std::uniform_int_distribution<int> pick(0, static_cast<int>(this->joint_entities_.size()) - 1);
          this->motor_idx_ = pick(this->rng_);
        }
        this->active_ = true;
        this->cut_logged_ = false;
        this->start_sim_s_ = sim_s;
        this->phase_ = 0.0;
        gzwarn << "PROGRESSIVE FAILURE STARTED on " << this->joint_names_[this->motor_idx_]
               << " (motor index " << this->motor_idx_ << ")\n";
        this->LogEvent("start", sim_s, 1.0);
      }
      if (this->active_ && !this->requested_) {
        this->active_ = false;
        gzmsg << "Motors RECOVERED.\n";
        this->LogEvent("recover", sim_s, 1.0);
        this->motor_idx_ = -1;
      }

      double k = 1.0;
      double elapsed = 0.0;
      if (this->active_) {
        elapsed = sim_s - this->start_sim_s_;
        k = this->Effectiveness(elapsed);
        const bool cut = this->cut_after_s_ >= 0.0 && elapsed >= this->ramp_s_ + this->cut_after_s_;
        if (cut && !this->cut_logged_) {
          this->cut_logged_ = true;
          gzwarn << "Motor " << this->motor_idx_ << " CUT after progressive degradation.\n";
          this->LogEvent("cut", sim_s, 0.0);
        }

        // stutter: amplitude grows with degradation (0 at start, stutter_amp_ at loe_final)
        const double degradation = (1.0 - k) / std::max(1e-9, 1.0 - this->loe_final_);
        this->phase_ += 2.0 * M_PI * this->stutter_hz_ * dt + this->phase_jitter_ * this->phase_noise_(this->rng_);
        const double gain = cut ? 0.0 : k + this->stutter_amp_ * degradation * std::sin(this->phase_);
        if (cut) k = 0.0;

        const gz::sim::Entity joint = this->joint_entities_[this->motor_idx_];
        auto force_cmd = _ecm.Component<gz::sim::components::JointForceCmd>(joint);
        if (force_cmd && !force_cmd->Data().empty()) {
          force_cmd->Data()[0] *= gain;
        }
        auto vel_cmd = _ecm.Component<gz::sim::components::JointVelocityCmd>(joint);
        if (vel_cmd && !vel_cmd->Data().empty()) {
          vel_cmd->Data()[0] *= cut ? 0.0 : k;  // no stutter on a velocity target
        }
      }

      // live state at ~50 Hz of sim time: [active, motor_index, effectiveness, elapsed_s]
      if (this->last_pub_sim_s_ < 0.0 || sim_s - this->last_pub_sim_s_ >= 0.02) {
        this->last_pub_sim_s_ = sim_s;
        gz::msgs::Double_V msg;
        msg.add_data(this->active_ ? 1.0 : 0.0);
        msg.add_data(static_cast<double>(this->motor_idx_));
        msg.add_data(k);
        msg.add_data(elapsed);
        this->state_pub_.Publish(msg);
      }
    }
  };
}

GZ_ADD_PLUGIN(motor_failure::MotorFailureSystem,
              gz::sim::System,
              motor_failure::MotorFailureSystem::ISystemConfigure,
              motor_failure::MotorFailureSystem::ISystemPreUpdate)
GZ_ADD_PLUGIN_ALIAS(motor_failure::MotorFailureSystem, "motor_failure_system")

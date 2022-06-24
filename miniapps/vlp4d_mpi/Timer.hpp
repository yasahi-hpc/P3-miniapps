#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include <chrono>
#include <vector>
#include <iostream>

struct Timer {
private:
  std::string label_;
  double accumulated_time_;
  int calls_;
  std::chrono::high_resolution_clock::time_point begin_, end_;

public:
  Timer() : accumulated_time_(0.0), calls_(0), label_(""){};
  Timer(const std::string label) : accumulated_time_(0.0), calls_(0), label_(label){};
  virtual ~Timer(){};

  void begin() {
    begin_ = std::chrono::high_resolution_clock::now();
  }

  void end() {
    end_ = std::chrono::high_resolution_clock::now();
    accumulated_time_ += std::chrono::duration_cast<std::chrono::duration<double> >(end_ - begin_).count();
    calls_++;
  }

  double seconds(){return accumulated_time_;};
  double milliseconds(){return accumulated_time_*1.e3;};
  int calls(){return calls_;};
  std::string label(){return label_;};
  void reset(){accumulated_time_ = 0.; calls_ = 0;};
};

enum TimerEnum : int {Total,
                      MainLoop,
                      pack,
                      comm,
                      unpack,
                      Advec2D,
                      Advec4D,
                      Field,
                      AllReduce,
                      Fourier,
                      Diag,
                      Splinecoeff_xy,
                      Splinecoeff_vxvy,
                      Nb_timers};

static void defineTimers(std::vector<Timer*> &timers) {
  // Set timers
  timers.resize(Nb_timers);
  timers[Total]                       = new Timer("total");
  timers[MainLoop]                    = new Timer("MainLoop");
  timers[TimerEnum::pack]             = new Timer("pack");
  timers[TimerEnum::comm]             = new Timer("comm");
  timers[TimerEnum::unpack]           = new Timer("unpack");
  timers[TimerEnum::Advec2D]          = new Timer("advec2D");
  timers[TimerEnum::Advec4D]          = new Timer("advec4D");
  timers[TimerEnum::Field]            = new Timer("field");
  timers[TimerEnum::AllReduce]        = new Timer("all_reduce");
  timers[TimerEnum::Fourier]          = new Timer("Fourier");
  timers[TimerEnum::Diag]             = new Timer("diag");
  timers[TimerEnum::Splinecoeff_xy]   = new Timer("splinecoeff_xy");
  timers[TimerEnum::Splinecoeff_vxvy] = new Timer("splinecoeff_vxvy");
}

static void printTimers(std::vector<Timer*> &timers) {
  // Print timer information
  for(auto it = timers.begin(); it != timers.end(); ++it) {
    std::cout << (*it)->label() << " " << (*it)->seconds() << " [s], " << (*it)->calls() << " calls" << std::endl;
  }
}

static void resetTimers(std::vector<Timer*> &timers) {
  for(auto it = timers.begin(); it != timers.end(); ++it) {
    (*it)->reset();
  }
};
        
static void freeTimers(std::vector<Timer*> &timers) {
  for(auto it = timers.begin(); it != timers.end(); ++it) {
    delete *it;
  }
};

#endif

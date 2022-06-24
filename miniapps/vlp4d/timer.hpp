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
                      Advec1D_x,
                      Advec1D_y,
                      Advec1D_vx,
                      Advec1D_vy,
                      Field,
                      Fourier,
                      Diag,
                      Nb_timers};

static void defineTimers(std::vector<Timer*> &timers) {
  // Set timers
  timers.resize(Nb_timers);
  timers[Total]                       = new Timer("total");
  timers[MainLoop]                    = new Timer("MainLoop");
  timers[TimerEnum::Advec1D_x]        = new Timer("advec1D_x");
  timers[TimerEnum::Advec1D_y]        = new Timer("advec1D_y");
  timers[TimerEnum::Advec1D_vx]       = new Timer("advec1D_vx");
  timers[TimerEnum::Advec1D_vy]       = new Timer("advec1D_vy");
  timers[TimerEnum::Field]            = new Timer("field");
  timers[TimerEnum::Fourier]          = new Timer("Fourier");
  timers[TimerEnum::Diag]             = new Timer("diag");
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

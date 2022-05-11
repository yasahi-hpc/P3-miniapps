#ifndef __TIMER_HPP__
#define __TIMER_HPP__

#include <chrono>

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
                      Heat,
                      HaloPack,
                      HaloUnpack,
                      HaloSwap,
                      Nb_timers};

static void defineTimers(std::vector<Timer*> &timers) {
  // Set timers
  timers.resize(Nb_timers);
  timers[Total]                 = new Timer("total");
  timers[MainLoop]              = new Timer("MainLoop");
  timers[TimerEnum::Heat]       = new Timer("Heat");
  timers[TimerEnum::HaloPack]   = new Timer("HaloPack");
  timers[TimerEnum::HaloUnpack] = new Timer("HaloUnpack");
  timers[TimerEnum::HaloSwap]   = new Timer("HaloSwap");
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

template < class FunctorType >
void exec_with_timer(const FunctorType &f, Timer *timer) {
  timer->begin();
  f();
  timer->end();
}

#endif

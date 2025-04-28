#ifndef __YAKL_SW_TIMER_H__
#define __YAKL_SW_TIMER_H__

#include <cstddef>
#include <vector>
#include <cstdio>
#include <string>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>

namespace yakl {
class swTimer {
  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::duration<double> Duration;
  typedef std::chrono::time_point<Clock> TimePoint;

public:
  struct Timer {
      std::string         label;
      size_t              label_hash;
      size_t              hits;
      Duration            accumulated_duration;
      Duration            max_duration;
      Duration            min_duration;
      TimePoint           previous_time_point;
      std::vector<size_t> child_hashes;
      int                 parent_index;
      bool                multiple_parents;
  };

  struct ActiveStackEntry {
    size_t label_hash;
    int    timer_index;
  };

  // suppose we only run Timer on MPE, there is no need to maintance thread data

  std::vector<ActiveStackEntry> active_stack; 
  std::vector<Timer> timers;

  void start(std::string label) {
    if (label.empty()) die("ERROR: calling start() with empty label");
    auto label_hash = hasher(label);
    auto timer_index = get_or_create_timer_index(label, label_hash);
    auto &timer = timers[timer_index];
    if (!active_stack.empty()) {
      int parent_timer_index = active_stack.back().timer_index;
      auto &parent_timer = timers[parent_timer_index];
      bool i_am_already_child = false;
      for (auto &child_hash : parent_timer.child_hashes) {
        if (child_hash == label_hash) {i_am_already_child = true; break;}
      }
      if ( ! i_am_already_child ) parent_timer.child_hashes.push_back(label_hash);
      if (timer.parent_index == parent_index_just_created) timer.parent_index = parent_timer_index;
      if (timer.parent_index != parent_timer_index) timer.multiple_parents = true;
    } else {
      if (timer.parent_index == parent_index_just_created) timer.parent_index = parent_index_main;
      if (timer.parent_index != parent_index_main) timer.multiple_parents = true;
    }
    active_stack.push_back({label_hash, timer_index});
    timer.hits ++;
    timer.previous_time_point = Clock::now();
  }

  void stop(std::string label) {
    auto now = Clock::now();
    if (label.empty()) die("ERROR: calling stop() with an empty label");
    auto label_hash = hasher( label );
    if ( label_hash != active_stack.back().label_hash ) die("ERROR: timers must be perfectly nested");
    auto &timer = timers[active_stack.back().timer_index];
    Duration duration = now - timer.previous_time_point;
    timer.max_duration = max(timer.max_duration, duration);
    timer.min_duration = min(timer.min_duration, duration);
    timer.accumulated_duration += duration;
    active_stack.pop_back();
  }

  void die(std::string msg) {std::cerr << msg << std::endl; throw std::runtime_error(msg); };

  void print_timer_and_children( int timer_index , std::vector<bool> &printed  ,
                                                                      int &level ) {
    auto &timer = timers[timer_index];
    if (! printed[timer_index]) {
      std::string label = timer.label;
      if (timer.multiple_parents) label = std::string("~") + label;
      for (int i=0; i < level; i++) { label = std::string("  ")+label; }
      label.resize( std::min(label_print_length-2,(int)label.size()) );
      std::cout << std::setw(label_print_length) << std::left << label
                << std::setw(12) << std::left << timer.hits
                << std::setw(15) << std::left << std::scientific << timer.accumulated_duration.count()
                << std::setw(15) << std::left << std::scientific << timer.min_duration.count()
                << std::setw(15) << std::left << std::scientific << timer.max_duration.count() << "\n";
      printed[timer_index] = true;
      for (int ichild = 0; ichild < timer.child_hashes.size(); ichild++) {
        int child_timer_index = get_or_create_timer_index( "" , timer.child_hashes[ichild] );
        int level_loc = level + 1;
        print_timer_and_children( child_timer_index , printed , level_loc );
      }
    }
  }

  void print_all_threads() {
    std::cout << "******* Timers for MPE *******" << "\n";
      if (! active_stack.empty())
        std::cout << "WARNING: printing timers while some are still active. Results will be inaccurate\n";
      std::vector<bool> printed ( timers.size() , false );
      std::cout << "________________________________________________________________________________________________________\n";
      std::cout << std::setw(label_print_length) << std::left << "Timer label"
                << std::setw(12) << std::left << "# calls"
                << std::setw(15) << std::left << "Total time"
                << std::setw(15) << std::left << "Min time"
                << std::setw(15) << std::left << "Max time" << "\n";
      std::cout << "________________________________________________________________________________________________________\n";
      // TODO: Find a way to detect timers with multiple parents
      for (int itimer = 0; itimer < timers.size(); itimer++) {
        int level = 0;
        if (! printed[itimer]) print_timer_and_children( itimer , printed , level );
      }
      std::cout << "________________________________________________________________________________________________________\n"
                << "The ~ character beginning a timer label indicates it has multiple parent timers.\n"
                << "Thus, those timers will likely not accumulate like you expect them to.\n";
      std::cout << std::endl << std::endl;
  }

private:
  std::hash<std::string> hasher;

  int static constexpr parent_index_just_created = -100;
  int static constexpr parent_index_main         = -1;
  int static constexpr label_print_length        = 50;

  long tic() {
    unsigned long t1;
    asm volatile("rcsr %0,4":"=r"(t1));
    return t1;
  }

  int get_or_create_timer_index(std::string label, size_t label_hash) {
    for (int i = 0; i < timers.size(); i ++) {
      if ( label_hash == timers[i].label_hash ) return i;
    }

    timers.push_back({label, label_hash, 0, 
                Duration::zero(), Duration::zero(), Duration::max(), TimePoint::min(), 
                std::vector<size_t>(), parent_index_just_created, false});
    
    return timers.size() - 1;
  }
};
}

#endif //__YAKL_SW_TIMER_H__
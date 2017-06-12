/*
 * high_res_timer.h
 *
 *      Author: Alex Teichman
 *
 * A class for measuring how long some piece of code takes to run.
 *
 */

#ifndef HIGH_RES_TIMER_H
#define HIGH_RES_TIMER_H

#if defined(_WIN32)
#include <Windows.h>
#else
#include <time.h>
#endif
#include <string>
#include <sstream>
#include <cstddef>
#include <iostream>
#include <cstdio>

// Easy way to print the time used for various functions.
// For more advanced timing analysis, we recommend use of a profiler.
//! CLOCK_MONOTONIC_RAW will not be adjusted by NTP.
//! See man clock_gettime.
class HighResTimer {
public:
  std::string description_;
#if defined(_WIN32)
  HighResTimer(const std::string& description = "HighResTimer");
#else
  HighResTimer(const std::string& description = "HighResTimer",
               const clockid_t& clock = CLOCK_PROCESS_CPUTIME_ID);
#endif
  void start();
  void stop();
  void reset(const std::string& description);
  void reset();
  double getMicroseconds() const;
  double getMilliseconds() const;
  double getSeconds() const;
  double getMinutes() const;
  double getHours() const;

  std::string report() const;
  std::string reportMicroseconds() const;
  std::string reportMilliseconds() const;
  std::string reportSeconds() const;
  std::string reportMinutes() const;
  std::string reportHours() const;
  
  void print() const {std::string msString = report(); printf("[TIMER] %s\n", msString.c_str());}
  void printSeconds() const {std::string msString = reportSeconds(); printf("[TIMER] %s\n", msString.c_str());}
  void printMilliseconds() const {std::string msString = reportMilliseconds(); printf("[TIMER] %s\n", msString.c_str());}
  void printMicroseconds() const {std::string msString = reportMicroseconds(); printf("[TIMER] %s\n", msString.c_str());}

private:
  
#if defined(_WIN32)
	LARGE_INTEGER total_us_;
	LARGE_INTEGER start_;
	LARGE_INTEGER end_;
	LARGE_INTEGER freq;
#else
  double total_us_;
  timespec start_;
  timespec end_;
  clockid_t clock_;
#endif
};

class ScopedTimer
{
public:
  HighResTimer hrt_;
  ScopedTimer(const std::string& description = "ScopedTimer");
  ~ScopedTimer();
};


#endif // HIGH_RES_TIMER_H

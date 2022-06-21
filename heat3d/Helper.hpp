#ifndef __HELPER_HPP__
#define __HELPER_HPP__

#include <string>
#include <sstream>
#include <iomanip>

std::string zfill(int n, int length);

std::string zfill(int n, int length) {
  std::ostringstream s;
  s << std::setfill('0') << std::setw(length) << n;
  return s.str();
}

#endif

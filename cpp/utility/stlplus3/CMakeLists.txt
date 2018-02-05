project(stlplus3)

include_directories(./)
set(stlplus3_sources
  file_system.cpp
  portability_fixes.cpp
  wildcard.cpp)

add_library(stlplus3 STATIC ${stlplus3_sources})

add_library(RModelLib SHARED Variables.cc BESRModel.cc DecayChannel.cc BESRParser.cc Plotting.cc RandomSampling.cc StatTools.cc PerturbativeQCD.cc Dispersion.cc)

target_link_libraries(RModelLib PUBLIC root_interface)
target_include_directories(RModelLib PUBLIC ${CMAKE_SOURCE_DIR}/include)

set_target_properties(RModelLib PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

add_library(LIB_BERT 
    he.cpp
    nonlinear.cpp
    linear.cpp
    bert_utils.cpp
    bert.cpp
    )

target_link_libraries(LIB_BERT SCI-FloatingPoint)
target_link_libraries(LIB_BERT SCI-HE)

# add_executable(BERT_NONLINEAR "test_nonlinear.cpp")
# target_link_libraries(BERT_NONLINEAR NONLINEAR)

add_executable(PPNLP_BERT "ppnlp_bert.cpp")
target_link_libraries(PPNLP_BERT LIB_BERT)

add_library(LIB_BERT_IRON 
    he.cpp
    nonlinear.cpp
    linear.cpp
    bert_utils.cpp
    bert.cpp
    matrix_utils.cpp
    )

target_link_libraries(LIB_BERT_IRON SCI-FloatingPoint)
target_link_libraries(LIB_BERT_IRON SCI-HE)

add_executable(IRON_BERT "iron_bert.cpp")
target_link_libraries(IRON_BERT LIB_BERT_IRON)

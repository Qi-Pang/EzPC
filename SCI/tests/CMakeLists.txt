# find_package(OpenFHE)
# set(CMAKE_CXX_FLAGS "${OpenFHE_CXX_FLAGS} -Wno-error")

# macro (add_test_OT _name)
# 	add_executable(${_name}-OT "test_ring_${_name}.cpp")
#     target_link_libraries(${_name}-OT SCI-OT) 
# endmacro()

# macro (add_test_HE _name)
#     add_executable(${_name}-HE "test_field_${_name}.cpp")
#     target_link_libraries(${_name}-HE SCI-HE) 
# endmacro()

# macro (add_test_float _name)
# 	add_executable(${_name}-float "test_float_${_name}.cpp")
#     target_link_libraries(${_name}-float SCI-FloatingPoint) 
# endmacro()

# add_test_OT(matmul)
# add_test_OT(value_extension)
# add_test_OT(truncation)
# add_test_OT(relu)
# add_test_OT(argmax)
# add_test_OT(aux_protocols)
# add_test_OT(maxpool)
# add_test_OT(hadamard_product)
# add_test_OT(sigmoid)
# add_test_OT(exp)
# add_test_OT(tanh)
# add_test_OT(sqrt)

# add_test_HE(relu)
# add_test_HE(maxpool)
# add_test_HE(argmax)
# add_test_HE(conv)
# add_test_HE(fc)
# add_test_HE(elemwise_prod)

# add_test_float(bench_op)
# add_test_float(primitive)
# add_test_float(compound)

# set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
# find_package(MPFR 2.3.0 REQUIRED)
# add_executable(math-float "test_float_math.cpp")
# target_include_directories(math-float PUBLIC ${MPFR_INCLUDES}) 
# target_link_libraries(math-float SCI-FloatingPoint ${MPFR_LIBRARIES}) 

# # add_subdirectory(GC)
add_subdirectory(activation)

# macro(add_test_secfloat name)
# 	add_executable(${name}-secfloatml "test_floatml_${name}.cpp")
# 	target_link_libraries(${name}-secfloatml SCI-SecfloatML)
# endmacro()

# macro(add_test_beacon name)
# 	add_executable(${name}-beacon "test_floatml_${name}.cpp")
#     target_link_libraries(${name}-beacon SCI-Beacon)
# endmacro()

# macro(add_test_floatml name)
# 	add_test_secfloat(${name})
# 	add_test_beacon(${name})
# endmacro()

# add_test_floatml(vsum)
# add_test_floatml(dotprod)
# add_test_floatml(matmul)
# add_test_floatml(sigmoid)
# add_test_floatml(softmax)

# include_directories( ${OPENMP_INCLUDES} )
# include_directories( ${OpenFHE_INCLUDE} )
# include_directories( ${OpenFHE_INCLUDE}/third-party/include )
# include_directories( ${OpenFHE_INCLUDE}/core )
# include_directories( ${OpenFHE_INCLUDE}/pke )

# link_directories( ${OpenFHE_LIBDIR} )
# link_directories( ${OPENMP_LIBRARIES} )
# set( CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS} )
# link_libraries( ${OpenFHE_SHARED_LIBRARIES} )

# add_executable(BERT-HE-SEAL "test_bert_matmul_seal.cpp")
# target_link_libraries(BERT-HE-SEAL SCI-HE)

# add_executable(BERT-HE-CIPHER-EFFICIENT "test_bert_efficient.cpp")
# target_link_libraries(BERT-HE-CIPHER-EFFICIENT SCI-HE)

# add_executable(BERT-HE-CIPHER-EFFICIENT-BFV "test_bert_efficient_seal.cpp")
# target_link_libraries(BERT-HE-CIPHER-EFFICIENT-BFV SCI-HE)

# add_executable(BERT-HE-CIPHER-EFFICIENT-CKKS "test_bert_efficient_seal_ckks.cpp")
# target_link_libraries(BERT-HE-CIPHER-EFFICIENT-CKKS SCI-HE)

# add_executable(BERT-HE-IRON "test_bert_iron.cpp")
# target_link_libraries(BERT-HE-IRON SCI-HE)

# add_executable(BERT-HE-CTPT "test_bert_ctpt.cpp")
# target_link_libraries(BERT-HE-CTPT SCI-HE)
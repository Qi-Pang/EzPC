macro (add_test_IRON _name)
	add_executable(IRON-${_name} "iron_${_name}.cpp")
    target_link_libraries(IRON-${_name} SCI-Math) 
    target_link_libraries(IRON-${_name} SCI-FloatingPoint) 
endmacro()

macro (add_test_PPNLP _name)
	add_executable(PPNLP-${_name} "ppnlp_${_name}.cpp")
    target_link_libraries(PPNLP-${_name} SCI-FloatingPoint)
    target_link_libraries(PPNLP-${_name} SCI-Math) 
endmacro()

add_test_IRON(layer_norm)
add_test_IRON(softmax)
add_test_IRON(gelu)
add_test_IRON(tanh)

add_test_PPNLP(softmax)
add_test_PPNLP(gelu)
add_test_PPNLP(tanh)
add_test_PPNLP(layer_norm)


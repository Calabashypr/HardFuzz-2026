if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist ,please check whether the cann package is installed")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(ascendc_kernels_${RUN_MODE} SHARED ${KERNEL_FILES})

ascendc_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
    HAVE_WORKSPACE
    HAVE_TILING
)

# Isolate device symbols before merging.
set(HF_KERNEL ascendc_kernels_${RUN_MODE})
if(TARGET ${HF_KERNEL}_aic_device AND TARGET ${HF_KERNEL}_aiv_device AND TARGET ${HF_KERNEL}_merge_obj)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    find_program(HF_OBJCOPY llvm-objcopy
        HINTS ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ccec_compiler/bin
              ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/ccec_compiler/bin
              ${ASCEND_CANN_PACKAGE_PATH}/compiler/ccec_compiler/bin
        NO_DEFAULT_PATH)
    if(NOT HF_OBJCOPY)
        message(FATAL_ERROR "CANN llvm-objcopy is required to isolate mixed-core symbols")
    endif()
    add_custom_target(${HF_KERNEL}_isolate_symbols
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/isolate_device_symbols.py
            --objcopy ${HF_OBJCOPY}
            ${CMAKE_CURRENT_BINARY_DIR}/${HF_KERNEL}_aic_device_dir/device.o
            ${CMAKE_CURRENT_BINARY_DIR}/${HF_KERNEL}_aiv_device_dir/device.o
        DEPENDS ${HF_KERNEL}_aic_device ${HF_KERNEL}_aiv_device
        VERBATIM)
    add_dependencies(${HF_KERNEL}_merge_obj ${HF_KERNEL}_isolate_symbols)
endif()

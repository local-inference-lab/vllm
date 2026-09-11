# Apply only to FetchContent-owned sources, never a caller's source override.
# Reconfiguration may repeat the patch step without refetching the dependency.
if(NOT DEFINED SOURCE_DIR OR NOT DEFINED PATCH_FILE)
  message(FATAL_ERROR "FlashKDA patching requires SOURCE_DIR and PATCH_FILE")
endif()
find_package(Git REQUIRED)
execute_process(
  COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${PATCH_FILE}"
  WORKING_DIRECTORY "${SOURCE_DIR}"
  RESULT_VARIABLE already_applied
  OUTPUT_QUIET ERROR_QUIET)
if(already_applied EQUAL 0)
  return()
endif()
execute_process(
  COMMAND "${GIT_EXECUTABLE}" apply --check "${PATCH_FILE}"
  WORKING_DIRECTORY "${SOURCE_DIR}"
  RESULT_VARIABLE compatible
  OUTPUT_VARIABLE check_output ERROR_VARIABLE check_error)
if(NOT compatible EQUAL 0)
  message(FATAL_ERROR
    "FlashKDA sources do not match the pinned packed-checkpoint patch: "
    "${check_output}${check_error}")
endif()
execute_process(
  COMMAND "${GIT_EXECUTABLE}" apply "${PATCH_FILE}"
  WORKING_DIRECTORY "${SOURCE_DIR}"
  RESULT_VARIABLE applied
  OUTPUT_VARIABLE patch_output ERROR_VARIABLE patch_error)
if(NOT applied EQUAL 0)
  message(FATAL_ERROR
    "Unable to apply the FlashKDA packed-checkpoint patch: "
    "${patch_output}${patch_error}")
endif()

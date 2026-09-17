#define _GNU_SOURCE
#include <cuda.h>
#include <dlfcn.h>
#include <pthread.h>

/*
 * vLLM's protected cuMem allocator predates shareable-handle allocation
 * properties. Interpose only cuMemCreate and require POSIX-FD capability.
 * Current drivers may make vLLM prefer FABRIC handles, which cannot be exported
 * as POSIX FDs. Allocation, mapping, and ownership stay entirely in the
 * original vLLM/CUDA implementation.
 */
typedef CUresult (*cuMemCreate_fn)(CUmemGenericAllocationHandle *, size_t,
                                  const CUmemAllocationProp *,
                                  unsigned long long);

static cuMemCreate_fn real_cuMemCreate;
static pthread_once_t resolve_once = PTHREAD_ONCE_INIT;

static void resolve_driver_symbol(void) {
  /*
   * CUDA is loaded in the extension's local dependency scope, so RTLD_NEXT
   * alone can miss it. Resolve from the actual driver DSO explicitly.
   */
  void *driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (driver != NULL) {
    real_cuMemCreate = (cuMemCreate_fn)dlsym(driver, "cuMemCreate");
  }
}

CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                     const CUmemAllocationProp *prop,
                     unsigned long long flags) {
  pthread_once(&resolve_once, resolve_driver_symbol);
  if (real_cuMemCreate == NULL) {
    return CUDA_ERROR_NOT_INITIALIZED;
  }
  if (prop == NULL ||
      prop->requestedHandleTypes == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    return real_cuMemCreate(handle, size, prop, flags);
  }
  CUmemAllocationProp shareable = *prop;
  shareable.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  return real_cuMemCreate(handle, size, &shareable, flags);
}

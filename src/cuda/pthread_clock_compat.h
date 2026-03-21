#pragma once

// Provide declarations for pthread_*clock* APIs that libstdc++'s
// <mutex>/<condition_variable> expect, even when glibc's pthread.h
// does not declare them because _GNU_SOURCE is not defined. This
// allows us to compile with -U_GNU_SOURCE to avoid glibc 2.38+'s
// rsqrt/rsqrtf C23 math declarations conflicting with CUDA's
// math_functions.h, while still satisfying libstdc++.

#include <pthread.h>
#include <time.h>

extern "C" {

int pthread_mutex_clocklock(pthread_mutex_t* __restrict mutex,
                            clockid_t clockid,
                            const struct timespec* __restrict abstime);

int pthread_cond_clockwait(pthread_cond_t* __restrict cond,
                           pthread_mutex_t* __restrict mutex,
                           clockid_t clockid,
                           const struct timespec* __restrict abstime);

}  // extern "C"

